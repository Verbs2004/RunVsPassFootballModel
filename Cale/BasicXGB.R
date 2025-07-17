# ───────────────────────────────────────────────────────────────
# Simplified NFL Play Prediction with XGBoost & PRI Metric
# ───────────────────────────────────────────────────────────────

# 0) Load libraries
suppressPackageStartupMessages({
  library(nflreadr)
  library(data.table)
  library(xgboost)
  library(pROC)
  library(stringr)
  library(doParallel)
})

# 1) Setup parallel processing (optional)
n_cores <- parallel::detectCores()
cat("Using", n_cores, "cores for parallel processing.\n")
cl <- makeCluster(n_cores)
doParallel::registerDoParallel(cl)
setDTthreads(0)  # Let data.table use all cores

# 2) Load or cache data
cache_file <- "nfl_data_cache.rds"

if (file.exists(cache_file)) {
  cat("Loading cached data...\n")
  data_list <- readRDS(cache_file)
  pbp <- data_list$pbp
  sched <- data_list$sched
  parts <- data_list$parts
} else {
  cat("Loading fresh data...\n")
  pbp <- setDT(load_pbp(2016:2023))
  sched <- setDT(load_schedules(2016:2023))[, .(game_id, roof, temp, wind)]
  
  parts <- tryCatch({
    setDT(load_participation(2016:2023))[, .(old_game_id, play_id, defenders_in_box, offense_personnel, defense_personnel, defense_players)]
  }, error = function(e) {
    cat("Warning: participation data not available; proceeding without it.\n")
    data.table()
  })
  
  saveRDS(list(pbp = pbp, sched = sched, parts = parts), cache_file)
  cat("Data cached for future use.\n")
}

# Ensure absolute_yardline_number column exists
if (!"absolute_yardline_number" %in% colnames(pbp)) {
  pbp[, absolute_yardline_number := fifelse(side_of_field == posteam, 100 - yardline_100, yardline_100)]
}

# 3) Filter and join key data
cat("Filtering plays and joining schedule & participation data...\n")

df <- pbp[
  play_type %in% c("run", "pass") &
    !is.na(down) & down %in% 1:4 &
    !is.na(ydstogo) & ydstogo >= 1 & ydstogo <= 50 &
    !is.na(yardline_100) & yardline_100 >= 1 & yardline_100 <= 99 &
    !is.na(score_differential) &
    !is.na(game_seconds_remaining) &
    !is.na(posteam) & !is.na(defteam)
]

# Join schedule info
if (exists("sched") && nrow(sched) > 0) {
  df[sched, on = "game_id", `:=`(sched_roof = i.roof, sched_temp = i.temp, sched_wind_speed = i.wind)]
}

# Join participation info if available
if (exists("parts") && nrow(parts) > 0) {
  setnames(parts, c("offense_personnel", "defense_personnel"), c("parts_off_pers", "parts_def_pers"), skip_absent = TRUE)
  parts[, old_game_id := as.character(old_game_id)]
  df[, old_game_id := as.character(old_game_id)]
  df[parts, on = c("old_game_id", "play_id"), `:=`(
    parts_off_pers = i.parts_off_pers,
    parts_def_pers = i.parts_def_pers,
    defenders_in_box = i.defenders_in_box,
    defense_players = i.defense_players
  )]
}

# 4) Feature engineering
cat("Feature engineering...\n")
setorder(df, game_id, fixed_drive, play_id)

df[, drive_play_idx := 1:.N, by = .(game_id, posteam)]
df[, `:=`(
  is_pass = as.integer(play_type == "pass"),
  is_first_play_of_drive = fifelse(drive_play_idx == 1, 1, 0),
  prev_play_was_pass = shift(fifelse(play_type == "pass", 1, 0), 1, fill = 0),
  yards_gained_on_prev_play = shift(yards_gained, 1, fill = 0)
), by = .(game_id, fixed_drive)]

# Parse personnel info, fallback to defaults if missing
df[, offense_personnel := if("parts_off_pers" %in% colnames(df)) fcoalesce(parts_off_pers, "1 RB, 1 TE, 3 WR") else "1 RB, 1 TE, 3 WR"]
df[, defense_personnel := if("parts_def_pers" %in% colnames(df)) fcoalesce(parts_def_pers, "4 DL, 3 LB, 4 DB") else "4 DL, 3 LB, 4 DB"]

df[, `:=`(
  RB_off_P = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*RB)")),
  TE_off_P = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*TE)")),
  WR_off_P = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*WR)")),
  DL_def_P = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*DL)")),
  LB_def_P = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*LB)")),
  DB_def_P = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*DB)"))
)]

# Replace NA with 0
cols_to_fix <- c("RB_off_P", "TE_off_P", "WR_off_P", "DL_def_P", "LB_def_P", "DB_def_P")
for (col in cols_to_fix) {
  df[is.na(get(col)), (col) := 0L]
}

df[, defenders_in_box := fcoalesce(defenders_in_box, DL_def_P + LB_def_P, 7L)]

# Add some binary features
df[, `:=`(
  heavy_set = as.integer((RB_off_P + TE_off_P) >= 3),
  empty_back = as.integer(RB_off_P == 0),
  is_nickel = as.integer(DB_def_P == 5),
  is_dime = as.integer(DB_def_P >= 6)
)]

# 5) Prepare modeling dataset
cat("Preparing modeling dataset...\n")
modeling_vars <- c("season", "game_id", "play_id", "is_pass", "down", "ydstogo", "qtr", "quarter_seconds_remaining", "yardline_100", "score_differential")
df_model <- df[, ..modeling_vars]
df_model <- na.omit(df_model)

# Split data into training and test sets
train_data <- df_model[season < 2023]
test_data  <- df_model[season == 2023]

y_train <- train_data$is_pass
y_test  <- test_data$is_pass

# Exclude non-numeric columns from matrix (game_id, play_id)
train_matrix <- as.matrix(train_data[, !c("is_pass", "game_id", "play_id"), with = FALSE])
test_matrix  <- as.matrix(test_data[, !c("is_pass", "game_id", "play_id"), with = FALSE])

dtrain <- xgb.DMatrix(data = train_matrix, label = y_train)
dtest  <- xgb.DMatrix(data = test_matrix, label = y_test)

# 6) Train XGBoost model
cat("Training XGBoost model...\n")
params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 4,
  subsample = 0.8,
  colsample_bytree = 0.8,
  tree_method = "hist",
  nthread = n_cores
)

watchlist <- list(train = dtrain, test = dtest)
set.seed(42)
xgb_model <- xgb.train(params = params, data = dtrain, nrounds = 500, watchlist = watchlist,
                       early_stopping_rounds = 30, verbose = 2)

cat("Model training completed. Best iteration:", xgb_model$best_iteration, "\n")

# 7) Evaluate model performance
cat("Evaluating model performance...\n")
pred_probs <- predict(xgb_model, dtest, ntreelimit = xgb_model$best_iteration)
pred_classes <- as.integer(pred_probs > 0.5)

conf_matrix <- table(Predicted = pred_classes, Actual = y_test)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
auc_val <- auc(roc(y_test, pred_probs))

cat("Accuracy:", round(accuracy, 4), "\n")
cat("AUC:", round(auc_val, 4), "\n")
print(conf_matrix)

library(data.table)
library(xgboost)
library(pROC)

# Prepare a list to store predictions keyed by year
predictions_by_year <- list()

for (test_year in 2019:2023) {
  cat("\n==== Training on 3 seasons before", test_year, ", testing on", test_year, "====\n")
  
  train_data <- df_model[season %in% c(test_year - 1, test_year - 2, test_year - 3)]
  test_data  <- df_model[season == test_year]
  
  y_train <- train_data$is_pass
  y_test  <- test_data$is_pass
  
  cols_to_remove <- c("game_id", "is_pass")
  train_matrix <- as.matrix(train_data[, !cols_to_remove, with = FALSE])
  test_matrix  <- as.matrix(test_data[, !cols_to_remove, with = FALSE])
  
  dtrain <- xgb.DMatrix(data = train_matrix, label = y_train)
  dtest  <- xgb.DMatrix(data = test_matrix, label = y_test)
  
  xgb_model <- xgb.train(
    params = list(
      objective = "binary:logistic",
      eval_metric = "auc",
      eta = 0.05,
      max_depth = 6,
      subsample = 0.75,
      colsample_bytree = 0.75
    ),
    data = dtrain,
    nrounds = 500,
    watchlist = list(train = dtrain, test = dtest),
    early_stopping_rounds = 30,
    verbose = 0
  )
  
  test_probs <- predict(xgb_model, dtest)
  
  # Save predictions with keys
  pred_dt <- test_data[, .(game_id, play_id)]
  pred_dt[, pred_pass_prob := test_probs]
  predictions_by_year[[as.character(test_year)]] <- pred_dt
  
  # Evaluate & print
  test_preds <- as.integer(test_probs > 0.5)
  conf_mat <- table(Predicted = test_preds, Actual = y_test)
  accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
  auc_score <- auc(roc(y_test, test_probs, quiet = TRUE))
  
  cat("AUC:", round(auc_score, 4), "\n")
  cat("Accuracy:", round(accuracy, 4), "\n")
}

######

library(pROC)

# Store ROC curves
roc_list <- list()
fpr_grid <- seq(0, 1, length.out = 1000)
tpr_matrix <- matrix(NA, nrow = length(fpr_grid), ncol = 0)
auc_values <- numeric()

# Re-loop to extract ROC curves
for (test_year in 2019:2023) {
  test_data <- df_model[season == test_year]
  y_test <- test_data$is_pass
  pred_probs <- predictions_by_year[[as.character(test_year)]]$pred_pass_prob
  
  roc_obj <- roc(y_test, pred_probs, quiet = TRUE)
  auc_values <- c(auc_values, auc(roc_obj))
  roc_list[[as.character(test_year)]] <- roc_obj
  
  # Interpolate TPR at common FPR points
  interp_tpr <- approx(roc_obj$specificities, roc_obj$sensitivities,
                       xout = 1 - fpr_grid, method = "linear", rule = 2)$y
  tpr_matrix <- cbind(tpr_matrix, interp_tpr)
}

# Calculate mean TPR
mean_tpr <- rowMeans(tpr_matrix)
mean_auc <- mean(auc_values)

library(ggplot2)

# Create a data frame for plotting
roc_df <- data.frame(
  fpr = fpr_grid,
  tpr = mean_tpr
)

# Create the plot
ggplot(roc_df, aes(x = fpr, y = tpr)) +
  geom_line(color = "#1c6ef2", size = 1.2) +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray") +
  labs(
    title = "Basic Model",
    x = "False Positive Rate",
    y = "True Positive Rate"
  ) +
  annotate("text", x = 0.65, y = 0.1,
           label = paste("Mean AUC =", round(mean_auc, 4)),
           hjust = 0, size = 4, color = "#1c6ef2") +
  theme_minimal()
######

# Combine all prediction tables into one
all_preds <- rbindlist(lapply(names(predictions_by_year), function(year) {
  dt <- predictions_by_year[[year]]
  dt[, season := as.integer(year)]
  return(dt)
}))

# Merge predictions into full df by game_id, play_id, and season
df <- merge(df, all_preds, by = c("season", "game_id", "play_id"), all.x = TRUE, sort = FALSE)

# Handle missing predicted probabilities (if any)
df[is.na(pred_pass_prob), pred_pass_prob := 0.5]  # neutral prediction for missing

library(stringr)

evaluate_pass_rushers <- function(data_season) {
  epsilon <- 1e-10
  data_season[, actual_pass := as.integer(play_type == "pass")]
  data_season[, pred_pass_prob_clipped := pmin(pmax(pred_pass_prob, epsilon), 1 - epsilon)]
  data_season[, surprisal := -log(ifelse(actual_pass == 1, pred_pass_prob_clipped, 1 - pred_pass_prob_clipped))]
  
  # Clean defense_players strings
  data_season[, defense_players := str_trim(defense_players)]
  data_season[, defense_players := gsub(";+$", "", defense_players)]
  
  # Explode defensive players with surprisal
  def_players_long <- data_season[!is.na(defense_players) & defense_players != "", .(
    gsis_id = unlist(strsplit(defense_players, ";")),
    surprisal = rep(surprisal, lengths(strsplit(defense_players, ";")))
  ), by = .(game_id, play_id)]
  
  player_surprisal_exposure <- def_players_long[, .(weighted_pass_rush_snaps = sum(surprisal, na.rm = TRUE)), by = gsis_id]
  
  sacks <- data_season[sack == 1 & !is.na(sack_player_id), .(gsis_id = sack_player_id, surprisal = surprisal)]
  sacks_weighted <- sacks[, .(weighted_sacks = sum(surprisal, na.rm = TRUE)), by = gsis_id]
  
  qb_hit_cols <- grep("^qb_hit_\\d+_player_id$", colnames(data_season), value = TRUE)
  qb_hits <- rbindlist(lapply(qb_hit_cols, function(col) {
    data_season[qb_hit == 1 & !is.na(get(col)), .(gsis_id = get(col), surprisal = surprisal)]
  }))
  qb_hits_weighted <- qb_hits[, .(weighted_qb_hits = sum(surprisal, na.rm = TRUE)), by = gsis_id]
  player_snap_counts <- def_players_long[, .(raw_pass_rush_snaps = .N), by = gsis_id]
  
  disruption_summary <- merge(player_surprisal_exposure, sacks_weighted, by = "gsis_id", all.x = TRUE)
  disruption_summary <- merge(disruption_summary, qb_hits_weighted, by = "gsis_id", all.x = TRUE)
  disruption_summary <- merge(disruption_summary, player_snap_counts, by = "gsis_id", all.x = TRUE)
  disruption_summary[is.na(weighted_sacks), weighted_sacks := 0]
  disruption_summary[is.na(weighted_qb_hits), weighted_qb_hits := 0]
  
  disruption_summary[, disruption_rate := (weighted_sacks + weighted_qb_hits) / weighted_pass_rush_snaps]
  
  # Load roster info for season (tryCatch for safety)
  rosters <- tryCatch({
    setDT(load_rosters(unique(data_season$season)))[, .(gsis_id, full_name, position, team)]
  }, error = function(e) {
    cat("Warning: Could not load roster data.\n")
    data.table()
  })
  
  disruption_summary <- merge(disruption_summary, rosters, by = "gsis_id", all.x = TRUE)
  
  pass_rusher_positions <- c("DE", "DT", "EDGE", "OLB", "ILB", "LB", "NT", "DL")
  disruption_summary <- disruption_summary[position %in% pass_rusher_positions]
  
  disruption_summary <- disruption_summary[raw_pass_rush_snaps >= 300]  # Threshold to reduce noise
  
  setorder(disruption_summary, -disruption_rate)
  
  disruption_summary[, .(
    Player = full_name,
    Team = team,
    Position = position,
    Weighted_Sacks = round(weighted_sacks, 3),
    Weighted_QB_Hits = round(weighted_qb_hits, 3),
    Pass_Rush_Snaps = raw_pass_rush_snaps,
    Disruption_Rate = round(disruption_rate, 4)
  )]
}

# Run evaluation for each test year and print results
for (yr in 2023) {
  cat("\n=== Surprisal-Weighted Pass Rushers for Season", yr, "===\n")
  season_data <- df[season == yr]
  res <- evaluate_pass_rushers(season_data)
  print(head(res, 30))
}







