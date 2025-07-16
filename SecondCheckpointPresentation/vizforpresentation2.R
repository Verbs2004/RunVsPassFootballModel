# ───────────────────────────────────────────────────────────────
# V8 ULTRA-OPTIMIZED NFL Play Prediction & Enhanced Pass Rusher Evaluation
# With Pass Rush Informativeness (PRI) Metric & Historical Analysis Loop
# ───────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# 0) Libraries & High-Performance Setup
# ───────────────────────────────────────────────────────────────
cat("Loading packages and setting up high-performance environment...\n")
suppressPackageStartupMessages({
  library(nflreadr)
  library(data.table)
  library(xgboost)
  library(fastDummies)
  library(pROC)
  library(knitr)
  library(doParallel)
  library(stringr)
  library("ggrepel")
})

# --- Performance Setup ---
# For Intel Ultra 7, using all cores is generally best.
# You can experiment by setting this to the number of P-Cores (e.g., 6 for the 155H)
# to see if it reduces overhead.
# n_cores <- 6 
n_cores <- detectCores()
setDTthreads(0) 
cat("Using", n_cores, "cores for parallel processing.\n")

# Setup parallel backend
cl <- makeCluster(n_cores)
registerDoParallel(cl)

# ───────────────────────────────────────────────────────────────
# 1) Ultra-Fast Data Loading with AUTOMATIC CACHING
# ───────────────────────────────────────────────────────────────
cat("Loading core data tables (2016-2023)...\n")
cache_file <- "nfl_data_cache.rds"

if (file.exists(cache_file)) {
  cat("Loading data from local cache file. To refresh, delete 'nfl_data_cache.rds'.\n")
  data_list <- readRDS(cache_file)
  pbp <- data_list$pbp
  sched <- data_list$sched
  parts <- data_list$parts
} else {
  cat("No cache found. Loading fresh data from nflreadr (this may take a while)...\n")
  pbp <- setDT(load_pbp(2016:2023))
  sched <- setDT(load_schedules(2016:2023))[, .(game_id, roof, temp, wind)]
  
  parts <- tryCatch({
    setDT(load_participation(2016:2023))[, .(old_game_id, play_id, defenders_in_box, offense_personnel, defense_personnel, defense_players)]
  }, error = function(e) {
    cat("Warning: Could not load participation data. It will be estimated.\n")
    data.table()
  })
  
  cat("Saving data to cache for future runs...\n")
  saveRDS(list(pbp = pbp, sched = sched, parts = parts), cache_file)
}

# Ensure absolute_yardline_number exists
if (!"absolute_yardline_number" %in% names(pbp)) {
  pbp[, absolute_yardline_number := fifelse(side_of_field == posteam, 100 - yardline_100, yardline_100)]
}

# ───────────────────────────────────────────────────────────────
# 2) High-Speed Data Joins with data.table
# ───────────────────────────────────────────────────────────────
cat("Filtering and joining data sources...\n")
df_base <- pbp[play_type %in% c("run", "pass") & !is.na(down) & down %in% 1:4 & !is.na(ydstogo) & ydstogo >= 1 & ydstogo <= 50 & !is.na(yardline_100) & yardline_100 >= 1 & yardline_100 <= 99 & !is.na(score_differential) & !is.na(game_seconds_remaining) & !is.na(posteam) & !is.na(defteam)]

# Join schedule
if (!is.null(sched) && nrow(sched) > 0) {
  df_base[sched, on = "game_id", `:=`(
    sched_roof = i.roof, 
    sched_temp = i.temp, 
    sched_wind_speed = i.wind
  )]
}

# Join participation data
if (!is.null(parts) && nrow(parts) > 0) {
  setnames(parts, c("offense_personnel", "defense_personnel"), c("parts_off_pers", "parts_def_pers"), skip_absent = TRUE)
  parts[, old_game_id := as.character(old_game_id)]
  df_base[, old_game_id := as.character(old_game_id)]
  df_base[parts, on = c("old_game_id", "play_id"), `:=`(
    parts_off_pers = i.parts_off_pers,
    parts_def_pers = i.parts_def_pers,
    defenders_in_box = i.defenders_in_box,
    defense_players = i.defense_players
  )]
}

# ───────────────────────────────────────────────────────────────
# 3) Accelerated Feature Engineering with data.table
# ───────────────────────────────────────────────────────────────
cat("Performing high-speed feature engineering...\n")
setorder(df_base, game_id, fixed_drive, play_id)

# Create target and sequential features in-place
df_base[, drive_play_idx := 1:.N, by = .(game_id, posteam)]
df_base[, `:=`(
  is_pass = as.numeric(play_type == "pass"),
  is_first_play_of_drive = fifelse(drive_play_idx == 1, 1, 0),
  prev_play_was_pass = shift(fifelse(play_type == "pass", 1, 0), 1, fill = 0),
  yards_gained_on_prev_play = shift(yards_gained, 1, fill = 0)
), by = .(game_id, fixed_drive)]

# --- Vectorized Personnel Parsing (Much Faster) ---
df_base[, offense_personnel := if("parts_off_pers" %in% names(df_base)) fcoalesce(parts_off_pers, "1 RB, 1 TE, 3 WR") else "1 RB, 1 TE, 3 WR"]
df_base[, defense_personnel := if("parts_def_pers" %in% names(df_base)) fcoalesce(parts_def_pers, "4 DL, 3 LB, 4 DB") else "4 DL, 3 LB, 4 DB"]

df_base[, `:=`(
  RB_off_P = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*RB)")),
  TE_off_P = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*TE)")),
  WR_off_P = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*WR)")),
  DL_def_P = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*DL)")),
  LB_def_P = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*LB)")),
  DB_def_P = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*DB)")),
  FB_off_P = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*FB)"))
)]
# Fill NAs that result from parsing
df_base[is.na(RB_off_P), RB_off_P := 0L][is.na(TE_off_P), TE_off_P := 0L][is.na(WR_off_P), WR_off_P := 0L][is.na(FB_off_P), FB_off_P := 0L]
df_base[is.na(DL_def_P), DL_def_P := 0L][is.na(LB_def_P), LB_def_P := 0L][is.na(DB_def_P), DB_def_P := 0L]

df_base[, defenders_in_box := fcoalesce(defenders_in_box, DL_def_P + LB_def_P, 7L)]

# Advanced personnel & contextual features
df_base[, `:=`(
  heavy_set = as.numeric((RB_off_P + TE_off_P) >= 3),
  empty_back = as.numeric(RB_off_P == 0),
  is_nickel = as.numeric(DB_def_P == 5),
  is_dime = as.numeric(DB_def_P >= 6),
  skill_diff_P = WR_off_P - DB_def_P,
  box_advantage_off = (5 + TE_off_P) - defenders_in_box,
  ep = fcoalesce(as.numeric(ep), 0),
  spread_line = fcoalesce(as.numeric(spread_line), 0),
  total_line = fcoalesce(as.numeric(total_line), mean(as.numeric(total_line), na.rm = TRUE)),
  goal_to_go = fcoalesce(as.numeric(goal_to_go), 0),
  play_clock_at_snap = fcoalesce(as.numeric(play_clock), 15),
  two_minute_drill = as.numeric((qtr == 2 | qtr == 4) & quarter_seconds_remaining <= 120),
  four_minute_drill = as.numeric(qtr == 4 & quarter_seconds_remaining <= 240 & score_differential > 0),
  shotgun_flag = as.numeric(fcoalesce(shotgun, 0) == 1),
  no_huddle_flag = as.numeric(fcoalesce(no_huddle, 0) == 1),
  wp = fcoalesce(fifelse(posteam == home_team, home_wp, away_wp), 0.5),
  late_desperation = as.numeric(game_seconds_remaining < 300 & fcoalesce(fifelse(posteam == home_team, home_wp, away_wp), 0.5) < 0.2),
  trailing_big = as.numeric(score_differential <= -14),
  leading_big = as.numeric(score_differential >= 14)
)]

# Categorical features
df_base[, `:=`(
  timeout_situation = fcase(
    posteam_timeouts_remaining > defteam_timeouts_remaining, "Offense Advantage",
    posteam_timeouts_remaining < defteam_timeouts_remaining, "Defense Advantage",
    posteam_timeouts_remaining == 0 & defteam_timeouts_remaining == 0, "None Left",
    default = "Equal Timeouts"
  ),
  form_cat = fifelse(fcoalesce(shotgun, 0) == 1, "Shotgun", "Other"),
  wp_tier = fcase(
    wp < 0.2, "desperate", 
    wp < 0.4, "low",
    wp > 0.8, "dominant", 
    wp > 0.6, "high",
    default = "medium"
  ),
  yards_to_goal_bin = fcase(
    yardline_100 <= 4, "goalline", 
    yardline_100 <= 10, "redzone_fringe",
    yardline_100 <= 20, "redzone", 
    yardline_100 <= 50, "own_territory",
    default = "backed_up"
  ),
  score_situation = fcase(
    leading_big == 1, "leading_big", 
    trailing_big == 1, "trailing_big",
    abs(score_differential) <= 4, "close",
    score_differential > 4, "leading", 
    default = "trailing"
  ),
  down_dist_cat = fcase(
    down == 1, "first",
    down == 2 & ydstogo <= 3, "second_short", 
    down == 2 & ydstogo >= 8, "second_long",
    down == 3 & ydstogo <= 3, "third_short", 
    down == 3 & ydstogo >= 8, "third_long",
    down == 4 & ydstogo <= 3, "fourth_short", 
    down == 4 & ydstogo >= 8, "fourth_long",
    default = "medium_yardage"
  )
)]

pbp[, run_situation := fcase(
  down == 1, "1st_down",
  down == 2 & ydstogo <= 3, "2nd_short",
  down == 2 & ydstogo >= 8, "2nd_long",
  down == 3 & ydstogo <= 3, "3rd_short",
  down == 3 & ydstogo >= 8, "3rd_long",
  score_differential > 7, "leading",
  score_differential < -7, "trailing",
  yardline_100 <= 20, "red_zone",
  default = "other"
)]

situational_run_rates <- pbp[
  !is.na(run_situation) & season < 2023,
  .(run_rate = mean(play_type == "run", na.rm = TRUE)),
  by = .(season, posteam, run_situation)
]

get_lagged_rate <- function(dt) {
  out <- dt[, .(posteam, season, run_situation, run_rate)]
  out[, lagged_run_rate := NA_real_]
  all_seasons <- sort(unique(out$season))
  
  for (s in all_seasons) {
    prev <- out[season < s, .(lagged_run_rate = mean(run_rate, na.rm = TRUE)),
                by = .(posteam, run_situation)]
    out[season == s, lagged_run_rate := prev[.SD, on = .(posteam, run_situation), lagged_run_rate], 
        .SDcols = c("posteam", "run_situation")]
  }
  return(out)
}

lagged_rates <- get_lagged_rate(situational_run_rates)

df_base[, run_situation := fcase(
  down == 1, "1st_down",
  down == 2 & ydstogo <= 3, "2nd_short",
  down == 2 & ydstogo >= 8, "2nd_long",
  down == 3 & ydstogo <= 3, "3rd_short",
  down == 3 & ydstogo >= 8, "3rd_long",
  score_differential > 7, "leading",
  score_differential < -7, "trailing",
  yardline_100 <= 20, "red_zone",
  default = "other"
)]

df_base <- merge(
  df_base, 
  lagged_rates[, .(posteam, season, run_situation, lagged_run_rate)], 
  by.x = c("posteam", "season", "run_situation"),
  by.y = c("posteam", "season", "run_situation"),
  all.x = TRUE
)

# ───────────────────────────────────────────────────────────────
# 4) Create Modeling and Evaluation Datasets (Stripped Down)
# ───────────────────────────────────────────────────────────────
cat("Creating simplified modeling dataset...\n")

modeling_vars <- c("season", "game_id", "play_id", "is_pass", "down", "ydstogo", "qtr", "quarter_seconds_remaining", 
                   "yardline_100", "score_differential", "WR_off_P", "lagged_run_rate")

df_simple <- df_base[, ..modeling_vars]
df_simple[, lagged_run_rate := fifelse(is.na(lagged_run_rate), mean(lagged_run_rate, na.rm = TRUE), lagged_run_rate)]
df_simple <- df_simple[complete.cases(df_simple)]


# Train/test split
train_data <- df_simple[season < 2023]
test_data <- df_simple[season == 2023]

# Labels
y_train <- train_data$is_pass
y_test <- test_data$is_pass

# Exclude non-numeric columns from matrix (game_id, play_id)
train_matrix <- as.matrix(train_data[, !c("season", "is_pass", "game_id", "play_id"), with = FALSE])
test_matrix  <- as.matrix(test_data[, !c("season", "is_pass", "game_id", "play_id"), with = FALSE])

cat("Train nrows:", nrow(train_matrix), " Test nrows:", nrow(test_matrix), "\n")
cat("Train label distribution:\n"); print(table(y_train))
cat("Test label distribution:\n"); print(table(y_test))

# Convert to DMatrix
dtrain <- xgb.DMatrix(data = train_matrix, label = y_train)
dtest <- xgb.DMatrix(data = test_matrix, label = y_test)

# ───────────────────────────────────────────────────────────────
# 5) Train Model
# ───────────────────────────────────────────────────────────────
cat("Training XGBoost model with only 7 pre-snap features...\n")

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
xgb_modeldos <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 500,
  watchlist = watchlist,
  early_stopping_rounds = 30,
  verbose = 0
)

cat("Model trained with", xgb_modeldos$best_iteration, "trees.\n")

# ───────────────────────────────────────────────────────────────
# 6) Evaluate
# ───────────────────────────────────────────────────────────────
cat("Evaluating simplified model...\n")

test_preds <- predict(
  xgb_modeldos,
  dtest,
  iteration_range = c(0L, xgb_modeldos$best_iteration)
)
test_class <- as.integer(test_preds > 0.5)

conf_matrix <- table(Predicted = test_class, Actual = y_test)
accuracy <- sum(diag(conf_matrix)) / sum(conf_matrix)
auc_val <- auc(roc(y_test, test_preds, quiet = TRUE))

cat("Accuracy:", round(accuracy, 4), "\n")
cat("AUC:", round(auc_val, 4), "\n")
print(conf_matrix)

# Optional: Feature importance
cat("\n=== FEATURE IMPORTANCE (Top 6) ===\n")
importance <- xgb.importance(model = xgb_modeldos)
print(kable(importance, format = "simple"))

# ───────────────────────────────────────────────────────────────
# 7) Surprisal-Weighted Evaluation Loop (2019–2023)
# ───────────────────────────────────────────────────────────────
library(data.table)
library(xgboost)
library(pROC)

# Prepare a list to store predictions keyed by year
predictions_by_year <- list()

for (test_year in 2019:2023) {
  cat("\n==== Training on 3 seasons before", test_year, ", testing on", test_year, "====\n")
  
  train_data <- df_simple[season %in% c(test_year - 1, test_year - 2, test_year - 3)]
  test_data  <- df_simple[season == test_year]
  
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
  
  pred_dt <- test_data[, .(game_id, play_id)]
  pred_dt[, pred_pass_prob := test_probs]
  predictions_by_year[[as.character(test_year)]] <- pred_dt
  
  test_preds <- as.integer(test_probs > 0.5)
  conf_mat <- table(Predicted = test_preds, Actual = y_test)
  accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
  auc_score <- auc(roc(y_test, test_probs, quiet = TRUE))
  
  cat("AUC:", round(auc_score, 4), "\n")
  cat("Accuracy:", round(accuracy, 4), "\n")
}

# Combine all prediction tables into one
all_preds <- rbindlist(lapply(names(predictions_by_year), function(year) {
  dt <- predictions_by_year[[year]]
  dt[, season := as.integer(year)]
  return(dt)
}))

# Merge predictions into full df by game_id, play_id, and season
df_base<- merge(df_base, all_preds, by = c("season", "game_id", "play_id"), all.x = TRUE, sort = FALSE)

# Handle missing predicted probabilities (if any)
df_base[is.na(pred_pass_prob), pred_pass_prob := 0.5]  # neutral prediction for missing

library(stringr)

evaluate_pass_rushers <- function(data_season) {
  epsilon <- 1e-10
  data_season[, actual_pass := as.integer(play_type == "pass")]
  data_season[, pred_pass_prob_clipped := pmin(pmax(pred_pass_prob, epsilon), 1 - epsilon)]
  data_season[, surprisal := -log(ifelse(actual_pass == 1, pred_pass_prob_clipped, 1 - pred_pass_prob_clipped))]
  
  data_season[, defense_players := str_trim(defense_players)]
  data_season[, defense_players := gsub(";+$", "", defense_players)]
  
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
  
  disruption_summary <- merge(player_surprisal_exposure, sacks_weighted, by = "gsis_id", all.x = TRUE)
  disruption_summary <- merge(disruption_summary, qb_hits_weighted, by = "gsis_id", all.x = TRUE)
  disruption_summary[is.na(weighted_sacks), weighted_sacks := 0]
  disruption_summary[is.na(weighted_qb_hits), weighted_qb_hits := 0]
  
  disruption_summary[, disruption_rate := (weighted_sacks + weighted_qb_hits) / weighted_pass_rush_snaps]
  
  rosters <- tryCatch({
    setDT(load_rosters(unique(data_season$season)))[, .(gsis_id, full_name, position, team)]
  }, error = function(e) {
    cat("Warning: Could not load roster data.\n")
    data.table()
  })
  
  disruption_summary <- merge(disruption_summary, rosters, by = "gsis_id", all.x = TRUE)
  
  pass_rusher_positions <- c("DE", "DT", "EDGE", "OLB", "ILB", "LB", "NT", "DL")
  disruption_summary <- disruption_summary[position %in% pass_rusher_positions]
  disruption_summary <- disruption_summary[weighted_pass_rush_snaps >= 100]
  
  setorder(disruption_summary, -disruption_rate)
  
  disruption_summary[, .(
    Player = full_name,
    Team = team,
    Position = position,
    Weighted_Sacks = round(weighted_sacks, 3),
    Weighted_QB_Hits = round(weighted_qb_hits, 3),
    Weighted_Pass_Rush_Snaps = round(weighted_pass_rush_snaps, 3),
    Disruption_Rate = round(disruption_rate, 4)
  )]
}

for (yr in 2023) {
  cat("\n=== Surprisal-Weighted Pass Rushers for Season", yr, "===\n")
  season_data <- df_base[season == yr]
  res <- evaluate_pass_rushers(season_data)
  print(head(res, 30))
}

# ───────────────────────────────────────────────────────────────
# 8) Advanced Visualizations for Presentation - FIXED VERSION
# ───────────────────────────────────────────────────────────────
library(ggplot2)
library(viridis)
library(plotly)
library(dplyr)
library(gridExtra)
library(RColorBrewer)
library(ggimage)  # For team logos in ggplot
library(scales)

cat("Creating advanced visualizations for presentation...\n")

# Create output directory for plots
if (!dir.exists("plots")) dir.create("plots")

# ───────────────────────────────────────────────────────────────
# 8.1) Model Performance Over Time
# ───────────────────────────────────────────────────────────────

# Calculate yearly AUC scores from our historical loop
yearly_performance <- data.table()
for (test_year in 2019:2023) {
  test_data <- df_base[season == test_year & !is.na(pred_pass_prob)]
  if (nrow(test_data) > 0) {
    auc_score <- auc(roc(test_data$is_pass, test_data$pred_pass_prob, quiet = TRUE))
    yearly_performance <- rbind(yearly_performance, data.table(year = test_year, auc = auc_score))
  }
}

p1 <- ggplot(yearly_performance, aes(x = year, y = auc)) +
  geom_line(color = "#2E86AB", size = 1.5) +
  geom_point(color = "#A23B72", size = 4) +
  geom_text(aes(label = round(auc, 3)), vjust = -0.5, size = 3.5) +
  scale_x_continuous(breaks = 2019:2023) +
  ylim(0.7, 0.8) +
  labs(title = "Model Performance Over Time",
       subtitle = "AUC Score by Season (3-Year Rolling Training Window)",
       x = "Season", y = "AUC Score") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        plot.subtitle = element_text(size = 12))

ggsave("plots/model_performance_timeline.png", p1, width = 10, height = 6, dpi = 300)

# ───────────────────────────────────────────────────────────────
# 8.2) Situational Pass Probability Heat Map
# ───────────────────────────────────────────────────────────────

situation_analysis <- df_base[season == 2023 & !is.na(pred_pass_prob), .(
  actual_pass_rate = mean(is_pass, na.rm = TRUE),
  predicted_pass_rate = mean(pred_pass_prob, na.rm = TRUE),
  play_count = .N
), by = .(down, ydstogo_bin = cut(ydstogo, breaks = c(0, 3, 7, 15, 50), 
                                  labels = c("1-3", "4-7", "8-15", "16+")))]

situation_analysis[, prediction_error := abs(actual_pass_rate - predicted_pass_rate)]

p2 <- ggplot(situation_analysis[play_count >= 20], aes(x = down, y = ydstogo_bin, fill = predicted_pass_rate)) +
  geom_tile(color = "white", size = 0.5) +
  geom_text(aes(label = paste0(round(predicted_pass_rate, 2), "\n(", play_count, ")")), 
            color = "white", size = 3, fontface = "bold") +
  scale_fill_viridis_c(name = "Pass\nProbability", option = "plasma") +
  labs(title = "Pass Probability by Down & Distance",
       subtitle = "2023 Season - Model Predictions with Play Counts",
       x = "Down", y = "Yards to Go") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        axis.text = element_text(size = 10))

ggsave("plots/situational_heatmap.png", p2, width = 10, height = 6, dpi = 300)

# ───────────────────────────────────────────────────────────────
# 8.3) Surprisal Distribution Analysis
# ───────────────────────────────────────────────────────────────

surprisal_data <- df_base[season == 2023 & !is.na(pred_pass_prob), .(
  play_id, game_id, play_type, pred_pass_prob,
  surprisal = -log(ifelse(play_type == "pass", pred_pass_prob, 1 - pred_pass_prob))
)]

p3 <- ggplot(surprisal_data, aes(x = surprisal, fill = play_type)) +
  geom_histogram(bins = 50, alpha = 0.7, position = "identity") +
  scale_fill_manual(values = c("pass" = "#E74C3C", "run" = "#3498DB"),
                    name = "Play Type") +
  labs(title = "Distribution of Play Surprisal",
       subtitle = "Higher values = more unexpected plays",
       x = "Surprisal (-log probability)", y = "Count") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"),
        legend.position = "top")

ggsave("plots/surprisal_distribution.png", p3, width = 10, height = 6, dpi = 300)

# ───────────────────────────────────────────────────────────────
# 8.4) Top Pass Rushers Visualization
# ───────────────────────────────────────────────────────────────

top_rushers <- evaluate_pass_rushers(df_base[season == 2023])
top_15 <- head(top_rushers, 15)

p4 <- ggplot(top_15, aes(x = reorder(Player, Disruption_Rate), y = Disruption_Rate, fill = Position)) +
  geom_col(alpha = 0.8) +
  geom_text(aes(label = round(Disruption_Rate, 3)), hjust = -0.1, size = 3) +
  scale_fill_brewer(type = "qual", palette = "Set2") +
  coord_flip() +
  labs(title = "Top 15 Pass Rushers by Surprisal-Weighted Disruption Rate",
       subtitle = "2023 Season - Minimum 100 Weighted Pass Rush Snaps",
       x = "Player", y = "Disruption Rate") +
  theme_minimal() +
  theme(plot.title = element_text(size = 14, face = "bold"),
        axis.text.y = element_text(size = 9))

ggsave("plots/top_pass_rushers.png", p4, width = 12, height = 8, dpi = 300)

# ───────────────────────────────────────────────────────────────
# 8.5) Feature Importance Visualization
# ───────────────────────────────────────────────────────────────

importance_df <- as.data.table(importance)[1:8]  # Top 8 features
importance_df[, Feature := factor(Feature, levels = rev(Feature))]

p5 <- ggplot(importance_df, aes(x = Feature, y = Gain, fill = Feature)) +
  geom_col(alpha = 0.8, show.legend = FALSE) +
  geom_text(aes(label = round(Gain, 3)), hjust = -0.1, size = 3.5) +
  scale_fill_viridis_d(option = "viridis") +
  coord_flip() +
  labs(title = "XGBoost Feature Importance",
       subtitle = "Information Gain by Feature",
       x = "Feature", y = "Gain") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"))

ggsave("plots/feature_importance.png", p5, width = 10, height = 6, dpi = 300)

# ───────────────────────────────────────────────────────────────
# 8.6) Prediction Calibration Plot
# ───────────────────────────────────────────────────────────────

calibration_data <- df_base[season == 2023 & !is.na(pred_pass_prob)]
calibration_data[, prob_bin := cut(pred_pass_prob, breaks = seq(0, 1, 0.1), include.lowest = TRUE)]

calibration_summary <- calibration_data[, .(
  mean_predicted = mean(pred_pass_prob, na.rm = TRUE),
  mean_actual = mean(is_pass, na.rm = TRUE),
  count = .N
), by = prob_bin]

p6 <- ggplot(calibration_summary, aes(x = mean_predicted, y = mean_actual)) +
  geom_point(aes(size = count), alpha = 0.7, color = "#E74C3C") +
  geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
  scale_size_continuous(name = "Play Count") +
  xlim(0, 1) + ylim(0, 1) +
  labs(title = "Model Calibration",
       subtitle = "Perfect calibration would fall on the diagonal line",
       x = "Predicted Pass Probability", y = "Actual Pass Rate") +
  theme_minimal() +
  theme(plot.title = element_text(size = 16, face = "bold"))

ggsave("plots/model_calibration.png", p6, width = 10, height = 6, dpi = 300)

# ───────────────────────────────────────────────────────────────
# 8.7) Team-Level Analysis WITH PROPER TEAM LOGOS
# ───────────────────────────────────────────────────────────────

# Calculate team analysis data
team_analysis <- df_base[season == 2023 & !is.na(pred_pass_prob), .(
  actual_pass_rate = mean(is_pass, na.rm = TRUE),
  predicted_pass_rate = mean(pred_pass_prob, na.rm = TRUE),
  avg_surprisal = mean(-log(ifelse(is_pass == 1, pred_pass_prob, 1 - pred_pass_prob)), na.rm = TRUE),
  play_count = .N
), by = .(team = posteam)]

team_analysis[, predictability_score := 1 / avg_surprisal]

# Load team data with logos
team_info <- tryCatch({
  setDT(load_teams())[, .(team_abbr, team_logo_espn, team_color, team_color2)]
}, error = function(e) {
  cat("Warning: Could not load team data. Creating fallback data.\n")
  # Create fallback team data
  teams <- unique(team_analysis$team)
  teams <- teams[!is.na(teams)]
  data.table(
    team_abbr = teams,
    team_logo_espn = paste0("https://a.espncdn.com/i/teamlogos/nfl/500/", tolower(teams), ".png"),
    team_color = "#FF0000",
    team_color2 = "#000000"
  )
})

# Merge team info with analysis
team_analysis_with_logos <- merge(team_analysis, team_info, 
                                  by.x = "team", by.y = "team_abbr", 
                                  all.x = TRUE)

# Remove any teams without logos
team_analysis_with_logos <- team_analysis_with_logos[!is.na(team_logo_espn)]

# Create static plot with team logos using ggimage
if (requireNamespace("ggimage", quietly = TRUE)) {
  p7_with_logos <- ggplot(team_analysis_with_logos, aes(x = actual_pass_rate, y = predicted_pass_rate)) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50", alpha = 0.7) +
    geom_image(aes(image = team_logo_espn), size = 0.05, by = "width") + # Use by="width" for better aspect ratio handling
    geom_text(aes(label = team), vjust = 2.2, size = 2.5, fontface = "bold", check_overlap = TRUE) +
    scale_x_continuous(labels = scales::percent, limits = c(0.35, 0.75)) +
    scale_y_continuous(labels = scales::percent, limits = c(0.35, 0.75)) +
    labs(title = "NFL Team Offensive Predictability with Team Logos",
         subtitle = "2023 Season - Actual vs Predicted Pass Rates",
         x = "Actual Pass Rate", y = "Predicted Pass Rate",
         caption = "Diagonal line = perfect prediction. Logos courtesy of ESPN.") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"),
          plot.subtitle = element_text(size = 12),
          panel.grid.minor = element_blank())
  
  ggsave("plots/team_predictability_with_logos.png", p7_with_logos, width = 12, height = 10, dpi = 300)
  cat("Team logo plot created successfully!\n")
} else {
  cat("ggimage package not found. Skipping team logo plot.\n")
}


# ───────────────────────────────────────────────────────────────
# 8.8) Enhanced Interactive Plotly Dashboard - FINAL & ROBUST VERSION
# ───────────────────────────────────────────────────────────────
library(ggrepel)

# --- Nudge the logo positions to prevent overlap using ggrepel ---
set.seed(42) # for reproducible nudging
p_for_data <- ggplot(team_analysis_with_logos, aes(x = actual_pass_rate, y = predicted_pass_rate)) +
  geom_point() +
  geom_text_repel(aes(label = team), 
                  box.padding = 1.5,
                  max.overlaps = Inf,
                  min.segment.length = 0)

# Extract the nudged data from the ggplot build
repel_data <- as.data.table(layer_data(p_for_data, 2))

# Merge the new x/y coordinates back into our main analysis table
team_analysis_with_logos[, `:=`(
  x_repel = repel_data$x,
  y_repel = repel_data$y
)]

# --- Create the list of image objects for the layout using nudged coordinates ---
logo_images <- lapply(1:nrow(team_analysis_with_logos), function(i) {
  list(
    source = team_analysis_with_logos$team_logo_espn[i],
    xref = "x",
    yref = "y",
    x = team_analysis_with_logos$x_repel[i],
    y = team_analysis_with_logos$y_repel[i],
    sizex = 0.03,
    sizey = 0.03,
    xanchor = "center",
    yanchor = "middle",
    layer = "above"
  )
})

# --- Create plot using the "Dummy Trace" method for robustness ---

# 1. Create a base plot with just the layout (no data)
p8_interactive_final <- plot_ly() %>%
  layout(
    title = list(
      text = "<b>NFL Team Offensive Predictability Analysis - 2023 Season</b><br><sub>Interactive Dashboard: Hover Over Logos for Details</sub>",
      font = list(size = 18, family = "Arial Black")
    ),
    xaxis = list(
      title = list(text = "<b>Actual Pass Rate</b>", font = list(size = 14)),
      range = c(0.35, 0.75),
      tickformat = ".0%"
    ),
    yaxis = list(
      title = list(text = "<b>Predicted Pass Rate</b>", font = list(size = 14)),
      range = c(0.35, 0.75),
      tickformat = ".0%"
    ),
    images = logo_images, # Add the nudged logos to the layout
    showlegend = TRUE,
    legend = list(
      orientation = "v",
      x = 1.02,
      y = 0.95,
      bgcolor = "rgba(255,255,255,0.8)",
      bordercolor = "rgba(128,128,128,0.5)",
      borderwidth = 1,
      title = list(text="<b>Legend</b>")
    )
  )

# 2. Add the invisible markers for the team hover data
p8_interactive_final <- p8_interactive_final %>% add_trace(
  data = team_analysis_with_logos,
  x = ~actual_pass_rate, 
  y = ~predicted_pass_rate,
  type = 'scatter', 
  mode = 'markers',
  marker = list(opacity = 0), # Invisible hover targets
  name = "NFL Team", # Legend name
  text = ~paste0(
    "<b style='font-size: 14px;'>", team, "</b><br>",
    "<b>Actual Pass Rate:</b> ", scales::percent(actual_pass_rate, accuracy = 0.1), "<br>",
    "<b>Predicted Pass Rate:</b> ", scales::percent(predicted_pass_rate, accuracy = 0.1), "<br>",
    "<b>Prediction Error:</b> ", scales::percent(abs(actual_pass_rate - predicted_pass_rate), accuracy = 0.1), "<br>",
    "<b>Avg Surprisal:</b> ", round(avg_surprisal, 2), "<br>",
    "<b>Predictability Score:</b> ", round(predictability_score, 3), "<br>",
    "<b>Total Plays:</b> ", format(play_count, big.mark = ",")
  ),
  hovertemplate = "%{text}<extra></extra>",
  showlegend = TRUE # Show this in the legend
)

# 3. Add the diagonal line as a separate trace
p8_interactive_final <- p8_interactive_final %>% add_trace(
  x = c(0.35, 0.75), 
  y = c(0.35, 0.75),
  type = 'scatter',
  mode = 'lines',
  name = "Perfect Prediction",
  line = list(dash = "dash", color = "rgba(128,128,128,0.8)", width = 2),
  hoverinfo = "none",
  showlegend = TRUE
)

# 4. Finalize layout and config (can be chained, but separated for clarity)
p8_interactive_final <- p8_interactive_final %>% layout(
  annotations = list(
    list(
      x = 1.03, y = 0.85, xref = "paper", yref = "paper",
      text = "Hover over logos for<br>detailed team stats.",
      showarrow = FALSE, font = list(size = 12), align = "left"
    )
  )
) %>%
  config(
    displayModeBar = TRUE, 
    displaylogo = FALSE,
    modeBarButtonsToRemove = c("pan2d", "select2d", "lasso2d", "autoScale2d"),
    toImageButtonOptions = list(
      format = "png",
      filename = "nfl_team_predictability_final",
      height = 800,
      width = 1200,
      scale = 2
    )
  )

# Save the final interactive plot
htmlwidgets::saveWidget(p8_interactive_final, "plots/interactive_team_dashboard_final.html", selfcontained = TRUE)

cat("Final interactive dashboard with repelled logos created successfully!\n")
# ───────────────────────────────────────────────────────────────
# 8.9) Team Logo Grid Plot (Alternative Approach)
# ───────────────────────────────────────────────────────────────

# Create a grid plot showing team logos with key metrics
create_team_logo_grid <- function(team_data) {
  # Sort teams by predictability score
  team_data <- team_data[order(-predictability_score)]
  
  # Create a grid layout
  n_teams <- nrow(team_data)
  n_cols <- 8
  n_rows <- ceiling(n_teams / n_cols)
  
  # Add grid positions
  team_data[, `:=`(
    grid_x = rep(1:n_cols, length.out = n_teams),
    grid_y = rep(n_rows:1, each = n_cols, length.out = n_teams)
  )]
  
  if (requireNamespace("ggimage", quietly = TRUE)) {
    p_grid <- ggplot(team_data, aes(x = grid_x, y = grid_y)) +
      geom_image(aes(image = team_logo_espn), size = 0.08) +
      geom_text(aes(label = paste0(team, "\n", round(predictability_score, 2))), 
                vjust = -2, size = 2.5, fontface = "bold") +
      scale_x_continuous(limits = c(0.5, n_cols + 0.5), breaks = NULL) +
      scale_y_continuous(limits = c(0.5, n_rows + 0.5), breaks = NULL) +
      labs(title = "NFL Teams by Predictability Score",
           subtitle = "2023 Season - Higher scores = more predictable play-calling",
           caption = "Logos courtesy of ESPN") +
      theme_void() +
      theme(plot.title = element_text(size = 16, face = "bold", hjust = 0.5),
            plot.subtitle = element_text(size = 12, hjust = 0.5),
            plot.caption = element_text(size = 8, hjust = 0.5))
    
    ggsave("plots/team_logo_grid.png", p_grid, width = 16, height = 12, dpi = 300)
    cat("Team logo grid created successfully!\n")
  }
}

# Create the grid plot
if (nrow(team_analysis_with_logos) > 0) {
  create_team_logo_grid(team_analysis_with_logos)
}

# ───────────────────────────────────────────────────────────────
# 8.10) Summary Dashboard
# ───────────────────────────────────────────────────────────────

# Create a summary statistics table
summary_stats <- data.table(
  Metric = c("Overall Model AUC (2023)", "Average Surprisal", "Most Predictable Team", 
             "Least Predictable Team", "Total Plays Analyzed", "Pass Rate (2023)"),
  Value = c(
    round(auc(roc(df_base[season == 2023]$is_pass, df_base[season == 2023]$pred_pass_prob, quiet = TRUE)), 3),
    round(mean(surprisal_data$surprisal, na.rm = TRUE), 3),
    team_analysis[which.max(predictability_score)]$team,
    team_analysis[which.min(predictability_score)]$team,
    format(nrow(df_base[season == 2023]), big.mark = ","),
    paste0(round(mean(df_base[season == 2023]$is_pass, na.rm = TRUE) * 100, 1), "%")
  )
)

cat("\n=== MODEL SUMMARY STATISTICS ===\n")
print(kable(summary_stats, format = "simple"))

# List all created files
cat("\n=== VISUALIZATION FILES CREATED ===\n")
created_files <- c(
  "plots/model_performance_timeline.png",
  "plots/situational_heatmap.png", 
  "plots/surprisal_distribution.png",
  "plots/top_pass_rushers.png",
  "plots/feature_importance.png",
  "plots/model_calibration.png",
  "plots/team_predictability_with_logos.png",
  "plots/interactive_team_dashboard.html",
  "plots/team_logo_grid.png"
)

for (file in created_files) {
  if (file.exists(file)) {
    cat("✓", file, "\n")
  } else {
    cat("✗", file, "(not created)\n")
  }
}

cat("\n=== VISUALIZATION SUMMARY ===\n")
cat("• Static plots with team logos: ggimage package used\n")
cat("• Interactive dashboard: Enhanced Plotly visualization\n")
cat("• Team logo grid: Sorted by predictability score\n")
cat("• All plots saved to 'plots/' directory\n")

# ───────────────────────────────────────────────────────────────
# 8.11) INTERACTIVE PLAYER PERFORMANCE DASHBOARD (NEW SECTION)
# ───────────────────────────────────────────────────────────────

cat("Creating interactive player performance dashboard...\n")

# 1. Get the 2023 player data using our existing function
player_data_2023 <- evaluate_pass_rushers(df_base[season == 2023])

# 2. Add a 'Total Disruptions' column for the y-axis
player_data_2023[, Total_Disruptions := Weighted_Sacks + Weighted_QB_Hits]

# 3. Create the interactive Plotly chart
p9_player_interactive <- plot_ly(
  data = player_data_2023,
  x = ~Weighted_Pass_Rush_Snaps,
  y = ~Total_Disruptions,
  type = 'scatter',
  mode = 'markers',
  # Color represents the player's efficiency
  color = ~Disruption_Rate,
  colors = viridis::cividis(10), # A colorblind-friendly palette
  # Size represents their total production
  size = ~Total_Disruptions,
  # The detailed hover text is key
  text = ~paste0(
    "<b style='font-size: 16px;'>", Player, "</b><br>",
    "<b>Team:</b> ", Team, " | <b>Position:</b> ", Position, "<br>",
    "------------------------------------<br>",
    "<b>Disruption Rate:</b> ", scales::percent(Disruption_Rate, accuracy = 0.01), "<br>",
    "<b>Total Disruptions:</b> ", round(Total_Disruptions, 2), "<br>",
    "  • Weighted Sacks: ", round(Weighted_Sacks, 2), "<br>",
    "  • Weighted QB Hits: ", round(Weighted_QB_Hits, 2), "<br>",
    "<b>Total Snaps:</b> ", round(Weighted_Pass_Rush_Snaps, 1)
  ),
  hovertemplate = "%{text}<extra></extra>",
  marker = list(
    sizemode = 'area',
    sizeref = max(player_data_2023$Total_Disruptions, na.rm = TRUE) / 40^2, # Controls bubble size scaling
    line = list(width = 1, color = 'rgba(0,0,0,0.5)')
  )
) %>%
  layout(
    title = list(
      text = "<b>Interactive Pass Rusher Analysis - 2023 Season</b><br><sub>Hover Over Players for Detailed Stats</sub>",
      font = list(size = 18, family = "Arial, sans-serif")
    ),
    xaxis = list(
      type = 'log', # Log scale helps spread out players with fewer snaps
      title = list(text = "<b>Pass Rush Volume (Weighted Snaps)</b>", font = list(size = 14)),
      showgrid = TRUE
    ),
    yaxis = list(
      title = list(text = "<b>Total Production (Weighted Sacks + Hits)</b>", font = list(size = 14)),
      showgrid = TRUE
    ),
    showlegend = TRUE,
    legend = list(title = list(text="Disruption<br>Rate")),
    # Add annotations to guide the user's interpretation
    annotations = list(
      list(
        x = log10(600), y = 14, xref = "x", yref = "y",
        text = "<b>Elite Tier</b><br>(High Volume & High Production)",
        showarrow = TRUE, arrowhead = 2, ax = 50, ay = -40,
        font = list(color = "#2c3e50", size = 12),
        bgcolor = "rgba(255,255,255,0.7)"
      ),
      list(
        x = log10(120), y = 8, xref = "x", yref = "y",
        text = "<b>Efficient Specialists</b><br>(High Rate, Lower Volume)",
        showarrow = TRUE, arrowhead = 2, ax = 60, ay = 30,
        font = list(color = "#2c3e50", size = 12),
        bgcolor = "rgba(255,255,255,0.7)"
      )
    )
  ) %>%
  config(
    displayModeBar = TRUE, 
    displaylogo = FALSE
  )

# 4. Save the new interactive plot
htmlwidgets::saveWidget(p9_player_interactive, "plots/interactive_player_dashboard.html", selfcontained = TRUE)

cat("Interactive player performance dashboard created successfully at 'plots/interactive_player_dashboard.html'\n")

# Don't forget to add the new file to your list of created files at the end
# created_files <- c(created_files, "plots/interactive_player_dashboard.html")