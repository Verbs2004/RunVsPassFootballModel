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










