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
  DB_def_P = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*DB)"))
)]
# Fill NAs that result from parsing
df_base[is.na(RB_off_P), RB_off_P := 0L][is.na(TE_off_P), TE_off_P := 0L][is.na(WR_off_P), WR_off_P := 0L]
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

# ───────────────────────────────────────────────────────────────
# 4) Create Modeling and Evaluation Datasets
# ───────────────────────────────────────────────────────────────
cat("Creating modeling and evaluation datasets...\n")
modeling_cols <- c(
  "game_id", "old_game_id", "play_id", "season", "is_pass", "down", "ydstogo", 
  "down_dist_cat", "yardline_100", "score_differential", "heavy_set", "empty_back", 
  "is_nickel", "is_dime", "skill_diff_P", "defenders_in_box", "box_advantage_off", 
  "form_cat", "shotgun_flag", "no_huddle_flag", "two_minute_drill", "four_minute_drill", 
  "wp", "late_desperation", "trailing_big", "leading_big", "score_situation", 
  "goal_to_go", "timeout_situation", "play_clock_at_snap", "ep", "is_first_play_of_drive", 
  "prev_play_was_pass", "yards_gained_on_prev_play", "spread_line", "total_line", 
  "yards_to_goal_bin", "posteam", "defteam", "game_seconds_remaining"
)
df_modeling <- df_base[, ..modeling_cols]

evaluation_cols <- c(
  "game_id", "old_game_id", "play_id", "season", "is_pass", "down", "ydstogo", 
  "yardline_100", "score_differential", "qtr", "quarter_seconds_remaining", 
  "game_seconds_remaining", "wp", "defenders_in_box", "pass_attempt", "sack", 
  "qb_hit", "sack_player_id", "half_sack_1_player_id", "half_sack_2_player_id", 
  "qb_hit_1_player_id", "qb_hit_2_player_id", "posteam", "defteam"
)
df_evaluation <- df_base[, ..evaluation_cols]

cat("Total rows after cleaning:", format(nrow(df_modeling), big.mark=","), "\n")
cat("Pass rate:", round(df_modeling[, mean(is_pass)], 3), "\n")

# ───────────────────────────────────────────────────────────────
# 5) Team Tendency Encoding with data.table
# ───────────────────────────────────────────────────────────────
team_stats <- df_modeling[, .(team_pass_rate = mean(is_pass)), by = posteam][, team_pass_tendency := as.numeric(scale(team_pass_rate))]
def_stats <- df_modeling[, .(def_pass_rate_allowed = mean(is_pass)), by = defteam][, def_pass_tendency := as.numeric(scale(def_pass_rate_allowed))]

df_modeling[team_stats, on = "posteam", team_pass_tendency := i.team_pass_tendency]
df_modeling[def_stats, on = "defteam", def_pass_tendency := i.def_pass_tendency]
df_modeling[is.na(team_pass_tendency), team_pass_tendency := 0]
df_modeling[is.na(def_pass_tendency), def_pass_tendency := 0]
df_modeling[, `:=`(posteam = NULL, defteam = NULL)]

# ───────────────────────────────────────────────────────────────
# 6) Train/Test Split and Preprocessing
# ───────────────────────────────────────────────────────────────
train_raw <- df_modeling[season < 2023]
test_raw <- df_modeling[season == 2023] # Still useful for model evaluation metric

# Interaction terms
create_interactions_dt <- function(dt) {
  dt[, `:=`(
    down_x_dist = down * ydstogo,
    third_and_long = as.numeric(down == 3 & ydstogo >= 7),
    shotgun_x_down = shotgun_flag * down,
    heavy_in_redzone = heavy_set * as.numeric(yardline_100 <= 20),
    empty_on_third = empty_back * as.numeric(down == 3),
    nickel_on_third_down = as.numeric(is_nickel & down == 3),
    trailing_late = trailing_big * as.numeric(game_seconds_remaining <= 600),
    leading_late = leading_big * as.numeric(game_seconds_remaining <= 600)
  )]
  dt[, game_seconds_remaining := NULL]
  return(dt)
}

train_data <- create_interactions_dt(copy(train_raw))
test_data <- create_interactions_dt(copy(test_raw))

# Prepare for XGBoost
y_train <- train_data$is_pass
y_test <- test_data$is_pass

# Remove identifiers and target
cols_to_remove <- c("game_id", "old_game_id", "play_id", "season", "is_pass")
train_features <- train_data[, !..cols_to_remove]
test_features <- test_data[, !..cols_to_remove]

# One-hot encode using fastDummies
train_dummies <- dummy_cols(train_features, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
test_dummies <- dummy_cols(test_features, remove_first_dummy = TRUE, remove_selected_columns = TRUE)

# Align columns
common_cols <- intersect(names(train_dummies), names(test_dummies))
train_final <- train_dummies[, ..common_cols]
test_final <- test_dummies[, ..common_cols]

# Convert to matrix
train_matrix <- as.matrix(train_final)
test_matrix <- as.matrix(test_final)

# Create DMatrix objects
dtrain <- xgb.DMatrix(data = train_matrix, label = y_train)
dtest <- xgb.DMatrix(data = test_matrix, label = y_test)

# ───────────────────────────────────────────────────────────────
# 7) Optimized XGBoost Training with Early Stopping
# ───────────────────────────────────────────────────────────────
cat("Training XGBoost model with optimized parameters and early stopping...\n")

params <- list(
  objective = "binary:logistic",
  eval_metric = "auc",
  eta = 0.05,
  max_depth = 8,
  subsample = 0.75,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  gamma = 0,
  tree_method = 'hist',
  nthread = n_cores
)

watchlist <- list(train = dtrain, test = dtest)

set.seed(42)
xgb_model <- xgb.train(
  params = params,
  data = dtrain,
  nrounds = 1000,
  watchlist = watchlist,
  early_stopping_rounds = 30,
  verbose = 0
)

cat("Model training completed at round:", xgb_model$best_iteration, "\n")

# ───────────────────────────────────────────────────────────────
# 8) Model Evaluation & Prediction for All Seasons
# ───────────────────────────────────────────────────────────────
# Evaluate on 2023 test set
test_probs <- predict(xgb_model, test_matrix, ntreelimit = xgb_model$best_iteration)
test_preds <- as.integer(test_probs > 0.5)
conf_mat <- table(Predicted = test_preds, Actual = y_test)
accuracy <- sum(diag(conf_mat)) / sum(conf_mat)
roc_obj <- roc(y_test, test_probs, quiet = TRUE)
auc_score <- auc(roc_obj)
cat("\n=== Final XGBoost MODEL PERFORMANCE (2023 TEST SET) ===\n")
cat("AUC:", round(auc_score, 4), "\n")
cat("Accuracy:", round(accuracy, 4), "\n")
print(conf_mat)

# Prepare all data for prediction
cat("Preparing all historical data for prediction...\n")
all_seasons_raw <- df_modeling
all_seasons_data <- create_interactions_dt(copy(all_seasons_raw))
all_seasons_ids <- all_seasons_data[, .(game_id, old_game_id, play_id, season)]
all_seasons_features <- all_seasons_data[, !..cols_to_remove]
all_seasons_dummies <- dummy_cols(all_seasons_features, remove_first_dummy = TRUE, remove_selected_columns = TRUE)
# Ensure all columns from training are present, fill missing with 0
missing_cols <- setdiff(common_cols, names(all_seasons_dummies))
if (length(missing_cols) > 0) {
  all_seasons_dummies[, (missing_cols) := 0]
}
all_seasons_final <- all_seasons_dummies[, ..common_cols]
all_seasons_matrix <- as.matrix(all_seasons_final)
d_all_seasons <- xgb.DMatrix(data = all_seasons_matrix)

# Generate predictions for all seasons
cat("Generating predictions for all seasons (2016-2023)...\n")
all_seasons_probs <- predict(xgb_model, d_all_seasons, ntreelimit = xgb_model$best_iteration)
predictions_all_seasons <- cbind(all_seasons_ids, data.table(pass_prob = all_seasons_probs))

# ───────────────────────────────────────────────────────────────
# 9) Enhanced Pass Rusher Evaluation with PRI Metric
# ───────────────────────────────────────────────────────────────

# --- Pass Rush Informativeness (PRI) Metric - Explanation ---
# This metric moves beyond raw sack counts to evaluate pass rushers based on the
# context of the play, using a concept from information theory called "Surprisal".
# Core Idea: A disruption (sack/hit) on a play where a pass was UNEXPECTED
# is more valuable and "informative" about a player's true skill than a
# disruption on an obvious passing down (e.g., 3rd & 15).
# The formula is: PRI Score = (Base Disruption Value) * (-log(pass_prob))
# This heavily rewards sacks/hits on low pass probability plays.
# ---

# --- Player Evaluation Function (PRI Metric) ---
pri_rusher_eval_dt <- function(eval_data, parts_data, rosters_data) {
  if (is.null(rosters_data) || is.null(parts_data) || nrow(rosters_data) == 0 || nrow(parts_data) == 0) {
    return(data.table())
  }
  
  player_snaps <- eval_data[pass_attempt == 1 | sack == 1]
  player_snaps <- player_snaps[parts_data, on = c("old_game_id", "play_id"), nomatch=0]
  player_snaps <- player_snaps[!is.na(defense_players) & defense_players != ""]
  
  player_snaps <- player_snaps[, .(gsis_id = unlist(strsplit(defense_players, ";"))), by = .(play_id, old_game_id, pass_prob, sack, qb_hit, sack_player_id, half_sack_1_player_id, half_sack_2_player_id, qb_hit_1_player_id, qb_hit_2_player_id)]
  
  player_snaps <- player_snaps[rosters_data, on = "gsis_id", nomatch=0]
  
  player_snaps <- player_snaps[position %in% c("DE", "DT", "EDGE", "OLB", "ILB", "LB", "NT", "DL")]
  
  player_snaps[, `:=`(
    base_disruption = fcase(
      !is.na(sack_player_id) & gsis_id == sack_player_id, 1.0,
      !is.na(half_sack_1_player_id) & gsis_id == half_sack_1_player_id, 0.5,
      !is.na(half_sack_2_player_id) & gsis_id == half_sack_2_player_id, 0.5,
      !is.na(qb_hit_1_player_id) & gsis_id == qb_hit_1_player_id, 0.5,
      !is.na(qb_hit_2_player_id) & gsis_id == qb_hit_2_player_id, 0.5,
      default = 0.0
    )
  )]
  
  player_snaps[base_disruption > 0, pri_score := base_disruption * (-log(pmax(pass_prob, 1e-9)))]
  player_snaps[is.na(pri_score), pri_score := 0]
  
  results <- player_snaps[, .(
    pass_rush_snaps = .N,
    total_disruptions = sum(base_disruption > 0),
    total_sacks = sum(fcase(!is.na(sack_player_id) & gsis_id == sack_player_id, 1.0,
                            !is.na(half_sack_1_player_id) & gsis_id == half_sack_1_player_id, 0.5,
                            !is.na(half_sack_2_player_id) & gsis_id == half_sack_2_player_id, 0.5,
                            default = 0.0)),
    total_qb_hits = sum(fcase(!is.na(qb_hit_1_player_id) & gsis_id == qb_hit_1_player_id, 1.0,
                              !is.na(qb_hit_2_player_id) & gsis_id == qb_hit_2_player_id, 1.0,
                              default = 0.0)),
    total_pri = sum(pri_score, na.rm = TRUE)
  ), by = .(full_name, position, team)]
  
  results <- results[pass_rush_snaps >= 50]
  results[, pri_rate := total_pri / pass_rush_snaps]
  
  setorder(results, -pri_rate)
  
  return(results[, .(
    Player = full_name, 
    Team = team, 
    Position = position, 
    PRI_Rate = pri_rate, 
    Total_PRI = total_pri,
    Total_Sacks = total_sacks,
    Total_QB_Hits = total_qb_hits,
    Pass_Rush_Snaps = pass_rush_snaps
  )])
}

# ───────────────────────────────────────────────────────────────
# 10) Execute Historical Evaluation & Display Results
# ───────────────────────────────────────────────────────────────
cat("\n--- Running Pass Rusher Evaluation for All Seasons (2016-2023) ---\n")

for (year in 2016:2023) {
  cat(paste("\n=== TOP 25 PASS RUSHERS BY PRI RATE (", year, ") ===\n"))
  
  current_year_eval <- df_evaluation[season == year]
  current_year_preds <- predictions_all_seasons[season == year]
  
  eval_data_with_probs <- current_year_preds[current_year_eval, on = c("game_id", "old_game_id", "play_id", "season")]
  
  rosters_for_year <- tryCatch({
    setDT(load_rosters(year))[, .(gsis_id, full_name, position, team)]
  }, error = function(e) {
    cat(paste("Warning: Could not load", year, "roster data.\n"))
    data.table()
  })
  
  parts_for_year <- if (!is.null(parts) && nrow(parts) > 0) {
    parts[substr(old_game_id, 1, 4) == as.character(year)]
  } else {
    data.table()
  }
  
  pri_rushers <- pri_rusher_eval_dt(eval_data_with_probs, parts_for_year, rosters_for_year)
  
  if (!is.null(pri_rushers) && nrow(pri_rushers) > 0) {
    print(kable(head(pri_rushers, 25), digits = 4, format = "simple"))
  } else {
    cat(paste("No sufficient pass rusher data for", year, ".\n"))
  }
}


# ───────────────────────────────────────────────────────────────
# 11) Model Feature Importance
# ───────────────────────────────────────────────────────────────
cat("\n=== TOP 20 MOST IMPORTANT FEATURES ===\n")
importance_matrix <- xgb.importance(model = xgb_model)
print(kable(head(importance_matrix, 20), format = "simple"))

# ───────────────────────────────────────────────────────────────
# 12) Clean Up Parallel Backend
# ───────────────────────────────────────────────────────────────
stopCluster(cl)
cat("\n=== ANALYSIS COMPLETE ===\n")
