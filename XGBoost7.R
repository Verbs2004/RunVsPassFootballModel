# ───────────────────────────────────────────────────────────────
# Fixed NFL Play Prediction Model & Enhanced Pass Rusher Evaluation
# ───────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# 0) Libraries
# ───────────────────────────────────────────────────────────────
library(nflreadr)
library(dplyr); library(tidyr); library(stringr)
library(pROC); library(readr); library(purrr)
library(ggplot2); library(slider)
library(fastDummies)
library(caret)
library(xgboost)
library(knitr)

# ───────────────────────────────────────────────────────────────
# 1) Load core tables (2016-23)
# ───────────────────────────────────────────────────────────────
pbp <- load_pbp(2016:2023)

sched <- load_schedules(2016:2023) %>%
  select(game_id,
         sched_roof = roof,
         sched_temp = temp,
         sched_wind_speed = wind)

parts <- tryCatch({
  load_participation(2016:2023) %>%
    select(old_game_id, play_id, defenders_in_box, offense_personnel, defense_personnel, defense_players)
}, error = function(e) {
  cat("Warning: Could not load participation data. It will be estimated.\n")
  data.frame()
})

plays <- tryCatch({
  read_csv("plays.csv", show_col_types = FALSE)
}, error = function(e) {
  cat("Warning: plays.csv not found. Related features will be estimated.\n")
  data.frame()
})

if (!"absolute_yardline_number" %in% names(pbp)) {
  pbp <- pbp %>% mutate(
    absolute_yardline_number = if_else(side_of_field == posteam, 100 - yardline_100, yardline_100)
  )
}

# ───────────────────────────────────────────────────────────────
# 2) Helper function for parsing personnel
# ───────────────────────────────────────────────────────────────
parse_pers <- function(x, pos) {
  if (is.na(x) || x == "" || is.null(x)) return(0L)
  
  pattern <- sprintf("(\\d+)\\s*%s", pos)
  matches <- str_extract(x, pattern)
  
  if (is.na(matches)) return(0L)
  
  num <- as.integer(str_extract(matches, "\\d+"))
  return(if_else(is.na(num), 0L, num))
}

# ───────────────────────────────────────────────────────────────
# 3) Join data sources
# ───────────────────────────────────────────────────────────────
df_base <- pbp %>%
  filter(
    play_type %in% c("run", "pass"),
    !is.na(down), down %in% 1:4,
    !is.na(ydstogo), ydstogo >= 1, ydstogo <= 50,
    !is.na(yardline_100), yardline_100 >= 1, yardline_100 <= 99,
    !is.na(score_differential),
    !is.na(game_seconds_remaining),
    !is.na(posteam), !is.na(defteam)
  ) %>%
  left_join(sched, by = "game_id")

# Join plays.csv if it exists
if (nrow(plays) > 0) {
  plays_j <- plays %>%
    rename(game_id = gameId, play_id = playId) %>%
    mutate(game_id = as.character(game_id), play_id = as.numeric(play_id)) %>%
    select(game_id, play_id, offenseFormation, playAction)
  
  df_base <- df_base %>%
    left_join(plays_j, by = c("game_id", "play_id"))
}

# Join participation data
if (nrow(parts) > 0) {
  parts_j <- parts %>%
    rename(
      parts_off_pers = offense_personnel,
      parts_def_pers = defense_personnel) %>%
    mutate(old_game_id = as.character(old_game_id))
  
  df_base <- df_base %>%
    mutate(old_game_id = as.character(old_game_id)) %>%
    left_join(parts_j, by = c("old_game_id", "play_id"))
}

# ───────────────────────────────────────────────────────────────
# 4) Feature engineering with FIXED personnel features
# ───────────────────────────────────────────────────────────────
df <- df_base %>%
  arrange(game_id, fixed_drive, play_id) %>%
  group_by(game_id, posteam) %>%
  mutate(drive_play_idx = row_number()) %>%
  ungroup() %>%
  
  # Create target variable
  mutate(is_pass = as.numeric(play_type == "pass")) %>%
  
  # Create sequential features
  group_by(game_id, fixed_drive) %>%
  mutate(
    is_first_play_of_drive = if_else(drive_play_idx == 1, 1, 0),
    prev_play_was_pass = lag(is_pass, 1),
    yards_gained_on_prev_play = lag(yards_gained, 1)
  ) %>%
  ungroup() %>%
  mutate(
    prev_play_was_pass = replace_na(prev_play_was_pass, 0),
    yards_gained_on_prev_play = replace_na(yards_gained_on_prev_play, 0)
  ) %>%
  
  # Personnel features - FIXED VERSION
  mutate(
    # Safely create personnel columns
    offense_personnel = if ("parts_off_pers" %in% names(.)) parts_off_pers else NA_character_,
    defense_personnel = if ("parts_def_pers" %in% names(.)) parts_def_pers else NA_character_,
    
    # Fill NAs with defaults
    offense_personnel = replace_na(offense_personnel, "1 RB, 1 TE, 3 WR"),
    defense_personnel = replace_na(defense_personnel, "4 DL, 3 LB, 4 DB"),
    
    # Parse personnel safely
    RB_off_P = map_int(offense_personnel, ~parse_pers(.x, "RB")),
    TE_off_P = map_int(offense_personnel, ~parse_pers(.x, "TE")),
    WR_off_P = map_int(offense_personnel, ~parse_pers(.x, "WR")),
    DL_def_P = map_int(defense_personnel, ~parse_pers(.x, "DL")),
    LB_def_P = map_int(defense_personnel, ~parse_pers(.x, "LB")),
    DB_def_P = map_int(defense_personnel, ~parse_pers(.x, "DB")),
    defenders_in_box = coalesce(defenders_in_box, DL_def_P + LB_def_P, 7)
  ) %>%
  
  # Advanced personnel features
  mutate(
    heavy_set = as.numeric((RB_off_P + TE_off_P) >= 3),
    empty_back = as.numeric(RB_off_P == 0),
    is_nickel = as.numeric(DB_def_P == 5),
    is_dime = as.numeric(DB_def_P >= 6),
    skill_diff_P = WR_off_P - DB_def_P,
    box_advantage_off = (5 + TE_off_P) - defenders_in_box
  ) %>%
  
  # Contextual features
  mutate(
    # Handle expected points and Vegas lines
    ep = coalesce(as.numeric(ep), 0),
    spread_line = coalesce(as.numeric(spread_line), 0),
    total_line = coalesce(as.numeric(total_line), mean(as.numeric(total_line), na.rm = TRUE)),
    
    goal_to_go = coalesce(as.numeric(goal_to_go), 0),
    
    # Timeout situation
    timeout_situation = case_when(
      posteam_timeouts_remaining > defteam_timeouts_remaining ~ "Offense Advantage",
      posteam_timeouts_remaining < defteam_timeouts_remaining ~ "Defense Advantage",
      posteam_timeouts_remaining == 0 & defteam_timeouts_remaining == 0 ~ "None Left",
      TRUE ~ "Equal Timeouts"
    ),
    
    play_clock_at_snap = coalesce(as.numeric(play_clock), 15),
    
    # FIXED formation categorization
    form_cat = case_when(
      # Check if offenseFormation column exists and use it
      "offenseFormation" %in% names(.) & !is.na(offenseFormation) & 
        str_detect(toupper(offenseFormation), "SHOTGUN") ~ "Shotgun",
      "offenseFormation" %in% names(.) & !is.na(offenseFormation) & 
        str_detect(toupper(offenseFormation), "EMPTY") ~ "Empty",
      "offenseFormation" %in% names(.) & !is.na(offenseFormation) & 
        str_detect(toupper(offenseFormation), "I_FORM|JUMBO") ~ "I-Form",
      "offenseFormation" %in% names(.) & !is.na(offenseFormation) & 
        str_detect(toupper(offenseFormation), "PISTOL") ~ "Pistol",
      "offenseFormation" %in% names(.) & !is.na(offenseFormation) & 
        str_detect(toupper(offenseFormation), "SINGLEBACK") ~ "Singleback",
      # Fall back to shotgun column if available
      coalesce(shotgun, 0) == 1 ~ "Shotgun",
      TRUE ~ "Other"
    ),
    
    two_minute_drill = as.numeric((qtr == 2 | qtr == 4) & quarter_seconds_remaining <= 120),
    four_minute_drill = as.numeric(qtr == 4 & quarter_seconds_remaining <= 240 & score_differential > 0),
    shotgun_flag = as.numeric(coalesce(shotgun, 0) == 1),
    no_huddle_flag = as.numeric(coalesce(no_huddle, 0) == 1),
    
    # Win probability
    wp = if_else(posteam == home_team, home_wp, away_wp),
    wp = coalesce(wp, 0.5),
    wp_tier = case_when(
      wp < 0.2 ~ "desperate", 
      wp < 0.4 ~ "low",
      wp > 0.8 ~ "dominant", 
      wp > 0.6 ~ "high",
      TRUE ~ "medium"
    ),
    
    late_desperation = as.numeric(game_seconds_remaining < 300 & wp < 0.2),
    
    yards_to_goal_bin = case_when(
      yardline_100 <= 4 ~ "goalline", 
      yardline_100 <= 10 ~ "redzone_fringe",
      yardline_100 <= 20 ~ "redzone", 
      yardline_100 <= 50 ~ "own_territory",
      TRUE ~ "backed_up"
    ),
    
    trailing_big = as.numeric(score_differential <= -14),
    leading_big = as.numeric(score_differential >= 14),
    
    score_situation = case_when(
      leading_big == 1 ~ "leading_big", 
      trailing_big == 1 ~ "trailing_big",
      abs(score_differential) <= 4 ~ "close",
      score_differential > 4 ~ "leading", 
      TRUE ~ "trailing"
    ),
    
    down_dist_cat = case_when(
      down == 1 ~ "first",
      down == 2 & ydstogo <= 3 ~ "second_short", 
      down == 2 & ydstogo >= 8 ~ "second_long",
      down == 3 & ydstogo <= 3 ~ "third_short", 
      down == 3 & ydstogo >= 8 ~ "third_long",
      down == 4 & ydstogo <= 3 ~ "fourth_short", 
      down == 4 & ydstogo >= 8 ~ "fourth_long",
      TRUE ~ "medium_yardage"
    )
  )

# ───────────────────────────────────────────────────────────────
# 5) Create separate datasets for modeling and evaluation
# ───────────────────────────────────────────────────────────────

# For modeling - reduced feature set (KEEP game_seconds_remaining for interaction creation)
df_modeling <- df %>%
  select(
    game_id, old_game_id, play_id, season, is_pass,
    # Core situational
    down, ydstogo, down_dist_cat, yardline_100, score_differential,
    # Personnel Features
    heavy_set, empty_back, is_nickel, is_dime,
    skill_diff_P, defenders_in_box, box_advantage_off,
    # Formation and pre-snap
    form_cat, shotgun_flag, no_huddle_flag,
    # Game context
    two_minute_drill, four_minute_drill, wp, late_desperation,
    trailing_big, leading_big, score_situation,
    # Additional Features
    goal_to_go, timeout_situation, play_clock_at_snap, ep,
    is_first_play_of_drive, prev_play_was_pass, yards_gained_on_prev_play,
    spread_line, total_line,
    # Field position
    yards_to_goal_bin,
    # Teams
    posteam, defteam,
    # KEEP game_seconds_remaining for interaction terms
    game_seconds_remaining
  )

# For evaluation - keep all columns needed for pass rush analysis INCLUDING qtr
df_evaluation <- df %>%
  select(
    # Basic identifiers
    game_id, old_game_id, play_id, season, is_pass,
    # Situational context for evaluation - INCLUDING qtr
    down, ydstogo, yardline_100, score_differential, qtr, quarter_seconds_remaining,
    game_seconds_remaining, wp, defenders_in_box,
    # Raw performance data
    pass_attempt, sack, qb_hit,
    sack_player_id, half_sack_1_player_id, half_sack_2_player_id,
    qb_hit_1_player_id, qb_hit_2_player_id,
    # Teams
    posteam, defteam
  )

cat("Total rows after cleaning:", format(nrow(df_modeling), big.mark=","), "\n")
cat("Pass rate:", round(mean(df_modeling$is_pass), 3), "\n")

# ───────────────────────────────────────────────────────────────
# 6) Team tendency encoding
# ───────────────────────────────────────────────────────────────
team_stats <- df_modeling %>%
  group_by(posteam) %>%
  summarise(team_pass_rate = mean(is_pass), .groups = "drop") %>%
  mutate(team_pass_tendency = as.numeric(scale(team_pass_rate)))

def_stats <- df_modeling %>%
  group_by(defteam) %>%
  summarise(def_pass_rate_allowed = mean(is_pass), .groups = "drop") %>%
  mutate(def_pass_tendency = as.numeric(scale(def_pass_rate_allowed)))

df_modeling <- df_modeling %>%
  left_join(team_stats %>% select(posteam, team_pass_tendency), by = "posteam") %>%
  left_join(def_stats %>% select(defteam, def_pass_tendency), by = "defteam") %>%
  mutate(
    team_pass_tendency = replace_na(team_pass_tendency, 0),
    def_pass_tendency = replace_na(def_pass_tendency, 0)
  ) %>%
  select(-posteam, -defteam)

# ───────────────────────────────────────────────────────────────
# 7) Train/test split and preprocessing
# ───────────────────────────────────────────────────────────────
train_raw <- df_modeling %>% filter(season < 2023)
test_raw <- df_modeling %>% filter(season == 2023)

model_features <- df_modeling %>%
  select(-c(game_id, old_game_id, play_id, season, is_pass)) %>% 
  names()

train_data <- train_raw %>% select(is_pass, all_of(model_features))
test_data <- test_raw %>% select(is_pass, all_of(model_features))

# Ensure factor levels are consistent
factor_cols <- c("down_dist_cat", "form_cat", "score_situation", "yards_to_goal_bin", "timeout_situation")

for (col in factor_cols) {
  if (col %in% names(train_data)) {
    all_levels <- unique(c(as.character(train_data[[col]]), as.character(test_data[[col]])))
    train_data[[col]] <- factor(train_data[[col]], levels = all_levels)
    test_data[[col]] <- factor(test_data[[col]], levels = all_levels)
  }
}

# ───────────────────────────────────────────────────────────────
# 8) Create interaction terms (FIXED VERSION)
# ───────────────────────────────────────────────────────────────
create_interactions <- function(data) {
  data %>%
    mutate(
      down_x_dist = down * ydstogo,
      third_and_long = as.numeric(down == 3 & ydstogo >= 7),
      shotgun_x_down = shotgun_flag * down,
      heavy_in_redzone = heavy_set * as.numeric(yardline_100 <= 20),
      empty_on_third = empty_back * as.numeric(down == 3),
      nickel_on_third_down = as.numeric(is_nickel & down == 3),
      trailing_late = trailing_big * as.numeric(game_seconds_remaining <= 600),
      leading_late = leading_big * as.numeric(game_seconds_remaining <= 600)
    ) %>%
    # Now remove game_seconds_remaining after creating interactions
    select(-game_seconds_remaining)
}

train_data <- create_interactions(train_data)
test_data <- create_interactions(test_data)
# ───────────────────────────────────────────────────────────────
# 9) Model fitting with Optimized Hyperparameters
# ───────────────────────────────────────────────────────────────
# Remove zero variance columns
const_cols <- names(train_data)[sapply(train_data, function(x) length(unique(x)) < 2)]
if (length(const_cols) > 0) {
  train_data <- train_data %>% select(-all_of(const_cols))
  test_data <- test_data %>% select(-all_of(const_cols))
}

# Ensure test data has same columns as train data
test_data <- test_data %>% select(any_of(names(train_data)))

cat("Final train dimensions:", dim(train_data), "\n")
cat("Final test dimensions:", dim(test_data), "\n")

# Set up cross-validation
set.seed(42)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE,
  verboseIter = TRUE
)

# Convert target to factor
train_data$is_pass <- factor(train_data$is_pass, levels = c(0, 1), labels = c("Run", "Pass"))

# Optimal hyperparameters
best_tune <- data.frame(
  nrounds = 300,
  max_depth = 8,
  eta = 0.05,
  gamma = 0,
  colsample_bytree = 0.8,
  min_child_weight = 1,
  subsample = 0.75
)

cat("Training XGBoost model with optimal parameters...\n")
xgb_model <- train(
  is_pass ~ .,
  data = train_data,
  method = "xgbTree",
  trControl = ctrl,
  metric = "ROC",
  tuneGrid = best_tune,
  verbose = FALSE
)

# ───────────────────────────────────────────────────────────────
# 10) Model evaluation
# ───────────────────────────────────────────────────────────────
test_probs <- predict(xgb_model, test_data, type = "prob")[, "Pass"]
test_preds <- predict(xgb_model, test_data)

test_labels <- factor(test_data$is_pass, levels = c(0, 1), labels = c("Run", "Pass"))
conf_mat <- confusionMatrix(test_preds, test_labels, positive = "Pass")
roc_obj <- roc(test_labels, test_probs, quiet = TRUE)

cat("\n=== Final XGBoost MODEL PERFORMANCE (2023 TEST SET) ===\n")
cat("AUC:", round(auc(roc_obj), 4), "\n")
cat("Accuracy:", round(conf_mat$overall["Accuracy"], 4), "\n")
print(conf_mat$table)
cat("\nModel training completed successfully!\n")

# ───────────────────────────────────────────────────────────────
# 11) Enhanced Pass Rusher Evaluation
# ───────────────────────────────────────────────────────────────
cat("\n--- Starting Enhanced Pass Rusher Evaluation ---\n")

# Prepare evaluation dataset with predictions
eval_2023 <- df_evaluation %>% 
  filter(season == 2023) %>%
  mutate(pass_prob = test_probs)

# Load 2023 roster data
rosters_2023 <- tryCatch({
  load_rosters(2023) %>%
    select(gsis_id, full_name, position, team)
}, error = function(e) {
  cat("Warning: Could not load 2023 roster data.\n")
  data.frame()
})

# Filter participation data for 2023
if (nrow(parts) > 0) {
  parts_2023 <- parts %>%
    filter(substr(old_game_id, 1, 4) == "2023") %>%
    select(old_game_id, play_id, defense_players)
} else {
  parts_2023 <- data.frame()
}

# ───────────────────────────────────────────────────────────────
# Enhanced Pass Rusher Evaluation Function
# ───────────────────────────────────────────────────────────────
enhanced_rusher_eval <- function(eval_data, parts_data, rosters_data) {
  
  if (nrow(rosters_data) == 0 || nrow(parts_data) == 0) {
    cat("Missing required data for enhanced evaluation.\n")
    return(data.frame())
  }
  
  # Create player-level data
  player_level_plays <- eval_data %>%
    # Keep only actual pass plays
    filter(pass_attempt == 1 | sack == 1) %>%
    # Join participation data
    left_join(parts_data, by = c("old_game_id", "play_id")) %>%
    # Filter for plays with defensive players
    filter(!is.na(defense_players) & defense_players != "") %>%
    # Un-nest defender IDs
    separate_rows(defense_players, sep = ";") %>%
    rename(gsis_id = defense_players) %>%
    # Join roster data
    left_join(rosters_data, by = "gsis_id") %>%
    filter(!is.na(full_name))
  
  if (nrow(player_level_plays) == 0) {
    cat("No player-level data found.\n")
    return(data.frame())
  }
  
  # Enhanced evaluation
  enhanced_results <- player_level_plays %>%
    # Create predictability tiers
    mutate(
      predictability_tier = case_when(
        pass_prob >= 0.45 & pass_prob <= 0.55 ~ "Truly Unpredictable",
        pass_prob >= 0.35 & pass_prob <= 0.65 ~ "Somewhat Unpredictable", 
        pass_prob >= 0.25 & pass_prob <= 0.75 ~ "Moderately Predictable",
        TRUE ~ "Highly Predictable"
      ),
      
      # Situational difficulty weighting
      situation_weight = case_when(
        # High-pressure situations worth more
        (qtr >= 3 & abs(score_differential) <= 7) ~ 1.5,
        (down >= 3 & ydstogo >= 7) ~ 1.3,
        (game_seconds_remaining <= 300 & wp >= 0.3 & wp <= 0.7) ~ 1.4,
        (yardline_100 <= 20) ~ 1.2,  # Red zone
        TRUE ~ 1.0
      )
    ) %>%
    
    # Focus on challenging situations
    filter(predictability_tier %in% c("Truly Unpredictable", "Somewhat Unpredictable")) %>%
    filter(position %in% c("DE", "DT", "EDGE", "OLB", "ILB", "LB", "NT", "DL")) %>%
    
    # Enhanced disruption calculation
    mutate(
      # Base disruption values
      base_disruption = case_when(
        !is.na(sack_player_id) & gsis_id == sack_player_id ~ 3.0,
        !is.na(half_sack_1_player_id) & gsis_id == half_sack_1_player_id ~ 1.5,
        !is.na(half_sack_2_player_id) & gsis_id == half_sack_2_player_id ~ 1.5,
        !is.na(qb_hit_1_player_id) & gsis_id == qb_hit_1_player_id ~ 1.0,
        !is.na(qb_hit_2_player_id) & gsis_id == qb_hit_2_player_id ~ 1.0,
        TRUE ~ 0.0
      ),
      
      # Apply situational weighting
      weighted_disruption = base_disruption * situation_weight,
      
      # Additional context metrics
      is_blitz_situation = defenders_in_box >= 6,
      is_obvious_passing_down = (down == 3 & ydstogo >= 7) | (down == 4 & ydstogo >= 3),
      
      # Role-based expectations
      expected_rush_role = case_when(
        position %in% c("DE", "EDGE") ~ "Primary",
        position %in% c("DT", "NT") ~ "Interior", 
        position %in% c("OLB", "LB") ~ "Situational",
        TRUE ~ "Other"
      )
    ) %>%
    
    # Group by player and role
    group_by(full_name, position, team, expected_rush_role) %>%
    summarise(
      # Volume metrics
      total_unpredictable_snaps = n(),
      truly_unpredictable_snaps = sum(predictability_tier == "Truly Unpredictable"),
      
      # Performance metrics
      total_disruptions = sum(weighted_disruption, na.rm = TRUE),
      raw_disruptions = sum(base_disruption, na.rm = TRUE),
      
      # Situation-specific performance
      high_pressure_disruptions = sum(weighted_disruption[situation_weight > 1.2], na.rm = TRUE),
      high_pressure_snaps = sum(situation_weight > 1.2),
      
      # Context metrics
      blitz_snaps = sum(is_blitz_situation, na.rm = TRUE),
      obvious_pass_snaps = sum(is_obvious_passing_down, na.rm = TRUE),
      
      .groups = "drop"
    ) %>%
    
    # Calculate rates and efficiency metrics
    mutate(
      # Primary metrics
      disruption_rate = total_disruptions / total_unpredictable_snaps,
      raw_disruption_rate = raw_disruptions / total_unpredictable_snaps,
      
      # Situational efficiency
      clutch_disruption_rate = ifelse(high_pressure_snaps >= 5, 
                                      high_pressure_disruptions / high_pressure_snaps, 
                                      NA),
      
      # Volume-adjusted score (rewards both rate and volume)
      volume_adjusted_score = disruption_rate * log(total_unpredictable_snaps + 1),
      
      # Consistency metric (higher is better)
      opportunity_rate = truly_unpredictable_snaps / total_unpredictable_snaps
    ) %>%
    
    # Filter for meaningful sample sizes
    filter(total_unpredictable_snaps >= 20) %>%
    
    # Create composite score
    mutate(
      # Weighted composite considering rate, volume, and clutch performance
      composite_score = (disruption_rate * 0.4) + 
        (volume_adjusted_score * 0.3) + 
        (coalesce(clutch_disruption_rate, disruption_rate) * 0.3)
    ) %>%
    
    # Sort by composite score
    arrange(desc(composite_score)) %>%
    
    # Clean up for display
    select(
      Player = full_name,
      Team = team,
      Position = position,
      Role = expected_rush_role,
      Composite_Score = composite_score,
      Disruption_Rate = disruption_rate,
      Clutch_Rate = clutch_disruption_rate,
      Total_Disruptions = total_disruptions,
      Snaps = total_unpredictable_snaps,
      Truly_Unpredictable = truly_unpredictable_snaps,
      High_Pressure_Snaps = high_pressure_snaps
    )
  
  return(enhanced_results)
}

# ───────────────────────────────────────────────────────────────
# Team-Level Pass Rush Effectiveness Function (CONTINUED)
# ───────────────────────────────────────────────────────────────
team_pass_rush_eval <- function(eval_data) {
  
  team_analysis <- eval_data %>%
    filter(pass_attempt == 1 | sack == 1) %>%
    mutate(
      predictability_tier = case_when(
        pass_prob >= 0.45 & pass_prob <= 0.55 ~ "Truly Unpredictable",
        pass_prob >= 0.35 & pass_prob <= 0.65 ~ "Somewhat Unpredictable", 
        pass_prob >= 0.25 & pass_prob <= 0.75 ~ "Moderately Predictable",
        TRUE ~ "Highly Predictable"
      ),
      
      situation_weight = case_when(
        (qtr >= 3 & abs(score_differential) <= 7) ~ 1.5,
        (down >= 3 & ydstogo >= 7) ~ 1.3,
        (game_seconds_remaining <= 300 & wp >= 0.3 & wp <= 0.7) ~ 1.4,
        (yardline_100 <= 20) ~ 1.2,
        TRUE ~ 1.0
      ),
      
      team_disruption = case_when(
        sack == 1 ~ 3.0,
        qb_hit == 1 ~ 1.0,
        TRUE ~ 0.0
      ),
      
      weighted_team_disruption = team_disruption * situation_weight
    ) %>%
    
    group_by(defteam) %>%
    summarise(
      total_pass_plays = n(),
      unpredictable_plays = sum(predictability_tier %in% c("Truly Unpredictable", "Somewhat Unpredictable")),
      
      total_sacks = sum(sack, na.rm = TRUE),
      total_qb_hits = sum(qb_hit, na.rm = TRUE),
      total_disruptions = sum(weighted_team_disruption, na.rm = TRUE),
      
      high_pressure_plays = sum(situation_weight > 1.2),
      high_pressure_disruptions = sum(weighted_team_disruption[situation_weight > 1.2], na.rm = TRUE),
      
      avg_defenders_in_box = mean(defenders_in_box, na.rm = TRUE),
      
      .groups = "drop"
    ) %>%
    
    mutate(
      sack_rate = total_sacks / total_pass_plays,
      qb_hit_rate = total_qb_hits / total_pass_plays,
      disruption_rate = total_disruptions / total_pass_plays,
      
      unpredictable_disruption_rate = ifelse(unpredictable_plays > 0, 
                                             total_disruptions / unpredictable_plays, 
                                             0),
      
      clutch_disruption_rate = ifelse(high_pressure_plays > 0,
                                      high_pressure_disruptions / high_pressure_plays,
                                      0),
      
      pass_rush_efficiency = (sack_rate * 3 + qb_hit_rate) / avg_defenders_in_box,
      
      composite_team_score = (disruption_rate * 0.4) + 
        (unpredictable_disruption_rate * 0.3) + 
        (clutch_disruption_rate * 0.3)
    ) %>%
    
    arrange(desc(composite_team_score)) %>%
    
    select(
      Team = defteam,
      Composite_Score = composite_team_score,
      Disruption_Rate = disruption_rate,
      Sack_Rate = sack_rate,
      QB_Hit_Rate = qb_hit_rate,
      Clutch_Rate = clutch_disruption_rate,
      Pass_Rush_Efficiency = pass_rush_efficiency,
      Total_Plays = total_pass_plays,
      Unpredictable_Plays = unpredictable_plays
    )
  
  return(team_analysis)
}

# ───────────────────────────────────────────────────────────────
# 12) Execute Enhanced Evaluations
# ───────────────────────────────────────────────────────────────

# Run enhanced player evaluation
cat("Running enhanced pass rusher evaluation...\n")
enhanced_rushers <- enhanced_rusher_eval(eval_2023, parts_2023, rosters_2023)

# Run team-level evaluation
cat("Running team pass rush evaluation...\n")
team_rush_results <- team_pass_rush_eval(eval_2023)

# ───────────────────────────────────────────────────────────────
# 13) Display Results
# ───────────────────────────────────────────────────────────────

# Display top individual pass rushers
if (nrow(enhanced_rushers) > 0) {
  cat("\n=== TOP 20 PASS RUSHERS (2023) - ENHANCED EVALUATION ===\n")
  print(kable(head(enhanced_rushers, 20), digits = 4, format = "simple"))
  
  # Position-specific leaders
  cat("\n=== POSITION-SPECIFIC LEADERS ===\n")
  pos_leaders <- enhanced_rushers %>%
    group_by(Position) %>%
    slice_head(n = 3) %>%
    ungroup() %>%
    arrange(Position, desc(Composite_Score))
  
  print(kable(pos_leaders, digits = 4, format = "simple"))
  
} else {
  cat("No enhanced rusher data available.\n")
}

# Display team results
if (nrow(team_rush_results) > 0) {
  cat("\n=== TEAM PASS RUSH RANKINGS (2023) ===\n")
  print(kable(team_rush_results, digits = 4, format = "simple"))
} else {
  cat("No team pass rush data available.\n")
}

# ───────────────────────────────────────────────────────────────
# 14) Model Feature Importance
# ───────────────────────────────────────────────────────────────

# Extract feature importance
feature_importance <- varImp(xgb_model, scale = FALSE)
importance_df <- data.frame(
  Feature = rownames(feature_importance$importance),
  Importance = feature_importance$importance[, 1]
) %>%
  arrange(desc(Importance)) %>%
  head(20)

cat("\n=== TOP 20 MOST IMPORTANT FEATURES ===\n")
print(kable(importance_df, digits = 4, format = "simple"))

# ───────────────────────────────────────────────────────────────
# 15) Generate Predictability Analysis
# ───────────────────────────────────────────────────────────────

predictability_analysis <- eval_2023 %>%
  filter(pass_attempt == 1 | sack == 1) %>%
  mutate(
    predictability_tier = case_when(
      pass_prob >= 0.45 & pass_prob <= 0.55 ~ "Truly Unpredictable",
      pass_prob >= 0.35 & pass_prob <= 0.65 ~ "Somewhat Unpredictable", 
      pass_prob >= 0.25 & pass_prob <= 0.75 ~ "Moderately Predictable",
      TRUE ~ "Highly Predictable"
    )
  ) %>%
  group_by(predictability_tier, posteam) %>%
  summarise(
    plays = n(),
    sack_rate = mean(sack, na.rm = TRUE),
    qb_hit_rate = mean(qb_hit, na.rm = TRUE),
    avg_pass_prob = mean(pass_prob, na.rm = TRUE),
    .groups = "drop"
  ) %>%
  arrange(predictability_tier, desc(sack_rate))

cat("\n=== PREDICTABILITY ANALYSIS BY TEAM ===\n")
print(kable(head(predictability_analysis, 25), digits = 4, format = "simple"))

# ───────────────────────────────────────────────────────────────
# 16) Summary Statistics
# ───────────────────────────────────────────────────────────────

cat("\n=== SUMMARY STATISTICS ===\n")
cat("Total 2023 pass plays analyzed:", nrow(eval_2023 %>% filter(pass_attempt == 1 | sack == 1)), "\n")
cat("Players with sufficient data:", nrow(enhanced_rushers), "\n")
cat("Teams analyzed:", nrow(team_rush_results), "\n")
cat("Model AUC:", round(auc(roc_obj), 4), "\n")
cat("Average pass probability:", round(mean(eval_2023$pass_prob, na.rm = TRUE), 4), "\n")

# Distribution of predictability
pred_dist <- eval_2023 %>%
  filter(pass_attempt == 1 | sack == 1) %>%
  mutate(
    predictability_tier = case_when(
      pass_prob >= 0.45 & pass_prob <= 0.55 ~ "Truly Unpredictable",
      pass_prob >= 0.35 & pass_prob <= 0.65 ~ "Somewhat Unpredictable", 
      pass_prob >= 0.25 & pass_prob <= 0.75 ~ "Moderately Predictable",
      TRUE ~ "Highly Predictable"
    )
  ) %>%
  count(predictability_tier) %>%
  mutate(pct = n / sum(n) * 100)

cat("\n=== PREDICTABILITY DISTRIBUTION ===\n")
print(kable(pred_dist, digits = 2, format = "simple"))

cat("\n=== ANALYSIS COMPLETE ===\n")
cat("Enhanced pass rusher evaluation completed successfully!\n")
cat("Results show performance in unpredictable situations with contextual weighting.\n")