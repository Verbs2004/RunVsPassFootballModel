# ───────────────────────────────────────────────────────────────
# Final NFL Play Prediction Model
# ───────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────
# 0) Libraries
# ───────────────────────────────────────────────────────────────
library(nflreadr)
library(dplyr); library(tidyr); library(stringr)
library(pROC); library(readr); library(purrr)
library(ggplot2); library(slider)
library(fastDummies)
library(caret) # For better model validation
library(xgboost) # Load the xgboost library

# ───────────────────────────────────────────────────────────────
# 1) Load core tables (2016-23)
# ───────────────────────────────────────────────────────────────
pbp <- load_pbp(2016:2023)

sched <- load_schedules(2016:2023) %>%
  select(game_id,
         sched_roof = roof,
         sched_temp = temp,
         sched_wind_speed = wind)

# --- Load all necessary columns from participation data ---
parts <- tryCatch({
  load_participation(2016:2023) %>%
    select(old_game_id, play_id, defenders_in_box, offense_personnel, defense_personnel)
}, error = function(e) {
  cat("Warning: Could not load participation data. It will be estimated.\n")
  data.frame() # Return empty frame if error
})


# Load advanced play data from BDB if available
plays <- tryCatch({
  read_csv("plays.csv", show_col_types = FALSE)
}, error = function(e) {
  cat("Warning: plays.csv not found. Related features will be estimated.\n")
  data.frame() # Return empty frame if error
})

# Add absolute yardline if missing
if (!"absolute_yardline_number" %in% names(pbp)) {
  pbp <- pbp %>% mutate(
    absolute_yardline_number = if_else(side_of_field == posteam, 100 - yardline_100, yardline_100)
  )
}

# ───────────────────────────────────────────────────────────────
# 2) Helper function for parsing personnel
# ───────────────────────────────────────────────────────────────
parse_pers <- function(x, pos) {
  # Return 0 if input is invalid
  if (is.na(x) || x == "" || is.null(x)) return(0L)
  
  # Extract the number for a given position (e.g., "2" from "2 RB")
  pattern <- sprintf("(\\d+)\\s*%s", pos)
  matches <- str_extract(x, pattern)
  
  if (is.na(matches)) return(0L)
  
  num <- as.integer(str_extract(matches, "\\d+"))
  return(if_else(is.na(num), 0L, num))
}


# ───────────────────────────────────────────────────────────────
# 3) Join data sources
# ───────────────────────────────────────────────────────────────
# Create a base dataframe for feature engineering
df_base <- pbp %>%
  # Filter for quality data first
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

# --- FIXED: Join participation data using the correct ID ---
if (nrow(parts) > 0) {
  parts_j <- parts %>%
    # Rename columns to avoid conflicts
    rename(
      parts_off_pers = offense_personnel,
      parts_def_pers = defense_personnel) %>%
    # Ensure the join key is character, just in case
    mutate(old_game_id = as.character(old_game_id))
  
  # Join using old_game_id, which is common to both datasets
  df_base <- df_base %>%
    mutate(old_game_id = as.character(old_game_id)) %>%
    left_join(parts_j, by = c("old_game_id", "play_id"))
}


# ───────────────────────────────────────────────────────────────
# 4) Feature engineering with ADVANCED personnel features
# ───────────────────────────────────────────────────────────────
df <- df_base %>%
  arrange(game_id, fixed_drive, play_id) %>%
  group_by(game_id, posteam) %>%
  mutate(drive_play_idx = row_number()) %>%
  ungroup() %>%
  
  # Create target variable
  mutate(is_pass = as.numeric(play_type == "pass")) %>%
  
  # --- Create Game Flow / Sequential Features ---
  group_by(game_id, fixed_drive) %>%
  mutate(
    is_first_play_of_drive = if_else(drive_play_idx == 1, 1, 0),
    prev_play_was_pass = lag(is_pass, 1),
    yards_gained_on_prev_play = lag(yards_gained, 1)
  ) %>%
  ungroup() %>%
  mutate(
    # Fill NAs for the first play of each drive
    prev_play_was_pass = replace_na(prev_play_was_pass, 0),
    yards_gained_on_prev_play = replace_na(yards_gained_on_prev_play, 0)
  ) %>%
  
  # --- Safely create and parse ALL personnel counts ---
  mutate(
    # Step 1: Safely create the personnel columns.
    offense_personnel = if ("parts_off_pers" %in% names(.)) parts_off_pers else NA_character_,
    defense_personnel = if ("parts_def_pers" %in% names(.)) parts_def_pers else NA_character_,
    
    # Step 2: Now that the columns are guaranteed to exist, fill NAs with defaults.
    offense_personnel = replace_na(offense_personnel, "1 RB, 1 TE, 3 WR"),
    defense_personnel = replace_na(defense_personnel, "4 DL, 3 LB, 4 DB"),
    
    # Step 3: Parse the guaranteed columns.
    RB_off_P = map_int(offense_personnel, ~parse_pers(.x, "RB")),
    TE_off_P = map_int(offense_personnel, ~parse_pers(.x, "TE")),
    WR_off_P = map_int(offense_personnel, ~parse_pers(.x, "WR")),
    DL_def_P = map_int(defense_personnel, ~parse_pers(.x, "DL")),
    LB_def_P = map_int(defense_personnel, ~parse_pers(.x, "LB")),
    DB_def_P = map_int(defense_personnel, ~parse_pers(.x, "DB")),
    defenders_in_box = coalesce(defenders_in_box, DL_def_P + LB_def_P, 7)
  ) %>%
  
  # --- Create ADVANCED derived personnel features ---
  mutate(
    # Offensive intent
    heavy_set = as.numeric((RB_off_P + TE_off_P) >= 3),
    empty_back = as.numeric(RB_off_P == 0),
    
    # Defensive posture
    is_nickel = as.numeric(DB_def_P == 5),
    is_dime = as.numeric(DB_def_P >= 6),
    
    # Mismatch / Advantage calculation
    skill_diff_P = WR_off_P - DB_def_P,
    # Assumes 5 OL. Positive number means offense has blocking advantage in the box.
    box_advantage_off = (5 + TE_off_P) - defenders_in_box
  ) %>%
  
  # --- CONTEXTUAL & SITUATIONAL FEATURES ---
  mutate(
    # Add probability and Vegas features
    ep = as.numeric(ep),
    spread_line = as.numeric(spread_line),
    total_line = as.numeric(total_line),
    
    # Fill NAs for new features
    ep = replace_na(ep, 0),
    spread_line = replace_na(spread_line, 0),
    total_line = replace_na(total_line, mean(total_line, na.rm=T)),
    
    
    goal_to_go = as.numeric(goal_to_go),
    
    # Categorical timeout feature
    timeout_situation = case_when(
      posteam_timeouts_remaining > defteam_timeouts_remaining ~ "Offense Advantage",
      posteam_timeouts_remaining < defteam_timeouts_remaining ~ "Defense Advantage",
      posteam_timeouts_remaining == 0 & defteam_timeouts_remaining == 0 ~ "None Left",
      TRUE ~ "Equal Timeouts"
    ),
    
    play_clock_at_snap = as.numeric(play_clock),
    
    # Fill NAs for other new features
    goal_to_go = replace_na(goal_to_go, 0),
    play_clock_at_snap = replace_na(play_clock_at_snap, 15),
    
    form_cat = case_when(
      exists("offenseFormation") & str_detect(toupper(offenseFormation), "SHOTGUN") ~ "Shotgun",
      exists("offenseFormation") & str_detect(toupper(offenseFormation), "EMPTY") ~ "Empty",
      exists("offenseFormation") & str_detect(toupper(offenseFormation), "I_FORM|JUMBO") ~ "I-Form",
      exists("offenseFormation") & str_detect(toupper(offenseFormation), "PISTOL") ~ "Pistol",
      exists("offenseFormation") & str_detect(toupper(offenseFormation), "SINGLEBACK") ~ "Singleback",
      shotgun == 1 ~ "Shotgun",
      TRUE ~ "Other"
    ),
    two_minute_drill = as.numeric((qtr == 2 | qtr == 4) & quarter_seconds_remaining <= 120),
    four_minute_drill = as.numeric(qtr == 4 & quarter_seconds_remaining <= 240 & score_differential > 0),
    shotgun_flag = as.numeric(coalesce(shotgun, 0) == 1),
    no_huddle_flag = as.numeric(coalesce(no_huddle, 0) == 1),
    wp = if_else(posteam == home_team, home_wp, away_wp),
    wp = coalesce(wp, 0.5),
    wp_tier = case_when(
      wp < 0.2 ~ "desperate", wp < 0.4 ~ "low",
      wp > 0.8 ~ "dominant", wp > 0.6 ~ "high",
      TRUE ~ "medium"
    ),
    late_desperation = as.numeric(game_seconds_remaining < 300 & wp < 0.2),
    yards_to_goal_bin = case_when(
      yardline_100 <= 4 ~ "goalline", yardline_100 <= 10 ~ "redzone_fringe",
      yardline_100 <= 20 ~ "redzone", yardline_100 <= 50 ~ "own_territory",
      TRUE ~ "backed_up"
    ),
    trailing_big = as.numeric(score_differential <= -14),
    leading_big = as.numeric(score_differential >= 14),
    score_situation = case_when(
      leading_big == 1 ~ "leading_big", trailing_big == 1 ~ "trailing_big",
      abs(score_differential) <= 4 ~ "close",
      score_differential > 4 ~ "leading", TRUE ~ "trailing"
    ),
    down_dist_cat = case_when(
      down == 1 ~ "first",
      down == 2 & ydstogo <= 3 ~ "second_short", down == 2 & ydstogo >= 8 ~ "second_long",
      down == 3 & ydstogo <= 3 ~ "third_short", down == 3 & ydstogo >= 8 ~ "third_long",
      down == 4 & ydstogo <= 3 ~ "fourth_short", down == 4 & ydstogo >= 8 ~ "fourth_long",
      TRUE ~ "medium_yardage"
    )
  ) %>%
  
  # --- FINAL CLEANUP & SELECTION ---
  select(
    game_id, season, is_pass,
    # Core situational
    down, ydstogo, down_dist_cat, yardline_100, score_differential,
    # **UPDATED** Personnel Features
    heavy_set, empty_back, is_nickel, is_dime,
    skill_diff_P, defenders_in_box, box_advantage_off,
    # Formation and pre-snap
    form_cat, shotgun_flag, no_huddle_flag,
    # Game context
    two_minute_drill, four_minute_drill, wp, late_desperation,
    trailing_big, leading_big, score_situation,
    # **NEWLY ADDED** Features
    goal_to_go, timeout_situation, play_clock_at_snap, ep,
    is_first_play_of_drive, prev_play_was_pass, yards_gained_on_prev_play,
    spread_line, total_line,
    # Field position
    yards_to_goal_bin,
    # Teams
    posteam, defteam,
    game_seconds_remaining
  )

cat("Total rows after cleaning:", nrow(df), "\n")
cat("Pass rate:", round(mean(df$is_pass), 3), "\n")

# ───────────────────────────────────────────────────────────────
# 5) Team tendency encoding
# ───────────────────────────────────────────────────────────────
team_stats <- df %>%
  group_by(posteam) %>%
  summarise(team_pass_rate = mean(is_pass), .groups = "drop") %>%
  mutate(team_pass_tendency = scale(team_pass_rate)[,1])

def_stats <- df %>%
  group_by(defteam) %>%
  summarise(def_pass_rate_allowed = mean(is_pass), .groups = "drop") %>%
  mutate(def_pass_tendency = scale(def_pass_rate_allowed)[,1])

df <- df %>%
  left_join(team_stats %>% select(posteam, team_pass_tendency), by = "posteam") %>%
  left_join(def_stats %>% select(defteam, def_pass_tendency), by = "defteam") %>%
  mutate(
    team_pass_tendency = replace_na(team_pass_tendency, 0),
    def_pass_tendency = replace_na(def_pass_tendency, 0)
  ) %>%
  select(-posteam, -defteam)


# ───────────────────────────────────────────────────────────────
# 6) Train/test split and preprocessing
# ───────────────────────────────────────────────────────────────
train_raw <- df %>% filter(season < 2023)
test_raw <- df %>% filter(season == 2023)

train_data <- train_raw %>% select(-game_id, -season)
test_data <- test_raw %>% select(-game_id, -season)

factor_cols <- c("down_dist_cat", "form_cat", "score_situation", "yards_to_goal_bin", "timeout_situation")

for (col in factor_cols) {
  if (col %in% names(train_data)) {
    all_levels <- unique(c(train_data[[col]], test_data[[col]]))
    train_data[[col]] <- factor(train_data[[col]], levels = all_levels)
    test_data[[col]] <- factor(test_data[[col]], levels = all_levels)
  }
}

# ───────────────────────────────────────────────────────────────
# 7) Create interaction terms
# ───────────────────────────────────────────────────────────────
create_interactions <- function(data) {
  data %>%
    mutate(
      down_x_dist = down * ydstogo,
      third_and_long = as.numeric(down == 3 & ydstogo >= 7),
      shotgun_x_down = shotgun_flag * down,
      heavy_in_redzone = heavy_set * as.numeric(yardline_100 <= 20),
      empty_on_third = empty_back * as.numeric(down == 3),
      # **NEW** Interaction with advanced personnel feature
      nickel_on_third_down = as.numeric(is_nickel & down == 3),
      trailing_late = trailing_big * as.numeric(game_seconds_remaining <= 600),
      leading_late = leading_big * as.numeric(game_seconds_remaining <= 600)
    ) %>%
    select(-game_seconds_remaining)
}

train_data <- create_interactions(train_data)
test_data <- create_interactions(test_data)


# ───────────────────────────────────────────────────────────────
# 8) Model fitting with Optimized Hyperparameters
# ───────────────────────────────────────────────────────────────
const_cols <- names(train_data)[sapply(train_data, n_distinct) < 2]
train_data <- train_data %>% select(-any_of(const_cols))
test_data <- test_data %>% select(-any_of(const_cols))

test_data <- test_data %>% select(any_of(names(train_data)))

cat("Final train dimensions:", dim(train_data), "\n")
cat("Final test dimensions:", dim(test_data), "\n")

set.seed(42)
ctrl <- trainControl(
  method = "cv",
  number = 5,
  classProbs = TRUE,
  summaryFunction = twoClassSummary,
  savePredictions = "final",
  allowParallel = TRUE,
  verboseIter = TRUE # See progress
)

train_data$is_pass <- factor(train_data$is_pass, levels = c(0, 1), labels = c("Run", "Pass"))
test_labels <- factor(test_data$is_pass, levels = c(0, 1), labels = c("Run", "Pass"))

# --- KEY CHANGE: Use the single best set of hyperparameters found previously ---
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
  tuneGrid = best_tune, # Use the single best tune
  verbose = FALSE
)

# ───────────────────────────────────────────────────────────────
# 9) Model evaluation
# ───────────────────────────────────────────────────────────────
test_probs <- predict(xgb_model, test_data, type = "prob")[, "Pass"]
test_preds <- predict(xgb_model, test_data)

conf_mat <- confusionMatrix(test_preds, test_labels, positive = "Pass")
roc_obj <- roc(test_labels, test_probs, quiet = TRUE)

cat("\n=== Final XGBoost MODEL PERFORMANCE (2023 TEST SET) ===\n")
cat("AUC:", round(auc(roc_obj), 4), "\n")
cat("Accuracy:", round(conf_mat$overall["Accuracy"], 4), "\n")

print(conf_mat$table)

var_imp <- varImp(xgb_model, scale = FALSE)
cat("\n=== TOP 40 MOST IMPORTANT VARIABLES ===\n")
print(plot(var_imp, top = 40))

cat("\nModel training and evaluation completed successfully!\n")
