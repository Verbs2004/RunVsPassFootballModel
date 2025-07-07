# --- Load Libraries ---
library(shiny)
library(lubridate)
library(ggplot2)
library(sportyR)
library(tidyverse)
library(dplyr)
library(ggfootball)
library(nflreadr)
library(broom)
library(pROC)

# --- Field Background ---
field_params <- list(
  field_apron = "springgreen3",
  field_border = "springgreen3",
  offensive_endzone = "springgreen3",
  defensive_endzone = "springgreen3",
  offensive_half = "springgreen3",
  defensive_half = "springgreen3"
)
field_background <- geom_football(
  league = "nfl",
  display_range = "in_bounds_only",
  x_trans = 60,
  y_trans = 26.6667,
  xlims = c(0, 120),
  color_updates = field_params
)

# --- Scrambles ---
scrambles <- plays |>
  filter(dropbackType == "SCRAMBLE", passResult == "R") |>
  select(gameId, playId) |>
  mutate(is_scramble = TRUE)

# --- Identify Run Plays ---
run_plays <- player_play |> 
  filter(hadRushAttempt == 1) |> 
  select(gameId, playId, ball_carrier_id = nflId, ball_carrier_team = teamAbbr) |>
  left_join(scrambles, by = c("gameId", "playId")) |>
  filter(is.na(is_scramble)) |>
  mutate(is_run = 1)

# --- Build model_data ---
run_plays <- run_plays |> rename(run_label = is_run)

model_data <- plays |>
  left_join(run_plays |> mutate(run_label = 1), by = c("gameId", "playId")) |>
  left_join(games |> select(gameId, homeTeamAbbr, visitorTeamAbbr, week), by = "gameId") |>
  mutate(is_run = ifelse(is.na(run_label), 0, 1))

# --- Prepare model_data with pre-snap features only ---
model_data <- model_data |>
  filter(
    !is.na(gameClock),
    !is.na(down),
    !is.na(yardsToGo),
    !is.na(is_run),
    !is.na(quarter),
    !is.na(possessionTeam),
    !is.na(yardlineNumber),
    !is.na(yardlineSide)
  ) |>
  mutate(
    clock_str = str_sub(as.character(gameClock), 1, 5),
    clock_seconds = as.numeric(ms(clock_str)),
    score_diff = case_when(
      possessionTeam == homeTeamAbbr ~ preSnapHomeScore - preSnapVisitorScore,
      possessionTeam == visitorTeamAbbr ~ preSnapVisitorScore - preSnapHomeScore,
      TRUE ~ NA_real_
    ),
    yardline = ifelse(possessionTeam == yardlineSide | yardlineNumber == 50,
                      yardlineNumber,
                      100 - yardlineNumber),
    quarter = factor(quarter)
  ) |>
  select(gameId, playId, is_run, week, quarter, down, yardsToGo, clock_seconds, score_diff, yardline)

# --- Rolling CV ---
run_week_cv <- function(holdout_week) {
  train <- model_data |> filter(week < holdout_week)
  test  <- model_data |> filter(week == holdout_week)
  
  if (nrow(train) == 0 || nrow(test) == 0) return(NULL)
  
  model <- glm(
    is_run ~ quarter + down + yardsToGo + clock_seconds + score_diff + yardline,
    data = train,
    family = binomial()
  )
  
  test <- test |>
    mutate(
      pred_prob = predict(model, newdata = test, type = "response"),
      pred_bin = ifelse(pred_prob > 0.5, 1, 0),
      fold_week = holdout_week
    ) |>
    select(gameId, playId, is_run, pred_prob, pred_bin, fold_week)
  
  return(test)
}

# --- Run CV ---
cv_preds <- map_dfr(sort(unique(model_data$week)), run_week_cv)

# --- Evaluate ---
cv_preds |>
  summarise(
    accuracy = mean(pred_bin == is_run, na.rm = TRUE),
    brier_score = mean((pred_prob - is_run)^2, na.rm = TRUE)
  )

pROC::roc(cv_preds$is_run, cv_preds$pred_prob) |> pROC::auc()

cv_preds <- map_dfr(sort(unique(model_data$week)), run_week_cv) |>
  mutate(expected_pass_prob = 1 - pred_prob)

pass_rushers <- player_play |>
  filter(wasInitialPassRusher == 1) |>
  select(gameId, playId, nflId, teamAbbr, causedPressure, quarterbackHit,
         sackYardsAsDefense, timeToPressureAsPassRusher, getOffTimeAsPassRusher)

pass_rushers <- pass_rushers |>
  left_join(cv_preds |> select(gameId, playId, expected_pass_prob, is_run), by = c("gameId", "playId")) |>
  filter(!is.na(expected_pass_prob))  # drop any unmatched plays

pass_rusher_summary <- pass_rushers |>
  group_by(nflId, teamAbbr) |>
  summarise(
    plays = n(),
    avg_expected_pass = mean(expected_pass_prob),
    true_pass_rate = mean(1 - is_run),
    pressure_rate = mean(causedPressure == TRUE, na.rm = TRUE),
    hit_rate = mean(quarterbackHit == 1, na.rm = TRUE),
    avg_getoff = mean(getOffTimeAsPassRusher, na.rm = TRUE),
    avg_time_to_pressure = mean(timeToPressureAsPassRusher[causedPressure == TRUE], na.rm = TRUE),
    total_sacks = sum(sackYardsAsDefense > 0, na.rm = TRUE)
  ) |>
  filter(plays >= 20)  # Optional: Filter out small sample sizes

