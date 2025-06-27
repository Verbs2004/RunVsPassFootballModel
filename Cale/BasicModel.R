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
field_background

# --- Preprocess Plays ---
plays <- plays |>
  mutate(
    ytg_bin = case_when(
      yardsToGo <= 1 ~ "Short (0-1)",
      yardsToGo <= 3 ~ "Short (2-3)",
      yardsToGo <= 6 ~ "Medium (4-6)",
      yardsToGo <= 10 ~ "Medium (7-10)",
      yardsToGo <= 20 ~ "Long (11-20)",
      TRUE ~ "Very Long (21+)"
    )
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

# --- Team Run % ---
offense_run_props <- plays |>
  group_by(possessionTeam) |>
  summarise(team_plays = n()) |>
  left_join(
    run_plays |> group_by(ball_carrier_team) |> summarise(team_run_plays = n()) |> rename(possessionTeam = ball_carrier_team),
    by = "possessionTeam"
  ) |>
  mutate(
    team_run_plays = ifelse(is.na(team_run_plays), 0, team_run_plays),
    team_run_prop = team_run_plays / team_plays
  )

defense_run_props <- plays |>
  group_by(defensiveTeam) |>
  summarise(defense_plays = n()) |>
  left_join(
    run_plays |>
      left_join(plays |> select(gameId, playId, defensiveTeam), by = c("gameId", "playId")) |>
      group_by(defensiveTeam) |>
      summarise(defense_run_plays = n()),
    by = "defensiveTeam"
  ) |>
  mutate(
    defense_run_plays = ifelse(is.na(defense_run_plays), 0, defense_run_plays),
    defense_run_prop = defense_run_plays / defense_plays
  )

# --- Build model_data ---
model_data <- plays |>
  select(-is_run) |>
  left_join(run_plays, by = c("gameId", "playId")) |>
  mutate(is_run = ifelse(is.na(is_run), 0, is_run))

model_data <- model_data |>
  left_join(
    offense_run_props |> select(possessionTeam, offense_run_prop = team_run_prop),
    by = "possessionTeam"
  ) |>
  left_join(
    defense_run_props |> select(defensiveTeam, defense_run_prop),
    by = "defensiveTeam"
  ) |>
  left_join(games |> select(gameId, homeTeamAbbr, visitorTeamAbbr), by = "gameId")

model_data <- model_data |>
  left_join(games |> select(gameId, week), by = "gameId")

# --- Specific Training Data ---
# train_data <- model_data |> filter(week <= 5)
# test_data  <- model_data |> filter(week >= 6)
# 
# ltrain_data <- train_data |>
#   filter(!is.na(gameClock), !is.na(down), !is.na(yardsToGo)) |>
#   mutate(
#     clock_str = str_sub(as.character(gameClock), 1, 5),
#     clock_seconds = as.numeric(ms(clock_str)),
#     quarter = factor(quarter),
#     score_diff = case_when(
#       possessionTeam == homeTeamAbbr ~ preSnapHomeScore - preSnapVisitorScore,
#       possessionTeam == visitorTeamAbbr ~ preSnapVisitorScore - preSnapHomeScore,
#       TRUE ~ NA_real_),
#     yardline = ifelse(possessionTeam == yardlineSide | yardlineNumber == 50, yardlineNumber, 100 - yardlineNumber)
#   ) |>
#   select(is_run, quarter, down, yardsToGo, clock_seconds, score_diff, yardline,
#          offense_run_prop, defense_run_prop, team_situational_run_prop)
# 
# run_model <- glm(
#   is_run ~ quarter + down + yardsToGo + clock_seconds + score_diff + yardline + 
#     offense_run_prop + defense_run_prop + team_situational_run_prop,
#   data = ltrain_data,
#   family = binomial()
# )
# 
# ltest_data <- test_data |>
#   filter(!is.na(gameClock), !is.na(down), !is.na(yardsToGo)) |>
#   mutate(
#     clock_str = str_sub(as.character(gameClock), 1, 5),
#     clock_seconds = as.numeric(ms(clock_str)),
#     quarter = factor(quarter, levels = levels(ltrain_data$quarter)),  # ensure consistent levels
#     score_diff = case_when(
#       possessionTeam == homeTeamAbbr ~ preSnapHomeScore - preSnapVisitorScore,
#       possessionTeam == visitorTeamAbbr ~ preSnapVisitorScore - preSnapHomeScore,
#       TRUE ~ NA_real_),
#     yardline = ifelse(possessionTeam == yardlineSide | yardlineNumber == 50, yardlineNumber, 100 - yardlineNumber)
#   ) |>
#   select(is_run, quarter, down, yardsToGo, clock_seconds, score_diff, yardline,
#          offense_run_prop, defense_run_prop, team_situational_run_prop)
# 
# ltest_data <- ltest_data |>
#   mutate(
#     pred_run_prob = predict(run_model, newdata = ltest_data, type = "response"),
#     pred_run_binary = ifelse(pred_run_prob > 0.5, 1, 0)
#   )
# 
# # Accuracy
# mean(ltest_data$pred_run_binary == ltest_data$is_run)
# 
# # AUC
# library(pROC)
# roc(ltest_data$is_run, ltest_data$pred_run_prob)$auc

# --- Add Situational Run Props ---
team_situational_run_props <- plays |>
  filter(!is.na(quarter), !is.na(down), !is.na(yardsToGo)) |>
  group_by(possessionTeam, quarter, down, ytg_bin) |>
  summarise(
    total_plays = n(),
    run_plays = sum(is_run),
    run_prop = run_plays / total_plays,
    .groups = "drop"
  )

model_data <- model_data |>
  mutate(
    ytg_bin = case_when(
      yardsToGo <= 1 ~ "Short (0-1)",
      yardsToGo <= 3 ~ "Short (2-3)",
      yardsToGo <= 6 ~ "Medium (4-6)",
      yardsToGo <= 10 ~ "Medium (7-10)",
      yardsToGo <= 20 ~ "Long (11-20)",
      TRUE ~ "Very Long (21+)"
    )
  ) |>
  left_join(team_situational_run_props, 
            by = c("possessionTeam", "quarter", "down", "ytg_bin")) |>
  rename(team_situational_run_prop = run_prop)

# --- Final Model Dataset ---
lmodel_data <- model_data |>
  filter(!is.na(gameClock), !is.na(down), !is.na(yardsToGo)) |>
  mutate(
    clock_str = str_sub(as.character(gameClock), 1, 5),
    clock_seconds = as.numeric(ms(clock_str)),
    minute_remaining = floor(clock_seconds / 60),
    quarter = factor(quarter),
    score_diff = case_when(
      possessionTeam == homeTeamAbbr ~ preSnapHomeScore - preSnapVisitorScore,
      possessionTeam == visitorTeamAbbr ~ preSnapVisitorScore - preSnapHomeScore,
      TRUE ~ NA_real_),
    yardline = ifelse(possessionTeam == yardlineSide | yardlineNumber == 50, yardlineNumber, 100 - yardlineNumber)
  ) |>
  select(is_run, quarter, down, yardsToGo, clock_seconds, score_diff, yardline, offense_run_prop, defense_run_prop,
         team_situational_run_prop)

# --- Build Model ---
run_model <- glm(
  is_run ~ quarter + down + yardsToGo + clock_seconds + score_diff + yardline + offense_run_prop + defense_run_prop
  + team_situational_run_prop,
  data = lmodel_data,
  family = binomial()
)

tidy(run_model)
tidy(run_model, conf.int = TRUE, exponentiate = TRUE)

# --- Add Predictions ---
lmodel_data <- lmodel_data |>
  mutate(
    pred_run_prob = predict(run_model, type = "response"),
    pred_run_class = ifelse(pred_run_prob > 0.5, "Run", "Pass"),
    pred_run_binary = ifelse(pred_run_prob > 0.5, 1, 0)
  )

# --- Evaluation Metrics ---
accuracy <- mean(lmodel_data$pred_run_binary == lmodel_data$is_run)
misclass_rate <- mean(lmodel_data$pred_run_binary != lmodel_data$is_run)
brier_score <- mean((lmodel_data$pred_run_binary - lmodel_data$pred_run_prob)^2)

accuracy
misclass_rate
brier_score

table(
  Predicted = factor(ifelse(lmodel_data$pred_run_binary == 1, "Run", "Pass"), levels = c("Run", "Pass")),
  Actual = factor(ifelse(lmodel_data$is_run == 1, "Run", "Pass"), levels = c("Run", "Pass"))
)

# --- ROC Curve ---
lmodel_data <- lmodel_data |>
  mutate(pred_prob = predict(run_model, type = "response"))

run_roc <- roc(lmodel_data$is_run, lmodel_data$pred_prob)
run_roc$auc

roc_df <- tibble(
  threshold = run_roc$thresholds,
  specificity = run_roc$specificities,
  sensitivity = run_roc$sensitivities
)

ggplot(roc_df, aes(x = 1 - specificity, y = sensitivity)) +
  geom_path(color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = paste0("ROC Curve (AUC = ", round(run_roc$auc, 3), ")"),
    x = "False Positive Rate (1 - Specificity)",
    y = "True Positive Rate (Sensitivity)"
  ) +
  theme_minimal()

# --- Calibration Plot ---
lmodel_data |>
  mutate(
    pred = predict(run_model, type = "response"),
    obs = is_run
  ) |>
  ggplot(aes(x = pred, y = obs)) +
  geom_point(alpha = 0.3) +
  geom_smooth(method = "loess", se = FALSE, color = "blue") +
  geom_abline(slope = 1, intercept = 0, linetype = "dashed", color = "red") +
  labs(
    title = "Calibration Plot: Observed vs Predicted Run Probability",
    x = "Predicted Probability",
    y = "Observed Outcome (Run = 1, Pass = 0)"
  ) +
  theme_minimal()

# --- Shiny App ---
library(shiny)
library(lubridate)

# UI
ui <- fluidPage(
  titlePanel("Run Play Probability Predictor"),
  
  sidebarLayout(
    sidebarPanel(
      numericInput("quarter", "Quarter:", value = 1, min = 1, max = 5),
      numericInput("down", "Down:", value = 1, min = 1, max = 4),
      numericInput("yardsToGo", "Yards to Go:", value = 10, min = 1, max = 30),
      textInput("clock", "Game Clock (MM:SS):", value = "15:00"),
      numericInput("score_diff", "Score Differential (possession team):", value = 0, min = -100, max = 100),
      numericInput("yardline", "Field Position (0 = own goal line, 100 = opponent’s):", value = 50, min = 1, max = 99),
      selectInput("off_team", "Offensive Team:", choices = sort(unique(model_data$possessionTeam)), selected = "KC"),
      selectInput("def_team", "Defensive Team:", choices = sort(unique(model_data$defensiveTeam)), selected = "SF")
    ),
    
    mainPanel(
      h3("Predicted Probability of a Run:"),
      verbatimTextOutput("run_prob"),
      h3("Field View:"),
      plotOutput("field_plot", height = "500px", width = "100%")
    )
  )
)

# Server
server <- function(input, output) {
  output$run_prob <- renderPrint({
    clock_seconds <- as.numeric(ms(input$clock))
    
    off_prop <- offense_run_props |>
      filter(possessionTeam == input$off_team) |>
      pull(team_run_prop)
    
    def_prop <- defense_run_props |>
      filter(defensiveTeam == input$def_team) |>
      pull(defense_run_prop)
    
    if (length(off_prop) == 0) off_prop <- mean(offense_run_props$team_run_prop, na.rm = TRUE)
    if (length(def_prop) == 0) def_prop <- mean(defense_run_props$defense_run_prop, na.rm = TRUE)
    
    ytg_bin_input <- case_when(
      input$yardsToGo <= 1 ~ "Short (0-1)",
      input$yardsToGo <= 3 ~ "Short (2-3)",
      input$yardsToGo <= 6 ~ "Medium (4-6)",
      input$yardsToGo <= 10 ~ "Medium (7-10)",
      input$yardsToGo <= 20 ~ "Long (11-20)",
      TRUE ~ "Very Long (21+)"
    )
    
    situational_prop <- team_situational_run_props |>
      filter(
        possessionTeam == input$off_team,
        quarter == input$quarter,
        down == input$down,
        ytg_bin == ytg_bin_input
      ) |>
      pull(run_prop)
    
    if (length(situational_prop) == 0) {
      situational_prop <- mean(team_situational_run_props$run_prop, na.rm = TRUE)
    }
    
    new_data <- data.frame(
      quarter = factor(input$quarter, levels = levels(lmodel_data$quarter)),
      down = input$down,
      yardsToGo = input$yardsToGo,
      clock_seconds = clock_seconds,
      score_diff = input$score_diff,
      yardline = input$yardline,
      offense_run_prop = off_prop,
      defense_run_prop = def_prop,
      team_situational_run_prop = situational_prop
    )
    
    prob <- predict(run_model, newdata = new_data, type = "response")
    cat(sprintf("%.1f%%", prob * 100))
  })
  output$field_plot <- renderPlot({
    los <- input$yardline + 10
    first_down <- los + input$yardsToGo
    
    plot <- field_background +
      geom_vline(xintercept = los, color = "blue", linewidth = 1) +
      labs(title = "Line of Scrimmage (Blue) and First Down Line (Yellow)") +
      theme_void()
    
    if (first_down < 110) {
      plot <- plot + geom_vline(xintercept = first_down, color = "yellow", linewidth = 1)
    }
    
    plot
  })
}

# Run the app
shinyApp(ui = ui, server = server)
