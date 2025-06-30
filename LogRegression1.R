library(nflreadr)
library(dplyr)
library(tidyr)
library(caret)
library(pROC)
library(ggplot2)
library(purrr)
library(broom)
library(stringr)

# 1) Load PBP data
cat("1) Loading PBP data (2020–2024)...\n")
pbp <- load_pbp(2020:2024)
cat("   → PBP dimensions:", dim(pbp), "\n\n")

# 2) Build modeling dataframe with derived features and encode roof & wind_dir numerically
cat("2) Building modeling dataframe with numeric roof & wind_dir...\n")
df <- pbp %>%
  filter(play_type %in% c("run","pass")) %>%
  arrange(game_id, drive, order_sequence) %>%
  group_by(game_id) %>%
  mutate(
    turnover_last_play       = lag(if_else(fumble_lost==1 | interception==1,1,0), default=0),
    turnover_count_in_drive  = lag(cumsum(if_else(fumble_lost==1 | interception==1,1,0)), default=0),
    # parse wind direction text from weather
    wind_dir_raw = str_extract(weather, "(?<=Wind: )\\b[A-Za-z]+\\b"),
    # encode roof
    roof_code = case_when(
      roof == "dome"     ~ 1,
      roof == "closed"   ~ 2,
      roof == "open"     ~ 3,
      roof == "outdoors" ~ 4,
      TRUE               ~ NA_real_
    ),
    # encode wind direction to 1–16 compass points
    wind_dir_code = case_when(
      wind_dir_raw == "N"   ~ 1,
      wind_dir_raw == "NNE" ~ 2,
      wind_dir_raw == "NE"  ~ 3,
      wind_dir_raw == "ENE" ~ 4,
      wind_dir_raw == "E"   ~ 5,
      wind_dir_raw == "ESE" ~ 6,
      wind_dir_raw == "SE"  ~ 7,
      wind_dir_raw == "SSE" ~ 8,
      wind_dir_raw == "S"   ~ 9,
      wind_dir_raw == "SSW" ~ 10,
      wind_dir_raw == "SW"  ~ 11,
      wind_dir_raw == "WSW" ~ 12,
      wind_dir_raw == "W"   ~ 13,
      wind_dir_raw == "WNW" ~ 14,
      wind_dir_raw == "NW"  ~ 15,
      wind_dir_raw == "NNW" ~ 16,
      TRUE                  ~ NA_real_
    ),
    # two-minute drill
    two_minute_drill = as.integer(qtr %in% 1:4 & quarter_seconds_remaining <= 120 & down <= 2),
    is_pass          = as.integer(play_type == "pass")
  ) %>%
  ungroup() %>%
  transmute(
    season,
    is_pass,
    down,
    ydstogo,
    yardline_100,
    qtr,
    score_differential,
    quarter_seconds_remaining,
    posteam_timeouts_remaining,
    defteam_timeouts_remaining,
    two_minute_drill,
    turnover_last_play,
    turnover_count_in_drive,
    roof_code,
    temp,
    wind_speed = wind,
    wind_dir_code
  ) %>%
  filter(complete.cases(.))

cat("   → Modeling df rows:", nrow(df),
    "; seasons:", paste(sort(unique(df$season)), collapse=", "), "\n\n")

# 3) Split train/test
cat("3) Splitting train/test...\n")
train <- df %>% filter(season < 2024) %>% select(-season)
test  <- df %>% filter(season == 2024) %>% select(-season)
cat("   → Train rows:", nrow(train), " (run/pass =", paste(table(train$is_pass), collapse="/"), ")\n")
cat("   → Test  rows:", nrow(test),  " (run/pass =", paste(table(test$is_pass), collapse="/"), ")\n\n")

# 4) Feature list
features <- setdiff(names(train), "is_pass")
cat("4) Features:", paste(features, collapse=", "), "\n\n")

# 5) Fit multivariate logistic regression
cat("5) Fitting multivariate logistic regression...\n")
multi_mod <- glm(is_pass ~ ., family=binomial, data=train)
cat("   → Model fit complete\n\n")

# 6) Evaluate AUC
cat("6) Evaluating AUC...\n")
train_pred <- predict(multi_mod, newdata=train, type="response")
test_pred  <- predict(multi_mod, newdata=test,  type="response")
cat(sprintf("   → Train AUC: %.4f\n", as.numeric(auc(train$is_pass, train_pred))))
cat(sprintf("   →  Test AUC: %.4f\n\n", as.numeric(auc(test$is_pass, test_pred))))

# 7) Variable importance (|t-stat|)
vi <- broom::tidy(multi_mod) %>%
  filter(term != "(Intercept)") %>%
  mutate(Importance = abs(statistic)) %>%
  arrange(desc(Importance))
cat("7) Top 10 predictors by |t-stat|:\n")
print(head(vi,10))

# plot top 10
top10 <- vi %>% slice_head(n=10) %>% arrange(Importance)
ggplot(top10, aes(x=reorder(term,Importance), y=Importance)) +
  geom_col(fill="steelblue") +
  coord_flip() +
  labs(title="Top 10 Predictors (|t-stat|)", x="", y="|t-stat|") +
  theme_minimal()

# 8) Univariate AUCs
cat("\n8) Computing univariate AUCs...\n")
auc_df <- map_df(features, function(feat) {
  cat("   →", feat, "...\n")
  m <- glm(reformulate(feat,"is_pass"), family=binomial, data=train)
  p <- predict(m, newdata=test, type="response")
  tibble(feature=feat, AUC=as.numeric(auc(test$is_pass,p)))
})
cat("   → Univariate AUCs:\n")
print(arrange(auc_df, desc(AUC)))

ggplot(auc_df, aes(x=reorder(feature,AUC), y=AUC)) +
  geom_col(fill="steelblue") +
  coord_flip() +
  labs(title="Univariate AUCs", x="", y="AUC") +
  theme_minimal()
