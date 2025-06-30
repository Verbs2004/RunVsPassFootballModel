# ───────────────────────────────────────────────────────────────
# 0) Libraries
# ───────────────────────────────────────────────────────────────
library(nflreadr)
library(dplyr);      library(tidyr);      library(stringr)
library(ranger);     library(pROC);       library(readr);     library(purrr)
library(ggplot2);    library(slider)
library(fastDummies); library(lubridate)

# ───────────────────────────────────────────────────────────────
# 1) Load core tables  (2016-23)  ← *game_id coerced to character*
# ───────────────────────────────────────────────────────────────
pbp <- load_pbp(2016:2023) %>% 
  mutate(game_id = as.character(game_id))            # ★ changed

sched <- load_schedules(2016:2023) %>% 
  select(game_id,
         sched_roof       = roof,
         sched_temp       = temp,
         sched_wind_speed = wind) %>% 
  mutate(game_id = as.character(game_id))            # ★ changed

parts <- load_participation(2016:2023)               # personnel through 2022
plays <- read_csv("plays.csv", show_col_types = FALSE)

if (!"absolute_yardline_number" %in% names(pbp)) {
  pbp <- pbp %>% mutate(
    absolute_yardline_number =
      if_else(side_of_field == posteam, 110 - yardline_100, 110 + yardline_100))
}

# ───────────────────────────────────────────────────────────────
# 2) Parse participation  (rename nflverse_game_id → game_id)
# ───────────────────────────────────────────────────────────────
parse_pers <- function(x, pos){
  str_extract(x, paste0("(?i)(\\d+)\\s*", pos)) %>%     # ignore-case, flexible space
    str_extract("\\d+") %>% 
    as.integer() %>% 
    replace_na(0L)
}

parts_parsed <- parts %>% 
  rename(game_id = nflverse_game_id) %>%               # ★ changed
  mutate(game_id = as.character(game_id)) %>% 
  select(game_id, play_id,
         offense_personnel, defense_personnel, defenders_in_box) %>% 
  mutate(
    # ---- offense ----
    RB_off_P = parse_pers(offense_personnel, "RB"),
    TE_off_P = parse_pers(offense_personnel, "TE"),
    WR_off_P = parse_pers(offense_personnel, "WR"),
    # ---- defense ----
    DL_def_P = parse_pers(defense_personnel, "DL"),
    LB_def_P = parse_pers(defense_personnel, "LB"),
    DB_def_P = parse_pers(defense_personnel, "DB"),
    # handy derived flags
    skill_diff_P = WR_off_P - DB_def_P,
    heavy_P      = RB_off_P + TE_off_P >= 3,
    trips_P      = WR_off_P >= 3
  )

# ───────────────────────────────────────────────────────────────
# 3) plays.csv (extra pre-snap)  ← game_id coerced to character
# ───────────────────────────────────────────────────────────────
plays_j <- plays %>% 
  rename(game_id = gameId, play_id = playId) %>% 
  mutate(game_id = as.character(game_id),
         play_id = as.numeric(play_id)) %>% 
  select(game_id, play_id,
         offenseFormation, receiverAlignment,
         playClockAtSnap, playAction)

# ───────────────────────────────────────────────────────────────
# 4) Feature engineering  (+ Table 2-1 features) – unchanged
# ───────────────────────────────────────────────────────────────
df <- pbp %>% 
  left_join(sched,  by = "game_id") %>% 
  left_join(plays_j,      by = c("game_id","play_id")) %>% 
  left_join(parts_parsed, by = c("game_id","play_id")) %>% 
  filter(play_type %in% c("run","pass")) %>% 
  arrange(game_id, order_sequence) %>% 
  #  … everything here is identical to your script …
  mutate(
    # (unchanged giant mutate block)
    # …
  ) %>% 
  mutate(across(c(RB_off_P:skill_diff_P, defenders_in_box,
                  run_rate_last5, box_plus_1,
                  game_rtp, season_rtp),
                ~replace_na(.,0)),
         across(c(heavy_P,trips_P,heavy_set,four_WR_set,empty_back,
                  last_call_run,last_call_pass),
                ~replace_na(.,FALSE))) %>% 
  select(
    game_id, season, posteam, defteam, is_pass,
    drive_num, quarter_curr, time_under_qtr,
    yard_line, yards_to_go, score_diff,
    no_score_prob, fg_prob, safety_prob, td_prob,
    win_prob, game_rtp, season_rtp, month,
    form_cat, pers_code,
    RB_off_P, TE_off_P, WR_off_P,
    DL_def_P, LB_def_P, DB_def_P,
    skill_diff_P, heavy_set, four_WR_set, empty_back,
    defenders_in_box, box_plus_1,
    last_call_run, last_call_pass, run_rate_last5,
    big_lead, two_minute_drill, down, ydstogo,
    shotgun, no_huddle, sgn_d2, sgn_d3,
    wp_tier, late_desperation, nepd_tier,
    yards_to_goal_bin, hash_side,
    first_play_of_drive, prev_run, prev_pass,
    roof_code, is_turf, wind_cat, rain_snow, indoor,
    trailing_big,
    posteam, defteam
  )

# ───────────────────────────────────────────────────────────────
# 4 B) One-hot teams  – unchanged
# ───────────────────────────────────────────────────────────────
df <- df %>% 
  mutate(posteam = factor(posteam),
         defteam = factor(defteam)) %>% 
  fastDummies::dummy_cols(
    select_columns          = c("posteam","defteam"),
    remove_first_dummy      = FALSE,
    remove_selected_columns = TRUE,
    ignore_na               = TRUE) %>% 
  rename_with(~str_replace(.x,"^posteam_","OffTeam_"),  starts_with("posteam_")) %>% 
  rename_with(~str_replace(.x,"^defteam_","DefTeam_"),  starts_with("defteam_"))

# ───────────────────────────────────────────────────────────────
# 5) Train / Test split  – unchanged
# ───────────────────────────────────────────────────────────────
train      <- df %>% filter(season < 2023)
test_raw   <- df %>% filter(season == 2023)
test_labels <- test_raw$is_pass
test        <- test_raw %>% select(-game_id,-season)

for(fc in names(train)[sapply(train,is.factor)]) {
  test[[fc]] <- factor(test[[fc]], levels = levels(train[[fc]]))
}

# ───────────────────────────────────────────────────────────────
# 6) ranger fit  (importance stored)
# ───────────────────────────────────────────────────────────────
set.seed(42)
rf_mod <- ranger(
  dependent.variable.name = "is_pass",
  data        = train %>% select(-game_id),
  probability = TRUE,
  num.trees   = 700,
  mtry        = floor(sqrt(ncol(train) - 1)),
  min.node.size = 5,
  importance  = "impurity",      # ★ changed (store importance)
  seed        = 42
)

# align columns
train_cols      <- rf_mod$forest$independent.variable.names
missing_in_test <- setdiff(train_cols, names(test))
for(col in missing_in_test) test[[col]] <- 0
test <- test[, train_cols]

# ───────────────────────────────────────────────────────────────
# 7) Predict & evaluate  – unchanged
# ───────────────────────────────────────────────────────────────
pred_prob  <- predict(rf_mod, data = test)$predictions[,"pass"]
pred_class <- ifelse(pred_prob > 0.5, "pass", "run")

auc_val <- auc(roc(as.numeric(test_labels == "pass"), pred_prob))
acc_val <- mean(pred_class == test_labels)

cat(sprintf("\nRandom-Forest (19-22 ➜ 23)\nAUC = %.3f | ACC = %.3f\n",
            auc_val, acc_val))

# ───────────────────────────────────────────────────────────────
# 8)  Feature predictiveness / importance
# ───────────────────────────────────────────────────────────────
library(tibble)
library(dplyr)
library(ggplot2)

## 8-A  Gini-impurity importance (built-in) ----------------------
imp_gini <- rf_mod$variable.importance          # stored because we set importance="impurity"

imp_gini_df <- tibble(
  Feature    = names(imp_gini),
  Importance = imp_gini
) %>%
  arrange(desc(Importance))

print(imp_gini_df, n = 25)   # top-25 Gini

## 8-B  Permutation importance (Altmann) -------------------------
# build a single data frame that contains both predictors & label
perm_df <- test %>% mutate(is_pass = test_labels)

set.seed(42)
imp_perm_raw <- ranger::importance_pvalues(
  rf_mod,
  data    = perm_df,
  formula = is_pass ~ .,      # outcome ~ all predictors
  method  = "altmann",
  num.permutations = 500      # adjust up/down for speed vs. stability
)

imp_perm_df <- imp_perm_raw %>%               # data.frame with row-names
  as_tibble(rownames = "Feature") %>%         # row-names → column
  rename(
    Importance = importance,
    P_value    = pvalue
  ) %>%
  arrange(desc(Importance))

print(imp_perm_df, n = 25)   # top-25 permutation

## 8-C  Quick bar plot of top-20 permutation importances --------
imp_perm_df %>%
  slice_max(Importance, n = 20) %>%
  ggplot(aes(x = reorder(Feature, Importance), y = Importance)) +
  geom_col() +
  coord_flip() +
  labs(title = "Top 20 Features (Permutation Importance)",
       x = NULL, y = "Decrease-Accuracy Importance")
