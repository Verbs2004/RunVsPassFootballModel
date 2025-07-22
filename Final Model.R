# ═══════════════════════════════════════════════════════════════════════════════
# ADVANCED NFL PASS/RUN PREDICTION ENSEMBLE
# Stacking Model with Full Features & Strict Cross-Validation
# ═══════════════════════════════════════════════════════════════════════════════

# ───────────────────────────────────────────────────────────────────────────────
# 0) SETUP AND LIBRARIES
# ───────────────────────────────────────────────────────────────────────────────
cat("=== [SETUP] INITIALIZING ENVIRONMENT ===\n")
suppressPackageStartupMessages({
  library(nflreadr)
  library(data.table)
  library(xgboost)
  library(fastDummies)
  library(pROC)
  library(knitr)
  library(doParallel)
  library(stringr)
  library(purrr)
  library(foreach)
  library(arrow)
  library(moments)
  library(glmnet)
  library(caret)
  library(MLmetrics)
  library(caTools)
  library(ggplot2)
  library(ROCR)
  library(caTools) # This line makes the `trapz` function available.
  library(scales)
  library(cowplot)
})

# Performance setup
n_cores <- parallel::detectCores()
setDTthreads(n_cores)
cat("=== [SETUP] Using", n_cores, "cores for processing. ===\n")

# ───────────────────────────────────────────────────────────────────────────────
# 1) ROBUST DATA LOADING WITH CACHING
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [LOAD] LOADING DATA ===\n")

# Historical PBP (1999-2024)
load_historical_pbp <- function() {
  cache_file <- "pbp_historical_cache.rds"
  if (file.exists(cache_file)) {
    cat("  [LOAD_PBP] Loading historical PBP from cache...\n")
    return(readRDS(cache_file))
  }
  
  cat("  [LOAD_PBP] Loading historical PBP (1999-2024) from source...\n")
  pbp_list <- vector("list", length(1999:2024))
  for (i in seq_along(1999:2024)) {
    year <- 1999 + i - 1
    cat("    [LOAD_PBP] Loading year", year, "...\n")
    tryCatch({
      pbp_list[[i]] <- nflreadr::load_pbp(year)
    }, error = function(e) {
      cat("    [LOAD_PBP_ERROR] Failed to load", year, ":", e$message, "\n")
      pbp_list[[i]] <- NULL
    })
  }
  
  pbp_hist <- rbindlist(pbp_list[!sapply(pbp_list, is.null)])
  saveRDS(pbp_hist, cache_file)
  cat("  [LOAD_PBP] Caching historical PBP data for future use.\n")
  return(pbp_hist)
}

# Load all required data
pbp_hist <- setDT(load_historical_pbp())
cat("  [LOAD] Historical PBP loaded:", nrow(pbp_hist), "rows\n")

# Modern PBP for Model 2
pbp_modern <- pbp_hist[season >= 2016 & season <= 2023]
cat("  [LOAD] Modern PBP filtered (2016-2023):", nrow(pbp_modern), "rows\n")

# Participation data for Models 2 & 3
load_participation_safe <- function() {
  tryCatch({
    cat("  [LOAD_PART] Loading participation data (2016-2023)...\n")
    parts <- setDT(nflreadr::load_participation(2016:2023))
    return(parts[, .(old_game_id, play_id, offense_personnel, defense_personnel, defenders_in_box)])
  }, error = function(e) {
    cat("  [LOAD_PART_ERROR] Could not load participation data:", e$message, "\n")
    return(data.table())
  })
}
participation_data <- load_participation_safe()
cat("  [LOAD] Participation data loaded:", nrow(participation_data), "rows\n")


# BDB 2022 data
load_bdb_data <- function() {
  required_files <- c("games.csv", "plays.csv", "players.csv")
  
  if (!all(file.exists(required_files))) {
    cat("  [LOAD_BDB_WARN] Not all BDB files found. Creating dummy data.\n")
    return(list(
      games = data.table(gameId = integer(), week = integer()),
      plays = data.table(gameId = integer(), playId = integer(), down = integer(), yardsToGo = integer(), isDropback = logical()),
      players = data.table(nflId = integer(), position = character()),
      tracking = data.table()
    ))
  }
  
  cat("  [LOAD_BDB] Loading BDB data files...\n")
  bdb_data <- list()
  
  # Load games and ensure week column is properly typed
  bdb_data$games <- fread("games.csv")
  cat("    [LOAD_BDB] games.csv loaded.\n")
  
  # Ensure week column exists and is properly typed
  if ("week" %in% names(bdb_data$games)) {
    bdb_data$games[, week := as.integer(week)]
    cat("    [LOAD_BDB] Week column found and converted to integer.\n")
  } else {
    bdb_data$games[, week := 1L]
    cat("    [LOAD_BDB_WARN] No week column found in games.csv, using default week 1.\n")
  }
  
  # Ensure gameId is properly typed
  bdb_data$games[, gameId := as.integer(gameId)]
  
  # Load other files
  bdb_data$plays <- fread("plays.csv")
  cat("    [LOAD_BDB] plays.csv loaded.\n")
  bdb_data$players <- fread("players.csv")
  cat("    [LOAD_BDB] players.csv loaded.\n")
  
  # Load tracking files
  tracking_files <- list.files(pattern = "tracking_week_\\d+\\.csv", full.names = TRUE)
  if (length(tracking_files) > 0) {
    cat("    [LOAD_BDB] Loading and combining", length(tracking_files), "tracking files...\n")
    bdb_data$tracking <- rbindlist(lapply(tracking_files, fread))
  } else {
    cat("    [LOAD_BDB_WARN] No tracking files found, creating empty tracking data.\n")
    bdb_data$tracking <- data.table()
  }
  
  cat("  [LOAD_BDB] BDB data loading complete.\n")
  cat("  [LOAD_BDB] Games data preview - gameId range:", 
      range(bdb_data$games$gameId, na.rm = TRUE), 
      "| Week range:", range(bdb_data$games$week, na.rm = TRUE), "\n")
  
  return(bdb_data)
}
bdb_data <- load_bdb_data()

# ───────────────────────────────────────────────────────────────────────────────
# 2) FEATURE ENGINEERING FUNCTIONS (USER'S ORIGINAL, UNMODIFIED)
# ───────────────────────────────────────────────────────────────────────────────

# ───────────────────────────────────────────────────────────────────────────────
# 2) FEATURE ENGINEERING FUNCTIONS (USER'S ORIGINAL, UNMODIFIED)
# ───────────────────────────────────────────────────────────────────────────────

create_model1_features <- function(pbp_data) {
  cat("--- [FEAT_M1] Creating Model 1 features ---\n")
  
  df <- pbp_data[!is.na(qb_dropback) & !is.na(down) & down %in% 1:4 & !is.na(ydstogo) & ydstogo >= 1 & ydstogo <= 50 & !is.na(yardline_100) & yardline_100 >= 1 & yardline_100 <= 99 & !is.na(old_game_id) & !is.na(play_id)]
  df[, season := as.integer(substr(old_game_id, 1, 4))]
  cat("  [FEAT_M1] After initial filtering:", nrow(df), "plays retained.\n")
  
  # --- THE FIX: Only create the target if it's not already there ---
  if (!"is_dropback" %in% names(df)) {
    df[, is_dropback := as.numeric(qb_dropback == 1)]
  }
  
  df[, `:=`(shotgun = fcoalesce(as.numeric(shotgun), 0), no_huddle = fcoalesce(as.numeric(no_huddle), 0), wp = fcoalesce(wp, 0.5), score_differential = fcoalesce(score_differential, 0), qtr = fcoalesce(qtr, 1), goal_to_go = fcoalesce(as.numeric(goal_to_go), 0), quarter_seconds_remaining = fcoalesce(quarter_seconds_remaining, 900), epa = fcoalesce(epa, 0))]
  cat("  [FEAT_M1] Base situational features created.\n")
  
  df[, `:=`(third_down = as.numeric(down == 3), fourth_down = as.numeric(down == 4), short_yardage = as.numeric(ydstogo <= 3), long_yardage = as.numeric(ydstogo >= 8), red_zone = as.numeric(yardline_100 <= 20), two_minute_warning = as.numeric((qtr == 2 | qtr == 4) & quarter_seconds_remaining <= 120), trailing = as.numeric(score_differential < 0), leading = as.numeric(score_differential > 0), close_game = as.numeric(abs(score_differential) <= 7), score_diff_x_time = score_differential * quarter_seconds_remaining, wp_leverage = abs(wp - 0.5), time_pressure = as.numeric(quarter_seconds_remaining <= 300 & abs(score_differential) <= 10), garbage_time = as.numeric(abs(score_differential) > 21))]
  cat("  [FEAT_M1] Advanced situational features created.\n")
  
  df[, `:=`(down_x_distance = down * ydstogo, third_and_long = as.numeric(down == 3 & ydstogo >= 7), shotgun_x_down = shotgun * down, wp_x_score_diff = wp * score_differential, red_zone_x_down = red_zone * down, short_yardage_x_down = short_yardage * down, time_x_score = quarter_seconds_remaining * abs(score_differential), leverage_x_distance = wp_leverage * ydstogo)]
  cat("  [FEAT_M1] Interaction features created.\n")
  
  feature_cols <- c("down", "ydstogo", "yardline_100", "qtr", "shotgun", "no_huddle", "wp", "score_differential", "goal_to_go", "third_down", "fourth_down", "short_yardage", "long_yardage", "red_zone", "two_minute_warning", "trailing", "leading", "close_game", "down_x_distance", "third_and_long", "shotgun_x_down", "wp_x_score_diff", "score_diff_x_time", "wp_leverage", "time_pressure", "garbage_time", "red_zone_x_down", "short_yardage_x_down", "time_x_score", "leverage_x_distance")
  
  id_cols_to_keep <- c("old_game_id", "play_id", "is_dropback", "season")
  if ("gameId" %in% names(df)) { id_cols_to_keep <- c(id_cols_to_keep, "gameId") }
  
  final_df <- df[, c(id_cols_to_keep, feature_cols), with = FALSE]
  
  for (col in feature_cols) { if (col %in% names(final_df)) set(final_df, which(is.na(final_df[[col]])), col, 0) }
  cat("  [FEAT_M1] Final feature set cleaned and prepared.\n")
  return(final_df)
}

create_model2_features <- function(pbp_data, participation_data) {
  cat("--- [FEAT_M2] Creating Model 2 features ---\n")
  df <- create_model1_features(pbp_data) # This now correctly returns `is_dropback`
  if (!is.null(participation_data) && nrow(participation_data) > 0) {
    cat("  [FEAT_M2] Merging with participation data...\n")
    df <- merge(df, participation_data, by = c("old_game_id", "play_id"), all.x = TRUE)
    cat("  [FEAT_M2] Parsing personnel strings...\n")
    df[, `:=`(offense_personnel = fcoalesce(offense_personnel, "1 RB, 1 TE, 3 WR"), defense_personnel = fcoalesce(defense_personnel, "4 DL, 3 LB, 4 DB"))]
    df[, `:=`(n_rb = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*RB)")), n_te = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*TE)")), n_wr = as.integer(str_extract(offense_personnel, "\\d+(?=\\s*WR)")), n_dl = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*DL)")), n_lb = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*LB)")), n_db = as.integer(str_extract(defense_personnel, "\\d+(?=\\s*DB)")))]
    df[, c("offense_personnel", "defense_personnel") := NULL]
    personnel_cols <- c("n_rb", "n_te", "n_wr", "n_dl", "n_lb", "n_db")
    for (col in personnel_cols) df[is.na(get(col)), (col) := 0]
    df[, defenders_in_box := fcoalesce(as.integer(defenders_in_box), 7L)]
    cat("  [FEAT_M2] Creating personnel features...\n")
    df[, `:=`(heavy_set = as.numeric((n_rb + n_te) >= 3), empty_backfield = as.numeric(n_rb == 0), trips_formation = as.numeric(n_wr >= 3), nickel_defense = as.numeric(n_db == 5), dime_defense = as.numeric(n_db >= 6), personnel_advantage = n_wr - n_db, box_count_advantage = 5 - defenders_in_box)]
    df[, `:=`(personnel_mismatch = abs(personnel_advantage), heavy_vs_light = as.numeric(heavy_set == 1 & n_db >= 5), speed_mismatch = as.numeric(empty_backfield == 1 & defenders_in_box >= 8))]
  } else {
    cat("  [FEAT_M2_WARN] No participation data found, adding default personnel columns.\n")
    df[, `:=`(defenders_in_box=7, n_rb=0, n_te=0, n_wr=0, n_dl=0, n_lb=0, n_db=0, heavy_set=0, empty_backfield=0, trips_formation=0, nickel_defense=0, dime_defense=0, personnel_advantage=0, box_count_advantage=-2, personnel_mismatch=0, heavy_vs_light=0, speed_mismatch=0)]
  }
  for (col in names(df)) { if (is.numeric(df[[col]])) set(df, which(is.na(df[[col]])), col, 0) }
  cat("  [FEAT_M2] Final feature set cleaned and prepared.\n")
  return(df)
}

# NOTE: NO CHANGES ARE NEEDED FOR THE ADVANCED HELPER FUNCTIONS (geometry, motion, etc.)
# They only calculate features and are independent of the target variable.

# ───────────────────────────────────────────────────────────────────────────────
# 2) ENHANCED FEATURE ENGINEERING WITH ADVANCED TRACKING FEATURES
# ───────────────────────────────────────────────────────────────────────────────

# Helper function to calculate geometric features (RESTORED TO ORIGINAL WORKING VERSION)
calculate_formation_geometry <- function(tracking_data, players_data) {
  # Get offensive players at ball snap
  snap_data <- tracking_data[event == "ball_snap"]
  snap_with_pos <- merge(snap_data, players_data[, .(nflId, position)], by = "nflId", all.x = TRUE)
  
  # Define offensive positions
  offensive_positions <- c("QB", "RB", "FB", "WR", "TE")
  off_players <- snap_with_pos[position %in% offensive_positions]
  
  # Calculate features by game and play
  formation_features <- off_players[, {
    if (.N < 2) {
      list(
        formation_width = 0, formation_depth = 0, formation_area = 0,
        formation_compactness = 0, formation_symmetry = 0, formation_balance = 0,
        wr_spread = 0, wr_depth_variance = 0, te_alignment = 0,
        backfield_depth = 0, pocket_width = 0, formation_density = 0,
        line_compactness = 0, receiver_cluster_count = 0, avg_receiver_separation = 0
      )
    } else {
      # Basic geometric measurements
      x_coords <- x[!is.na(x)]
      y_coords <- y[!is.na(y)]
      
      if (length(x_coords) < 2 || length(y_coords) < 2) {
        return(list(
          formation_width = 0, formation_depth = 0, formation_area = 0,
          formation_compactness = 0, formation_symmetry = 0, formation_balance = 0,
          wr_spread = 0, wr_depth_variance = 0, te_alignment = 0,
          backfield_depth = 0, pocket_width = 0, formation_density = 0,
          line_compactness = 0, receiver_cluster_count = 0, avg_receiver_separation = 0
        ))
      }
      
      # Formation dimensions
      width <- max(y_coords) - min(y_coords)
      depth <- max(x_coords) - min(x_coords)
      area <- width * depth
      
      # Formation compactness (inverse of average pairwise distance)
      coords_matrix <- as.matrix(data.frame(x = x_coords, y = y_coords))
      if (nrow(coords_matrix) > 1) {
        avg_distance <- mean(dist(coords_matrix))
        compactness <- 1 / (1 + avg_distance)
      } else {
        compactness <- 0
      }
      
      # Fixed formation_symmetry calculation
      formation_symmetry <- {
        # Find field center or use ball position as reference
        ball_y <- y[is.na(nflId)][1]  # Ball position
        if (is.na(ball_y)) {
          # Use field center if no ball position
          ball_y <- 26.65  # Half of 53.3 yard field width
        }
        
        # Get offensive players only
        off_players_y <- y[position %in% c("QB", "RB", "FB", "WR", "TE") & !is.na(y)]
        
        if (length(off_players_y) > 0) {
          # Calculate weighted symmetry based on distance from center
          left_players <- off_players_y[off_players_y < ball_y]
          right_players <- off_players_y[off_players_y > ball_y]
          
          # Weight by distance from center
          left_weight <- sum(abs(left_players - ball_y))
          right_weight <- sum(abs(right_players - ball_y))
          
          if (left_weight + right_weight > 0) {
            # Normalized symmetry score (0 = completely asymmetric, 1 = perfectly symmetric)
            1 - abs(left_weight - right_weight) / (left_weight + right_weight)
          } else {
            1  # Perfect symmetry if no lateral spread
          }
        } else {
          0
        }
      }
      
      # Formation balance (weighted by position distance from center)
      center_y <- mean(y_coords, na.rm = TRUE)
      weighted_left <- sum(pmax(0, center_y - y_coords) * (center_y - y_coords), na.rm = TRUE)
      weighted_right <- sum(pmax(0, y_coords - center_y) * (y_coords - center_y), na.rm = TRUE)
      balance <- 1 / (1 + abs(weighted_left - weighted_right))
      
      # Wide receiver specific features
      wr_data <- .SD[position == "WR"]
      wr_spread <- if (nrow(wr_data) > 1) max(wr_data$y, na.rm = TRUE) - min(wr_data$y, na.rm = TRUE) else 0
      wr_depth_var <- if (nrow(wr_data) > 1) var(wr_data$x, na.rm = TRUE) else 0
      
      # Tight end alignment (distance from offensive line)
      te_data <- .SD[position == "TE"]
      ol_x <- mean(x[position %in% c("C", "G", "T")], na.rm = TRUE)
      # Fixed te_alignment calculation
      te_alignment <- if (nrow(te_data) > 0) {
        # Get offensive line positions (C, G, T are not in tracking data - use approximate LOS)
        qb_x <- x[position == "QB"][1]
        if (!is.na(qb_x) && length(qb_x) > 0) {
          # Approximate LOS as QB position + 1 yard (since QB is typically behind center)
          approx_los <- qb_x + 1
          mean(abs(te_data$x - approx_los), na.rm = TRUE)
        } else {
          0
        }
      } else {
        0
      }
      # Backfield depth
      qb_x <- x[position == "QB" & !is.na(x)]
      rb_x <- x[position %in% c("RB", "FB") & !is.na(x)]
      if (length(qb_x) > 0 && length(rb_x) > 0) {
        backfield_depth <- mean(abs(rb_x - qb_x[1]), na.rm = TRUE)
      } else {
        backfield_depth <- 0
      }
      
      # Pocket width (distance between tackles)
      tackle_y <- y[position == "T" & !is.na(y)]
      pocket_width <- if (length(tackle_y) >= 2) max(tackle_y) - min(tackle_y) else 0
      
      # Formation density (players per unit area)
      formation_density <- if (area > 0) .N / area else 0
      
      # Line compactness (how tight the offensive line is)
      ol_positions <- c("C", "G", "T")
      ol_data <- .SD[position %in% ol_positions]
      line_compactness <- if (nrow(ol_data) > 1) {
        ol_spread <- max(ol_data$y, na.rm = TRUE) - min(ol_data$y, na.rm = TRUE)
        1 / (1 + ol_spread)
      } else {
        0
      }
      
      # Receiver clustering
      receiver_positions <- c("WR", "TE")
      rec_data <- .SD[position %in% receiver_positions]
      if (nrow(rec_data) > 1) {
        rec_coords <- as.matrix(data.frame(x = rec_data$x, y = rec_data$y))
        rec_distances <- as.matrix(dist(rec_coords))
        # Count clusters (receivers within 3 yards of each other)
        cluster_threshold <- 3
        close_pairs <- sum(rec_distances < cluster_threshold & rec_distances > 0) / 2
        receiver_cluster_count <- close_pairs
        avg_receiver_separation <- mean(rec_distances[rec_distances > 0])
      } else {
        receiver_cluster_count <- 0
        avg_receiver_separation <- 0
      }
      
      list(
        formation_width = width,
        formation_depth = depth,
        formation_area = area,
        formation_compactness = compactness,
        formation_symmetry = formation_symmetry,
        formation_balance = balance,
        wr_spread = wr_spread,
        wr_depth_variance = wr_depth_var,
        te_alignment = te_alignment,
        backfield_depth = backfield_depth,
        pocket_width = pocket_width,
        formation_density = formation_density,
        line_compactness = line_compactness,
        receiver_cluster_count = receiver_cluster_count,
        avg_receiver_separation = avg_receiver_separation
      )
    }
  }, by = .(gameId, playId)]
  
  return(formation_features)
}

# Helper function to calculate motion and pre-snap features (RESTORED TO ORIGINAL WORKING VERSION)
calculate_motion_features <- function(tracking_data, players_data) {
  # Get pre-snap events
  pre_snap_events <- c("line_set", "shift", "motion_start", "motion_end")
  motion_data <- tracking_data[event %in% pre_snap_events]
  motion_with_pos <- merge(motion_data, players_data[, .(nflId, position)], by = "nflId", all.x = TRUE)
  
  # Calculate motion features
  motion_features <- motion_with_pos[, {
    if (.N == 0) {
      list(
        has_motion = 0, motion_player_count = 0, motion_distance = 0,
        motion_speed = 0, motion_direction_change = 0, wr_motion = 0,
        te_motion = 0, rb_motion = 0, motion_toward_los = 0,
        motion_lateral = 0, pre_snap_shifts = 0
      )
    } else {
      # Identify players with motion
      motion_players <- unique(nflId[event %in% c("motion_start", "motion_end")])
      has_motion <- as.numeric(length(motion_players) > 0)
      motion_player_count <- length(motion_players)
      
      # Calculate motion distance and speed for each player
      motion_stats <- lapply(motion_players, function(player_id) {
        player_motion <- .SD[nflId == player_id]
        if (nrow(player_motion) > 1) {
          # Calculate distance traveled
          coords <- player_motion[order(frameId), .(x, y)]
          if (nrow(coords) > 1) {
            distances <- sqrt(diff(coords$x)^2 + diff(coords$y)^2)
            total_distance <- sum(distances, na.rm = TRUE)
            
            # Calculate speed
            time_diff <- max(player_motion$frameId) - min(player_motion$frameId)
            speed <- if (time_diff > 0) total_distance / time_diff else 0
            
            # Direction change
            if (length(distances) > 1) {
              angles <- atan2(diff(coords$y), diff(coords$x))
              direction_changes <- sum(abs(diff(angles)) > pi/4, na.rm = TRUE)
            } else {
              direction_changes <- 0
            }
            
            list(distance = total_distance, speed = speed, direction_changes = direction_changes)
          } else {
            list(distance = 0, speed = 0, direction_changes = 0)
          }
        } else {
          list(distance = 0, speed = 0, direction_changes = 0)
        }
      })
      
      # Aggregate motion stats
      motion_distance <- if (length(motion_stats) > 0) mean(sapply(motion_stats, function(x) x$distance)) else 0
      motion_speed <- if (length(motion_stats) > 0) mean(sapply(motion_stats, function(x) x$speed)) else 0
      motion_direction_change <- if (length(motion_stats) > 0) mean(sapply(motion_stats, function(x) x$direction_changes)) else 0
      
      # Position-specific motion
      wr_motion <- as.numeric(any(nflId[event %in% c("motion_start", "motion_end")] %in% 
                                    nflId[position == "WR"]))
      te_motion <- as.numeric(any(nflId[event %in% c("motion_start", "motion_end")] %in% 
                                    nflId[position == "TE"]))
      rb_motion <- as.numeric(any(nflId[event %in% c("motion_start", "motion_end")] %in% 
                                    nflId[position %in% c("RB", "FB")]))
      
      # Motion direction relative to line of scrimmage
      if (has_motion > 0) {
        # Simplified: assume motion toward smaller x is toward LOS
        motion_toward_los <- as.numeric(any(diff(x[order(frameId)]) < 0, na.rm = TRUE))
        motion_lateral <- as.numeric(any(abs(diff(y[order(frameId)])) > abs(diff(x[order(frameId)])), na.rm = TRUE))
      } else {
        motion_toward_los <- 0
        motion_lateral <- 0
      }
      
      # Pre-snap shifts
      pre_snap_shifts <- length(unique(event[event %in% c("shift", "motion_start", "motion_end")]))
      
      list(
        has_motion = has_motion,
        motion_player_count = motion_player_count,
        motion_distance = motion_distance,
        motion_speed = motion_speed,
        motion_direction_change = motion_direction_change,
        wr_motion = wr_motion,
        te_motion = te_motion,
        rb_motion = rb_motion,
        motion_toward_los = motion_toward_los,
        motion_lateral = motion_lateral,
        pre_snap_shifts = pre_snap_shifts
      )
    }
  }, by = .(gameId, playId)]
  
  return(motion_features)
}

# Helper function to calculate defensive alignment features (RESTORED TO ORIGINAL WORKING VERSION)
calculate_defensive_features <- function(tracking_data, players_data) {
  snap_data <- tracking_data[event == "ball_snap"]
  snap_with_pos <- merge(snap_data, players_data[, .(nflId, position)], by = "nflId", all.x = TRUE)
  
  # Define defensive positions
  defensive_positions <- c("DE", "DT", "NT", "OLB", "MLB", "ILB", "CB", "S", "FS", "SS")
  def_players <- snap_with_pos[position %in% defensive_positions]
  
  defensive_features <- def_players[, {
    if (.N < 2) {
      list(
        def_front_width = 0, def_depth = 0, def_box_count = 0,
        def_coverage_depth = 0, def_pressure_alignment = 0,
        def_leverage = 0, def_gap_integrity = 0, def_underneath_coverage = 0
      )
    } else {
      # Get line of scrimmage (approximate from ball position)
      ball_x <- x[is.na(nflId)][1]  # Ball position
      if (is.na(ball_x)) ball_x <- mean(x, na.rm = TRUE)
      
      # Defensive front width
      front_positions <- c("DE", "DT", "NT", "OLB")
      front_players <- .SD[position %in% front_positions]
      def_front_width <- if (nrow(front_players) > 1) {
        max(front_players$y, na.rm = TRUE) - min(front_players$y, na.rm = TRUE)
      } else {
        0
      }
      
      # Defensive depth
      def_depth <- max(x, na.rm = TRUE) - min(x, na.rm = TRUE)
      
      # Box count (defenders within 8 yards of LOS)
      def_box_count <- sum(abs(x - ball_x) <= 8, na.rm = TRUE)
      
      # Coverage depth (average depth of secondary)
      secondary_positions <- c("CB", "S", "FS", "SS")
      secondary_players <- .SD[position %in% secondary_positions]
      def_coverage_depth <- if (nrow(secondary_players) > 0) {
        mean(abs(secondary_players$x - ball_x), na.rm = TRUE)
      } else {
        0
      }
      
      # Pressure alignment (how many pass rushers)
      pass_rush_positions <- c("DE", "DT", "NT", "OLB")
      def_pressure_alignment <- sum(position %in% pass_rush_positions, na.rm = TRUE)
      
      # Defensive leverage (average distance from sideline)
      def_leverage <- mean(pmin(abs(y - 0), abs(y - 53.3)), na.rm = TRUE)
      
      # Gap integrity (how well gaps are covered)
      # Simplified: measure spacing between front 7 players
      front_seven_positions <- c("DE", "DT", "NT", "OLB", "MLB", "ILB")
      front_seven <- .SD[position %in% front_seven_positions]
      if (nrow(front_seven) > 1) {
        front_seven_y <- sort(front_seven$y)
        gaps <- diff(front_seven_y)
        def_gap_integrity <- 1 / (1 + var(gaps, na.rm = TRUE))
      } else {
        def_gap_integrity <- 0
      }
      
      # Underneath coverage (defenders 5-15 yards from LOS)
      def_underneath_coverage <- sum(abs(x - ball_x) >= 5 & abs(x - ball_x) <= 15, na.rm = TRUE)
      
      list(
        def_front_width = def_front_width,
        def_depth = def_depth,
        def_box_count = def_box_count,
        def_coverage_depth = def_coverage_depth,
        def_pressure_alignment = def_pressure_alignment,
        def_leverage = def_leverage,
        def_gap_integrity = def_gap_integrity,
        def_underneath_coverage = def_underneath_coverage
      )
    }
  }, by = .(gameId, playId)]
  
  return(defensive_features)
}

# ───────────────────────────────────────────────────────────────────────────────
# ADVANCED MATHEMATICAL FEATURE ENGINEERING (NEW SECTION)
# ───────────────────────────────────────────────────────────────────────────────
suppressPackageStartupMessages({
  library(deldir)
  library(igraph)
  library(sp) # Used for polygon area calculations
})

# Helper function to calculate Voronoi/Delaunay features
# Helper function to calculate Voronoi/Delaunay features (FIXED)
calculate_voronoi_features <- function(snap_data) {
  cat("    [FEAT_MATH] Calculating Voronoi & Delaunay features...\n")
  
  # The ONLY change is here: by = .(gameId, play_id)
  voronoi_features <- snap_data[, {
    offense_players <- .SD[club == possessionTeam & !is.na(nflId)]
    defense_players <- .SD[club != possessionTeam & !is.na(nflId)]
    
    if (nrow(offense_players) < 2 || nrow(defense_players) < 2) {
      list(
        off_voronoi_area_mean = 0, def_voronoi_area_mean = 0,
        pitch_control_ratio = 0.5, def_voronoi_area_variance = 0,
        def_delaunay_edge_mean = 0, def_delaunay_edge_variance = 0
      )
    } else {
      field_boundary <- c(0, 120, 0, 53.3)
      vd <- tryCatch(deldir(x, y, rw = field_boundary, suppressMsge = TRUE), error = function(e) NULL)
      
      if (is.null(vd)) {
        return(list(
          off_voronoi_area_mean = 0, def_voronoi_area_mean = 0,
          pitch_control_ratio = 0.5, def_voronoi_area_variance = 0,
          def_delaunay_edge_mean = 0, def_delaunay_edge_variance = 0
        ))
      }
      
      tile_list <- tile.list(vd)
      player_areas <- map_dbl(seq_along(tile_list), ~tile_list[[.x]]$area)
      all_players <- .SD[!is.na(nflId)]
      all_players$voronoi_area <- player_areas[1:nrow(all_players)]
      
      off_areas <- all_players[club == possessionTeam, voronoi_area]
      def_areas <- all_players[club != possessionTeam, voronoi_area]
      
      off_total_area <- sum(off_areas, na.rm = TRUE)
      def_total_area <- sum(def_areas, na.rm = TRUE)
      
      pitch_control_ratio <- if ((off_total_area + def_total_area) > 0) {
        off_total_area / (off_total_area + def_total_area)
      } else { 0.5 }
      
      def_indices <- which(.SD$club != .SD$possessionTeam)
      def_delaunay <- vd$delsgs[vd$delsgs$ind1 %in% def_indices & vd$delsgs$ind2 %in% def_indices, ]
      edge_lengths <- sqrt((def_delaunay$x1 - def_delaunay$x2)^2 + (def_delaunay$y1 - def_delaunay$y2)^2)
      
      list(
        off_voronoi_area_mean = mean(off_areas, na.rm = TRUE),
        def_voronoi_area_mean = mean(def_areas, na.rm = TRUE),
        pitch_control_ratio = pitch_control_ratio,
        def_voronoi_area_variance = var(def_areas, na.rm = TRUE),
        def_delaunay_edge_mean = mean(edge_lengths, na.rm = TRUE),
        def_delaunay_edge_variance = var(edge_lengths, na.rm = TRUE)
      )
    }
  }, by = .(gameId, play_id)] # <--- THE FIX
  
  return(voronoi_features)
}

# Helper function to calculate graph theory features (FIXED)
calculate_graph_features <- function(snap_data) {
  cat("    [FEAT_MATH] Calculating graph theory features...\n")
  
  # The ONLY change is here: by = .(gameId, play_id)
  graph_features <- snap_data[, {
    defense_players <- .SD[club != possessionTeam & !is.na(nflId)]
    
    if (nrow(defense_players) < 3) {
      list(
        def_graph_density = 0, def_graph_avg_betweenness = 0,
        def_graph_clustering_coef = 0, def_graph_hub_score = 0
      )
    } else {
      dist_matrix <- as.matrix(dist(defense_players[, .(x, y)]))
      adj_matrix <- ifelse(dist_matrix <= 10 & dist_matrix > 0, 1, 0)
      
      g <- graph_from_adjacency_matrix(adj_matrix, mode = "undirected", diag = FALSE)
      
      density <- edge_density(g)
      betweenness_centrality <- mean(betweenness(g, normalized = TRUE), na.rm = TRUE)
      clustering_coef <- transitivity(g, type = "global")
      hub_score <- max(hub_score(g)$vector, na.rm = TRUE)
      
      list(
        def_graph_density = density,
        def_graph_avg_betweenness = betweenness_centrality,
        def_graph_clustering_coef = clustering_coef,
        def_graph_hub_score = hub_score
      )
    }
  }, by = .(gameId, play_id)] # <--- THE FIX
  
  return(graph_features)
}

# Helper function to calculate convex hull features (FIXED)
calculate_convex_hull_features <- function(snap_data) {
  cat("    [FEAT_MATH] Calculating convex hull features...\n")
  
  polygon_area <- function(poly) {
    if (is.null(poly) || nrow(poly@coords) < 3) return(0)
    return(poly@area)
  }
  
  # The ONLY change is here: by = .(gameId, play_id)
  hull_features <- snap_data[, {
    offense_players <- .SD[club == possessionTeam & !is.na(nflId)]
    defense_players <- .SD[club != possessionTeam & !is.na(nflId)]
    
    if (nrow(offense_players) < 3 || nrow(defense_players) < 3) {
      list(
        off_hull_area = 0, def_hull_area = 0,
        hull_area_ratio = 1, def_hull_aspect_ratio = 1
      )
    } else {
      off_hull_indices <- chull(offense_players$x, offense_players$y)
      off_coords <- offense_players[off_hull_indices, .(x, y)]
      off_poly <- Polygon(off_coords, hole = FALSE)
      off_hull_area <- polygon_area(off_poly)
      
      def_hull_indices <- chull(defense_players$x, defense_players$y)
      def_coords <- defense_players[def_hull_indices, .(x, y)]
      def_poly <- Polygon(def_coords, hole = FALSE)
      def_hull_area <- polygon_area(def_poly)
      
      def_hull_x <- def_coords$x
      def_hull_y <- def_coords$y
      width <- max(def_hull_x, na.rm=T) - min(def_hull_x, na.rm=T)
      height <- max(def_hull_y, na.rm=T) - min(def_hull_y, na.rm=T)
      def_aspect_ratio <- if (height > 0) width / height else 1
      
      hull_ratio <- if (def_hull_area > 0) off_hull_area / def_hull_area else 1
      
      list(
        off_hull_area = off_hull_area,
        def_hull_area = def_hull_area,
        hull_area_ratio = hull_ratio,
        def_hull_aspect_ratio = def_aspect_ratio
      )
    }
  }, by = .(gameId, play_id)] # <--- THE FIX
  
  return(hull_features)
}

# Helper function to calculate potential field features (FIXED)
calculate_potential_field_features <- function(snap_data) {
  cat("    [FEAT_MATH] Calculating potential field features...\n")
  
  # The ONLY change is here: by = .(gameId, play_id)
  potential_features <- snap_data[, {
    qb_player <- .SD[position == "QB"]
    defense_players <- .SD[club != possessionTeam & !is.na(nflId)]
    
    if (nrow(qb_player) == 0 || nrow(defense_players) == 0) {
      list(qb_pressure_potential = 0, running_lane_potential = 1)
    } else {
      qb_x <- qb_player$x[1]
      qb_y <- qb_player$y[1]
      
      distances_to_qb <- sqrt((defense_players$x - qb_x)^2 + (defense_players$y - qb_y)^2)
      qb_pressure_potential <- sum(1 / (distances_to_qb^2 + 1e-6), na.rm = TRUE)
      
      ball_player <- .SD[is.na(nflId)]
      if (nrow(ball_player) > 0) {
        los_x <- ball_player$x[1]
        field_center_y <- 53.3 / 2
        play_dir_mult <- if(playDirection[1] == 'left') -1 else 1
        target_point_x <- los_x + (3 * play_dir_mult)
        
        distances_to_target <- sqrt((defense_players$x - target_point_x)^2 + (defense_players$y - field_center_y)^2)
        running_lane_potential <- sum(1 / (distances_to_target^2 + 1e-6), na.rm = TRUE)
      } else {
        running_lane_potential <- 1
      }
      
      list(
        qb_pressure_potential = qb_pressure_potential,
        running_lane_potential = running_lane_potential
      )
    }
  }, by = .(gameId, play_id)] # <--- THE FIX
  
  return(potential_features)
}

# Updated create_model3_features function with enhanced tracking features
# Updated create_model3_features function with enhanced tracking features
create_model3_features <- function(bdb_data, participation_data, pbp_hist) {
  cat("--- [FEAT_M3] Creating Model 3 features with enhanced tracking data ---\n")
  if (is.null(bdb_data) || nrow(bdb_data$plays) == 0) { 
    return(data.table()) 
  }
  
  plays_bdb <- copy(bdb_data$plays)
  games_bdb <- bdb_data$games
  tracking_bdb <- bdb_data$tracking
  
  plays_bdb[, old_game_id := as.character(gameId)]
  
  if (!is.null(games_bdb) && nrow(games_bdb) > 0) {
    plays_bdb <- merge(plays_bdb, games_bdb[, .(gameId, week)], by = "gameId", all.x = TRUE)
  } else { 
    plays_bdb[, week := 1L]
  }
  
  plays_bdb[, is_dropback := ifelse(!is.na(isDropback), as.numeric(isDropback), NA_real_)]
  
  cat("  [FEAT_M3] Merging BDB plays with PBP context...\n")
  pbp_context <- pbp_hist[, .(old_game_id, play_id, play_type, qb_dropback, score_differential, wp, qtr, shotgun, no_huddle, epa, quarter_seconds_remaining, down, ydstogo, yardline_100, goal_to_go)]
  model_df <- merge(plays_bdb, pbp_context, by.x=c("old_game_id", "playId"), by.y=c("old_game_id", "play_id"), all.x=TRUE)
  
  model_df[, is_dropback := fcoalesce(is_dropback, as.numeric(qb_dropback == 1))]
  
  if (!"week" %in% names(model_df)) model_df[, week := 1L]
  else model_df[is.na(week), week := 1L]
  
  model_df[, down := fcoalesce(as.integer(down.x), as.integer(down.y))]
  model_df[, ydstogo := fcoalesce(as.integer(yardsToGo), as.integer(ydstogo))]
  model_df[, c("down.x", "down.y", "yardsToGo") := NULL]
  setnames(model_df, "playId", "play_id", skip_absent=TRUE)
  
  model_df <- model_df[!is.na(is_dropback)]
  
  cat("  [FEAT_M3] Calling create_model2_features to generate base feature set...\n")
  
  # --- THE FIX ---
  # The `is_dropback` column is now created and preserved through the function calls.
  # The complex and buggy logic of saving, deleting, and restoring the column has been removed.
  model_df <- create_model2_features(model_df, participation_data)
  
  cat("  [FEAT_M3] Adding enhanced tracking features...\n")
  if (!is.null(tracking_bdb) && nrow(tracking_bdb) > 0) {
    
    snap_tracking <- tracking_bdb[event == "ball_snap"]
    if (nrow(snap_tracking) > 0) {
      
      # --- Prepare Data in ONE Central Place ---
      setnames(snap_tracking, "playId", "play_id", skip_absent=TRUE)
      snap_tracking <- merge(snap_tracking, bdb_data$players[,.(nflId, position)], by="nflId", all.x=TRUE)
      snap_tracking <- merge(snap_tracking, bdb_data$plays[, .(gameId, playId, possessionTeam)], by.x = c("gameId", "play_id"), by.y = c("gameId", "playId"), all.x=TRUE)
      
      # --- START: STEP-BY-STEP FEATURE CALCULATION AND MERGING ---
      
      # Step 1: Calculate your original tracking features (THEY ARE NOW RESTORED)
      cat("  [FEAT_M3] Calculating original tracking features...\n")
      qb_coords <- snap_tracking[position == "QB", .(gameId, play_id, qb_x = x, qb_y = y)]
      ball_coords <- snap_tracking[is.na(nflId), .(gameId, play_id, ball_x = x, ball_y = y)]
      
      tracking_features <- data.table()
      if (nrow(qb_coords) > 0 && nrow(ball_coords) > 0) {
        depth_calc <- merge(qb_coords, ball_coords, by = c("gameId", "play_id"))
        depth_calc[, qb_depth := abs(qb_x - ball_x)]
        
        other_features <- snap_tracking[, .(
          db_spread = {
            db_y <- y[position %in% c("CB", "S", "FS", "SS", "DB") & !is.na(y)]
            if (length(db_y) > 1) sd(db_y, na.rm = TRUE) else 0
          },
          player_density = {
            all_coords <- data.table(x = x[!is.na(x)], y = y[!is.na(y)])
            if (nrow(all_coords) > 1) mean(dist(all_coords), na.rm = TRUE) else 0
          },
          avg_speed = mean(s, na.rm = TRUE), max_speed = max(s, na.rm = TRUE),
          speed_variance = var(s, na.rm = TRUE), avg_acceleration = mean(a, na.rm = TRUE),
          direction_variance = var(dir, na.rm = TRUE),
          orientation_alignment = {
            orientations <- o[!is.na(o)]
            if (length(orientations) > 1) {
              rad_o <- orientations * pi / 180
              1 - sqrt(mean(cos(rad_o), na.rm=T)^2 + mean(sin(rad_o), na.rm=T)^2)
            } else { 0 }
          }
        ), by = .(gameId, play_id)]
        
        tracking_features <- merge(depth_calc[, .(gameId, play_id, qb_depth)], other_features, by = c("gameId", "play_id"), all = TRUE)
      }
      
      # Step 2: Merge the first set of features into the main data frame
      model_df <- merge(model_df, tracking_features, by = c("gameId", "play_id"), all.x = TRUE)
      
      # Step 3: Calculate and merge the next feature sets one by one
      cat("  [FEAT_M3] Calculating and merging additional feature sets...\n")
      
      formation_features <- calculate_formation_geometry(tracking_bdb, bdb_data$players)
      setnames(formation_features, "playId", "play_id", skip_absent=TRUE)
      model_df <- merge(model_df, formation_features, by = c("gameId", "play_id"), all.x = TRUE)
      
      motion_features <- calculate_motion_features(tracking_bdb, bdb_data$players)
      setnames(motion_features, "playId", "play_id", skip_absent=TRUE)
      model_df <- merge(model_df, motion_features, by = c("gameId", "play_id"), all.x = TRUE)
      
      defensive_features <- calculate_defensive_features(tracking_bdb, bdb_data$players)
      setnames(defensive_features, "playId", "play_id", skip_absent=TRUE)
      model_df <- merge(model_df, defensive_features, by = c("gameId", "play_id"), all.x = TRUE)
      
      voronoi_feats <- calculate_voronoi_features(snap_tracking)
      model_df <- merge(model_df, voronoi_feats, by = c("gameId", "play_id"), all.x = TRUE)
      
      graph_feats <- calculate_graph_features(snap_tracking)
      model_df <- merge(model_df, graph_feats, by = c("gameId", "play_id"), all.x = TRUE)
      
      hull_feats <- calculate_convex_hull_features(snap_tracking)
      model_df <- merge(model_df, hull_feats, by = c("gameId", "play_id"), all.x = TRUE)
      
      potential_feats <- calculate_potential_field_features(snap_tracking)
      model_df <- merge(model_df, potential_feats, by = c("gameId", "play_id"), all.x = TRUE)
      
      cat("  [FEAT_M3] All tracking features successfully merged.\n")
    }
  }
  
  # Define all possible tracking feature columns to ensure they exist
  tracking_feature_cols <- c(
    "qb_depth", "db_spread", "player_density", "avg_speed", "max_speed", "speed_variance",
    "avg_acceleration", "direction_variance", "orientation_alignment",
    "formation_width", "formation_depth", "formation_area", "formation_compactness",
    "formation_symmetry", "formation_balance", "wr_spread", "wr_depth_variance",
    "te_alignment", "backfield_depth", "pocket_width", "formation_density",
    "line_compactness", "receiver_cluster_count", "avg_receiver_separation",
    "has_motion", "motion_player_count", "motion_distance", "motion_speed",
    "motion_direction_change", "wr_motion", "te_motion", "rb_motion",
    "motion_toward_los", "motion_lateral", "pre_snap_shifts",
    "def_front_width", "def_depth", "def_box_count", "def_coverage_depth",
    "def_pressure_alignment", "def_leverage", "def_gap_integrity", "def_underneath_coverage",
    "off_voronoi_area_mean", "def_voronoi_area_mean", "pitch_control_ratio", "def_voronoi_area_variance",
    "def_delaunay_edge_mean", "def_delaunay_edge_variance", "def_graph_density", "def_graph_avg_betweenness",
    "def_graph_clustering_coef", "def_graph_hub_score", "off_hull_area", "def_hull_area", "hull_area_ratio",
    "def_hull_aspect_ratio", "qb_pressure_potential", "running_lane_potential"
  )
  
  # Ensure all tracking feature columns exist and fill NAs
  for (col in tracking_feature_cols) { 
    if (!col %in% names(model_df)) model_df[, (col) := 0] 
  }
  
  for (col in names(model_df)) { 
    if(is.numeric(model_df[[col]])) set(model_df, which(is.na(model_df[[col]])), col, 0) 
  }
  
  cat("  [FEAT_M3] Enhanced feature set created and cleaned.\n")
  return(model_df)
}

# ───────────────────────────────────────────────────────────────────────────────
# 3) FEATURE CREATION
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [FEATURES] CREATING FEATURES FOR ALL MODELS ===\n")

# NOTE: Your feature creation is excellent and remains unchanged.
# We will create one final, consolidated feature set for our new model.
features_m1 <- create_model1_features(pbp_hist)
features_m2 <- create_model2_features(pbp_modern, participation_data)
features_m3 <- create_model3_features(bdb_data, participation_data, pbp_hist)

# ───────────────────────────────────────────────────────────────────────────────
# 4) MODEL TRAINING (v7 - GAM with Principled Feature Selection)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [TRAINING] TRAINING FINAL PREDICTIVE MODEL (GAM) ===\n")

suppressPackageStartupMessages(library(mgcv))

# --- Step 1: Define the "Keeper" Features Based on Statistical Significance ---
# This list is derived directly from your previous gam_summary output (p < 0.05).
# We are manually selecting only the variables that proved to be predictive.

keeper_linear_features <- c(
  "no_huddle", "third_down", "fourth_down", "shotgun_x_down", "two_minute_warning",
  "n_te", "red_zone_x_down", "n_rb", "n_wr", "trips_formation"
)

keeper_smooth_features <- c(
  "yardline_100", "wp", "down_x_distance", "direction_variance", "formation_compactness",
  "formation_symmetry", "formation_balance", "wr_spread", "backfield_depth",
  "def_graph_avg_betweenness", "def_graph_clustering_coef", "off_hull_area",
  "score_differential", "score_diff_x_time", "def_depth", "def_coverage_depth",
  "db_spread", "max_speed", "te_alignment", "formation_density", "off_voronoi_area_mean",
  "def_voronoi_area_mean", "pitch_control_ratio", "def_voronoi_area_variance"
)

total_keepers <- length(keeper_linear_features) + length(keeper_smooth_features)
cat("  [TRAIN_SETUP] Selected", total_keepers, 
    "statistically significant features from the initial 108.\n")

# --- Step 2: Build the New, Leaner Formula ---
# The formula is now built using only our curated list of "keeper" features.
smooth_terms <- paste0("s(", keeper_smooth_features, ")", collapse = " + ")
linear_terms <- paste(keeper_linear_features, collapse = " + ")

gam_formula_string <- paste("is_dropback ~", smooth_terms, "+", linear_terms)
gam_formula <- as.formula(gam_formula_string)

# --- Step 3: Train the Final, Optimized GAM ---
cat("  [TRAIN] Training OPTIMIZED GAM on", nrow(features_m3), "plays with", total_keepers, "features...\n")
cat("  [TRAIN] This will be significantly faster and more robust than the previous version.\n")

# Train the GAM using parallel processing.
final_gam_model <- gam(
  gam_formula, 
  data = features_m3, 
  family = binomial,
  control = gam.control(nthreads = n_cores) # Parallelization is enabled
)

# --- Step 4: Review the New, Improved Model ---
cat("  [TRAIN] GAM training complete. Displaying new model summary:\n")
print(summary(final_gam_model))

# ───────────────────────────────────────────────────────────────────────────────
# 5) PREDICTION & MASTER ANALYSIS DATA FRAME CREATION (v4 - FINAL)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [ANALYSIS_SETUP] CREATING THE MASTER ANALYSIS DATA FRAME ===\n")

# Start with our BDB feature set as the base
analysis_df <- copy(features_m3)

# Add the high-quality, calibrated predictions from our new GAM
cat("  [ANALYSIS_SETUP] Generating calibrated predictions from GAM...\n")
analysis_df[, gam_prediction := predict(final_gam_model, newdata = analysis_df, type = "response")]

cat("  [ANALYSIS_SETUP] Merging event and roster data...\n")

# --- THE FIX: Add the `epa` column to the PBP data we pull ---
pbp_for_analysis <- pbp_hist[season == 2022, .(
  old_game_id, play_id, 
  # Key columns for aggregation:
  posteam, epa, # <-- 'epa' IS NOW INCLUDED
  # Reporting columns:
  desc, week, home_team, away_team, 
  # Analysis columns:
  play_type, qb_scramble, sack, sack_player_id, qb_hit, 
  qb_hit_1_player_id, qb_hit_2_player_id
)]
pbp_for_analysis <- unique(pbp_for_analysis, by = c("old_game_id", "play_id"))

participation_full <- nflreadr::load_participation(2016:2023)
setDT(participation_full)
participation_full[, season := as.integer(substr(nflverse_game_id, 1, 4))]
participation_2022 <- participation_full[season == 2022, .(old_game_id, play_id, defense_players)]
participation_2022 <- unique(participation_2022, by = c("old_game_id", "play_id"))

pbp_for_analysis <- merge(pbp_for_analysis, participation_2022, by = c("old_game_id", "play_id"), all.x = TRUE)

# Now, merge this complete PBP data into our main analysis_df
analysis_df <- merge(analysis_df, pbp_for_analysis, by = c("old_game_id", "play_id"), all.x = TRUE)

# --- END FIX ---

rosters_2022 <- nflreadr::load_rosters(2022)
rosters_2022 <- setDT(rosters_2022)[, .(gsis_id, full_name, position, team)]

cat("  [ANALYSIS_SETUP] Master 'analysis_df' is ready for analysis.\n")
# ───────────────────────────────────────────────────────────────────────────────
# 6) SURPRISAL-WEIGHTED PASS RUSHER EVALUATION
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [SURPRISAL_EVAL] EVALUATING PASS RUSHERS WITH GAM PREDICTIONS ===\n")

if (exists("analysis_df")) {
  
  # The evaluation function is now simpler as we only have one model to test.
  evaluate_pass_rushers_gam <- function(data) {
    eval_data <- copy(data)
    
    # Calculate Surprisal based on the GAM's calibrated probability
    epsilon <- 1e-10
    eval_data[, surprisal := -log(fifelse(is_dropback == 1, 
                                          pmin(pmax(gam_prediction, epsilon), 1 - epsilon), 
                                          1 - pmin(pmax(gam_prediction, epsilon), 1 - epsilon)))]
    
    dropback_snaps <- eval_data[is_dropback == 1 & !is.na(surprisal)]
    
    # ... (The rest of this function is identical to the previous robust version) ...
    dropback_snaps[, defense_players := str_trim(gsub(";$", "", defense_players))]
    def_players_long <- dropback_snaps[!is.na(defense_players) & defense_players != "", .(
      gsis_id = unlist(strsplit(defense_players, ";")),
      surprisal = rep(surprisal, lengths(strsplit(defense_players, ";")))
    ), by = .(old_game_id, play_id)]
    
    player_exposure <- def_players_long[, .(
      weighted_pass_rush_snaps = sum(surprisal, na.rm = TRUE),
      raw_pass_rush_snaps = .N
    ), by = gsis_id]
    
    sacks_weighted <- dropback_snaps[sack == 1 & !is.na(sack_player_id), 
                                     .(weighted_sacks = sum(surprisal, na.rm = TRUE)), by = .(gsis_id = sack_player_id)]
    
    qb_hits_long <- rbindlist(list(
      dropback_snaps[!is.na(qb_hit_1_player_id), .(gsis_id = qb_hit_1_player_id, surprisal)],
      dropback_snaps[!is.na(qb_hit_2_player_id), .(gsis_id = qb_hit_2_player_id, surprisal)]
    ))
    qb_hits_weighted <- qb_hits_long[!is.na(gsis_id), .(weighted_qb_hits = sum(surprisal, na.rm = TRUE)), by = gsis_id]
    
    summary_table <- merge(player_exposure, sacks_weighted, by = "gsis_id", all.x = TRUE)
    summary_table <- merge(summary_table, qb_hits_weighted, by = "gsis_id", all.x = TRUE)
    summary_table[is.na(weighted_sacks), weighted_sacks := 0]
    summary_table[is.na(weighted_qb_hits), weighted_qb_hits := 0]
    summary_table <- summary_table[weighted_pass_rush_snaps > 0]
    
    summary_table[, total_disruption := weighted_sacks + weighted_qb_hits]
    summary_table[, disruption_rate := total_disruption / weighted_pass_rush_snaps]
    
    summary_table <- merge(summary_table, rosters_2022, by = "gsis_id", all.x = TRUE)
    pass_rusher_positions <- c("DE", "DT", "EDGE", "OLB", "ILB", "LB", "NT", "DL")
    summary_table <- summary_table[position %in% pass_rusher_positions & raw_pass_rush_snaps >= 100]
    
    setorder(summary_table, -disruption_rate)
    final_output <- summary_table[, .(Rank = .I, Player = full_name, Team = team, Position = position, Disruption_Rate = round(disruption_rate, 4), Total_Disruption = round(total_disruption, 2), Rush_Snaps = raw_pass_rush_snaps)]
    
    print(head(final_output, 30))
    return(invisible(final_output))
  }
  
  # Run the final evaluation
  rusher_eval_gam <- evaluate_pass_rushers_gam(analysis_df)
  
} else {
  cat("  [SURPRISAL_EVAL] Skipped: `analysis_df` object not found.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 6) MODEL PERFORMANCE EVALUATION (GAM)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [EVALUATION] EVALUATING FINAL GAM MODEL PERFORMANCE ===\n")

if (exists("analysis_df") && "gam_prediction" %in% names(analysis_df)) {
  
  cat("  [EVALUATION] Calculating AUC and Accuracy on the 2022 BDB training data...\n")
  
  # --- Area Under the Curve (AUC) ---
  # Measures the model's ability to correctly rank plays. 
  # A value of 0.5 is random chance, 1.0 is a perfect model.
  # An AUC above 0.90 is generally considered excellent for this type of problem.
  gam_auc <- pROC::auc(
    response = analysis_df$is_dropback,
    predictor = analysis_df$gam_prediction,
    quiet = TRUE
  )
  cat("  >>> Final Model AUC:", round(as.numeric(gam_auc), 4), "\n")
  
  # --- Accuracy and Confusion Matrix ---
  # Measures the model's correctness if we use a 0.5 probability threshold.
  threshold <- 0.5
  predicted_classes <- factor(ifelse(analysis_df$gam_prediction > threshold, 1, 0), levels = c(0, 1))
  actual_classes <- factor(analysis_df$is_dropback, levels = c(0, 1))
  
  # Using caret's confusionMatrix for a detailed report
  gam_conf_matrix <- caret::confusionMatrix(predicted_classes, actual_classes)
  
  cat("  >>> Final Model Accuracy:", round(gam_conf_matrix$overall['Accuracy'], 4), "\n\n")
  print(gam_conf_matrix)
  
} else {
  cat("  [EVALUATION] Skipped: `analysis_df` or GAM predictions not found.\n")
}


# ───────────────────────────────────────────────────────────────────────────────
# 7) ANALYSIS: IDENTIFY MOST SURPRISING PLAYS (GAM) - FINAL VERSION
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [ANALYSIS] IDENTIFYING THE MOST SURPRISING PLAYS OF 2022 (GAM) ===\n")

if (exists("analysis_df")) {
  
  # --- Part 1: The Most UNEXPECTED Dropbacks (Gutsy Pass Calls) ---
  unexpected_dropbacks <- analysis_df[is_dropback == 1]
  setorder(unexpected_dropbacks, gam_prediction)
  
  unexpected_dropback_report <- head(unexpected_dropbacks, 20)[, .(
    Rank = .I,
    ## MODIFICATION: Added the 'week' to the Game string
    Game = paste0("Week ", week, ": ", away_team, " @ ", home_team),
    Situation = paste0("Q", qtr, ", ", down, " & ", ydstogo),
    `Play Description` = substr(desc, 1, 120),
    `Predicted Dropback %` = paste0(round(gam_prediction * 100, 1), "%")
  )]
  
  cat("\n--- Top 20 Most UNEXPECTED Dropback Plays (Gutsy Calls) ---\n")
  print(unexpected_dropback_report)
  
  
  # --- Part 2: The Most UNEXPECTED Designed Runs (Deceptive Run Calls) ---
  unexpected_runs <- analysis_df[is_dropback == 0]
  setorder(unexpected_runs, -gam_prediction)
  
  unexpected_run_report <- head(unexpected_runs, 20)[, .(
    Rank = .I,
    ## MODIFICATION: Added the 'week' to the Game string
    Game = paste0("Week ", week, ": ", away_team, " @ ", home_team),
    Situation = paste0("Q", qtr, ", ", down, " & ", ydstogo),
    `Play Description` = substr(desc, 1, 120),
    `Predicted Dropback %` = paste0(round(gam_prediction * 100, 1), "%")
  )]
  
  cat("\n\n--- Top 20 Most UNEXPECTED Designed Run Plays (Deceptive Calls) ---\n")
  print(unexpected_run_report)
  
} else {
  cat("  [ANALYSIS] Skipped: `analysis_df` not found.\n")
}


# ───────────────────────────────────────────────────────────────────────────────
# 9) SURPRISAL-WEIGHTED PASS RUSHER EVALUATION (v3 - FINAL, HIERARCHICAL)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [SURPRISAL_EVAL] EVALUATING PASS RUSHERS (HIERARCHICAL METHOD) ===\n")

if (exists("analysis_df")) {
  
  # This is the new, definitive evaluation function that implements all the expert critiques.
  evaluate_pass_rushers_final <- function(data, rosters) {
    
    eval_data <- copy(data)
    
    # --- Step 1: Calculate Surprisal ---
    eval_data[, surprisal := -log(fifelse(is_dropback == 1, 
                                          pmin(pmax(gam_prediction, 1e-10), 1 - 1e-10), 
                                          1 - pmin(pmax(gam_prediction, 1e-10), 1 - 1e-10)))]
    
    dropback_snaps <- eval_data[is_dropback == 1 & !is.na(surprisal)]
    
    # --- Step 2: Calculate Raw Disruption Metrics ---
    dropback_snaps[, defense_players := str_trim(gsub(";$", "", defense_players))]
    def_players_long <- dropback_snaps[!is.na(defense_players) & defense_players != "", .(
      gsis_id = unlist(strsplit(defense_players, ";")),
      surprisal = rep(surprisal, lengths(strsplit(defense_players, ";")))
    ), by = .(old_game_id, play_id)]
    
    # Get total snaps and also the number of games played for our per-game filter
    player_exposure <- def_players_long[, .(
      weighted_pass_rush_snaps = sum(surprisal, na.rm = TRUE),
      raw_pass_rush_snaps = .N,
      games_played = uniqueN(old_game_id)
    ), by = gsis_id]
    
    sacks_weighted <- dropback_snaps[sack == 1 & !is.na(sack_player_id), 
                                     .(weighted_sacks = sum(surprisal, na.rm = TRUE)), by = .(gsis_id = sack_player_id)]
    
    qb_hits_long <- rbindlist(list(
      dropback_snaps[!is.na(qb_hit_1_player_id), .(gsis_id = qb_hit_1_player_id, surprisal)],
      dropback_snaps[!is.na(qb_hit_2_player_id), .(gsis_id = qb_hit_2_player_id, surprisal)]
    ))
    qb_hits_weighted <- qb_hits_long[!is.na(gsis_id), .(weighted_qb_hits = sum(surprisal, na.rm = TRUE)), by = gsis_id]
    
    summary_table <- merge(player_exposure, sacks_weighted, by = "gsis_id", all.x = TRUE)
    summary_table <- merge(summary_table, qb_hits_weighted, by = "gsis_id", all.x = TRUE)
    summary_table[is.na(weighted_sacks), weighted_sacks := 0]
    summary_table[is.na(weighted_qb_hits), weighted_qb_hits := 0]
    summary_table[, total_disruption := weighted_sacks + weighted_qb_hits]
    
    # --- Step 3: Apply the Advanced Fixes ---
    summary_table <- merge(summary_table, rosters, by = "gsis_id", all.x = TRUE)
    
    ## FIX 1: Create Role-Based Groups
    primary_rusher_positions <- c("DE", "DT", "EDGE", "NT", "DL")
    situational_rusher_positions <- c("LB", "ILB", "OLB", "DB")
    
    summary_table[, role := fcase(
      position %in% primary_rusher_positions, "Primary Rusher",
      position %in% situational_rusher_positions, "Situational Rusher",
      default = "Other"
    )]
    
    ## FIX 2: Apply a better per-game snap minimum
    min_snaps_per_game <- 15
    summary_table[, snaps_per_game := raw_pass_rush_snaps / games_played]
    summary_table <- summary_table[snaps_per_game >= min_snaps_per_game]
    
    ## FIX 3: Apply Role-Specific Bayesian Smoothing
    summary_table[, disruption_rate := fcase(
      role == "Primary Rusher", total_disruption / (weighted_pass_rush_snaps + 50), # Higher constant
      role == "Situational Rusher", total_disruption / (weighted_pass_rush_snaps + 20), # Lower constant
      default = 0
    )]
    
    # --- Step 4: Create and Print the Leaderboards ---
    
    # Leaderboard 1: Primary Rushers (EDGE / DL)
    primary_leaderboard <- summary_table[role == "Primary Rusher"]
    setorder(primary_leaderboard, -disruption_rate)
    
    primary_report <- primary_leaderboard[, .(
      Rank = .I, Player = full_name, Team = team, Position = position,
      `Credible Rate` = round(disruption_rate, 5),
      `Total Surprisal` = round(total_disruption, 2),
      `Rush Snaps` = raw_pass_rush_snaps
    )]
    
    cat("\n--- Final Leaderboard: Primary Pass Rushers (EDGE/DL) ---\n")
    print(head(primary_report, 30))
    
    # Leaderboard 2: Situational Rushers (LB / DB)
    situational_leaderboard <- summary_table[role == "Situational Rusher"]
    setorder(situational_leaderboard, -disruption_rate)
    
    situational_report <- situational_leaderboard[, .(
      Rank = .I, Player = full_name, Team = team, Position = position,
      `Credible Rate` = round(disruption_rate, 5),
      `Total Surprisal` = round(total_disruption, 2),
      `Rush Snaps` = raw_pass_rush_snaps
    )]
    
    cat("\n--- Final Leaderboard: Situational Rushers (LB/DB Blitzers) ---\n")
    print(head(situational_report, 20))
    
    return(invisible(list(primary = primary_report, situational = situational_report)))
  }
  
  # Run the final, robust evaluation
  final_leaderboards <- evaluate_pass_rushers_final(analysis_df, rosters_2022)
  
} else {
  cat("  [SURPRISAL_EVAL] Skipped: `analysis_df` object not found.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 9) FINAL EVALUATION: SURPRISAL-ADJUSTED PRESSURE QUALITY (SAPQ 2.0) - FINAL
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [SAPQ_EVAL] BUILDING THE DEFINITIVE BLENDED METRIC (SAPQ 2.0) ===\n")

if (exists("analysis_df")) {
  
  # --- Step 1: Prepare All Necessary Data Sources ---
  cat("  [SAPQ_EVAL] Loading and preparing all data sources...\n")
  player_play_df <- fread("player_play.csv")
  
  play_level_data <- analysis_df[, .(gameId, play_id, gam_prediction, is_dropback)]
  setnames(play_level_data, "play_id", "playId")
  
  # --- THE FIX: We will now use player names as the robust key for all merges ---
  
  # The BDB `players` table is our ground truth for nflId <-> Name mapping
  bdb_roster <- bdb_data$players[, .(nflId, displayName, position)]
  
  # The `nflreadr` roster is now ONLY for Name <-> Team mapping
  nflreadr_roster <- nflreadr::load_rosters(2022)
  nflreadr_roster <- setDT(nflreadr_roster)[, .(full_name, team)]
  nflreadr_roster <- unique(nflreadr_roster, by = "full_name") # Ensure one team per player
  
  # Get sack counts from PBP, aggregated by player NAME.
  pbp_sacks <- pbp_hist[season == 2022 & sack == 1, .(sack_player_name)]
  sacks_by_player <- pbp_sacks[!is.na(sack_player_name), .N, by = .(full_name = sack_player_name)]
  setnames(sacks_by_player, "N", "sacks")
  
  # --- Step 2: Calculate SAPQ scores per snap ---
  cat("  [SAPQ_EVAL] Calculating Surprisal and Pressure Quality for every snap...\n")
  player_pass_rush_snaps <- merge(
    player_play_df[wasInitialPassRusher == TRUE],
    play_level_data,
    by = c("gameId", "playId")
  )
  epsilon <- 1e-10
  player_pass_rush_snaps[, surprisal := -log(fifelse(is_dropback == 1, 
                                                     pmin(pmax(gam_prediction, epsilon), 1 - epsilon), 
                                                     1 - pmin(pmax(gam_prediction, epsilon), 1 - epsilon)))]
  k <- 1.0
  player_pass_rush_snaps[!is.na(timeToPressureAsPassRusher), pressure_quality := exp(-k * timeToPressureAsPassRusher)]
  player_pass_rush_snaps[is.na(pressure_quality), pressure_quality := 0]
  player_pass_rush_snaps[, sapq_score := surprisal * pressure_quality]
  
  # --- Step 3: Aggregate to the Player Level ---
  cat("  [SAPQ_EVAL] Aggregating final SAPQ scores...\n")
  sapq_leaderboard <- player_pass_rush_snaps[, .(
    avg_sapq = mean(sapq_score, na.rm = TRUE),
    total_sapq = sum(sapq_score, na.rm = TRUE),
    rush_snaps = .N,
    pressures = sum(causedPressure == TRUE, na.rm = TRUE)
  ), by = .(nflId)]
  
  # --- Step 4: Create the Blended Score and Finalize the Leaderboard ---
  cat("  [SAPQ_EVAL] Creating the blended performance score...\n")
  sapq_leaderboard[, z_avg_sapq := scale(avg_sapq)]
  sapq_leaderboard[, z_total_sapq := scale(total_sapq)]
  sapq_leaderboard[, performance_score := z_avg_sapq + z_total_sapq]
  
  # Merge with the BDB roster first to get names and positions.
  sapq_leaderboard <- merge(sapq_leaderboard, bdb_roster, by = "nflId")
  
  # Now, merge sacks and teams using the player's name as the key.
  setnames(sapq_leaderboard, "displayName", "full_name") # Standardize name column
  sapq_leaderboard <- merge(sapq_leaderboard, sacks_by_player, by = "full_name", all.x = TRUE)
  sapq_leaderboard <- merge(sapq_leaderboard, nflreadr_roster, by = "full_name", all.x = TRUE)
  sapq_leaderboard[is.na(sacks), sacks := 0]
  
  primary_rusher_positions <- c("DE", "DT", "EDGE", "NT", "DL")
  sapq_leaderboard[, role := fifelse(position %in% primary_rusher_positions, "Primary Rusher", "Situational")]
  sapq_leaderboard <- sapq_leaderboard[rush_snaps >= 150]
  
  setorder(sapq_leaderboard, -performance_score)
  
  final_report <- sapq_leaderboard[, .(
    Rank = .I, Player = full_name, Team = team, Position = position, Role = role,
    `Perf Score` = round(performance_score, 2),
    `Avg SAPQ` = round(avg_sapq, 4),
    `Total SAPQ` = round(total_sapq, 2),
    `Rush Snaps` = rush_snaps, Sacks = sacks, Pressures = pressures
  )]
  
  cat("\n--- Final Leaderboard: SAPQ 2.0 (Blended Rate + Volume) ---\n")
  print(head(final_report, 40))
  
} else {
  cat("  [SAPQ_EVAL] Skipped: `analysis_df` object not found.\n")
}
# ───────────────────────────────────────────────────────────────────────────────
# 10) NEW DIAGNOSTIC PLOTS
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [DIAGNOSTICS] ADVANCED METRIC HEALTH CHECKS ===\n")

if (exists("sapq_leaderboard")) {
  
  # --- Plot 1: Rate vs. Volume ---
  # This plot shows who is efficient vs. who is a workhorse
  rate_vs_volume_plot <- ggplot(sapq_leaderboard, aes(x = rush_snaps, y = avg_sapq, color = role)) +
    geom_point(aes(size = total_sapq), alpha = 0.7) +
    scale_size_continuous(range = c(3, 15)) +
    labs(
      title = "Pass Rusher Performance: Rate vs. Volume (Weeks 1-9, 2022)",
      subtitle = "Bubble size represents Total SAPQ. Top-right quadrant is elite.",
      x = "Total Pass Rush Snaps",
      y = "Average SAPQ (Efficiency)",
      color = "Player Role",
      size = "Total SAPQ (Impact)"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  
  cat("\n  [DIAGNOSTICS] Generating Rate vs. Volume plot...\n")
  print(rate_vs_volume_plot)
  
  # --- Plot 2: Surprisal Spread ---
  # This checks if our SAPQ score is biased by a few high-surprisal plays
  surprisal_spread_plot <- ggplot(player_pass_rush_snaps, aes(x = surprisal, y = sapq_score)) +
    geom_point(alpha = 0.1, color = "#0072B2") +
    geom_smooth(method = "gam", formula = y ~ s(x), color = "#D55E00") +
    labs(
      title = "SAPQ Score vs. Play Surprisal",
      subtitle = "A healthy, upward trend shows the metric rewards skill in surprising situations.",
      x = "Surprisal of Play Call",
      y = "SAPQ Score Generated on Snap"
    ) +
    theme_minimal()
  
  cat("\n  [DIAGNOSTICS] Generating Surprisal Spread plot...\n")
  print(surprisal_spread_plot)
  
}

# ───────────────────────────────────────────────────────────────────────────────
# 11) RELATIONSHIP ANALYSIS: SAPQ vs. TRADITIONAL METRICS (v2 - FINAL)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [ANALYSIS] EXPLORING THE RELATIONSHIP BETWEEN SAPQ AND TRADITIONAL STATS ===\n")

if (exists("sapq_leaderboard")) {
  
  # --- Step 1: Calculate Traditional Rate Metrics ---
  sapq_leaderboard[, pressure_rate := pressures / rush_snaps]
  sapq_leaderboard[, sack_rate := sacks / rush_snaps]
  
  # --- Step 2: Create a Diagnostic Scatter Plot ---
  cat("  [ANALYSIS] Generating plot of SAPQ vs. Pressure Rate...\n")
  
  suppressPackageStartupMessages(library(ggrepel))
  
  # --- THE FIX: Use the correct column name `full_name` for the labels ---
  relationship_plot <- ggplot(sapq_leaderboard, aes(x = pressure_rate, y = avg_sapq, color = role)) +
    geom_point(aes(size = total_sapq), alpha = 0.7) +
    scale_size_continuous(range = c(3, 15)) +
    geom_smooth(method = "lm", aes(group = 1), color = "red", linetype = "dashed", se = FALSE) +
    geom_text_repel(aes(label = ifelse(performance_score > 3.0 | avg_sapq > 0.005, full_name, "")), 
                    box.padding = 0.5, max.overlaps = 10, min.segment.length = 0) + # Use `full_name`
    scale_x_continuous(labels = scales::percent) +
    labs(
      title = "SAPQ Efficiency vs. Traditional Pressure Rate (Weeks 1-9, 2022)",
      subtitle = "Players above the red line generate higher quality pressure than their raw rate suggests.",
      x = "Pressure Rate (Pressures / Rush Snaps)",
      y = "Average SAPQ (Efficiency)",
      color = "Player Role",
      size = "Total SAPQ (Impact)"
    ) +
    theme_minimal() +
    theme(legend.position = "bottom")
  # --- END FIX ---
  
  print(relationship_plot)
  
} else {
  cat("  [ANALYSIS] Skipped: `sapq_leaderboard` not found.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 12) PLAYER VALUATION: RISERS & FALLERS (v2 - FINAL)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [VALUATION] IDENTIFYING PLAYERS PROMOTED OR DEMOTED BY SAPQ ===\n")

if (exists("sapq_leaderboard") && "pressure_rate" %in% names(sapq_leaderboard)) {
  
  # --- Step 1: Create Ranks for Both Metrics ---
  setorder(sapq_leaderboard, -performance_score)
  sapq_leaderboard[, sapq_rank := .I]
  
  setorder(sapq_leaderboard, -pressure_rate)
  sapq_leaderboard[, traditional_rank := .I]
  
  # --- Step 2: Calculate the Rank Difference ---
  sapq_leaderboard[, rank_diff := traditional_rank - sapq_rank]
  
  # --- Step 3: Identify the Biggest Risers (Underrated by Traditional Stats) ---
  setorder(sapq_leaderboard, -rank_diff)
  
  # --- THE FIX: Use the correct column name `full_name` and rename it to `Player` ---
  risers_report <- head(sapq_leaderboard, 15)[, .(
    Player = full_name, # Use the correct column name
    Team = team, 
    Position = position, 
    `SAPQ Rank` = sapq_rank, 
    `Pressure Rate Rank` = traditional_rank,
    `Rank Change` = paste0("+", rank_diff)
  )]
  
  cat("\n--- SAPQ Top 15 Risers: The Underrated --- \n")
  cat("These players are more valuable than their raw pressure numbers suggest.\n")
  print(risers_report)
  
  # --- Step 4: Identify the Biggest Fallers (Overrated by Traditional Stats) ---
  setorder(sapq_leaderboard, rank_diff)
  
  # --- THE FIX: Use the correct column name `full_name` and rename it to `Player` ---
  fallers_report <- head(sapq_leaderboard, 15)[, .(
    Player = full_name, # Use the correct column name
    Team = team, 
    Position = position, 
    `SAPQ Rank` = sapq_rank, 
    `Pressure Rate Rank` = traditional_rank,
    `Rank Change` = rank_diff
  )]
  
  cat("\n--- SAPQ Top 15 Fallers: The Overrated ---\n")
  cat("These players' high pressure rates are less impactful (e.g., 'empty calories').\n")
  print(fallers_report)
  
} else {
  cat("  [VALUATION] Skipped: `sapq_leaderboard` not found or missing pressure_rate.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 9) FINAL ANALYSIS: OFFENSIVE DNA - SURPRISAL vs. SUCCESS (v5 - FINAL)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [FINAL_ANALYSIS] MAPPING THE OFFENSIVE DNA OF EVERY TEAM ===\n")

if (exists("analysis_df")) {
  
  suppressPackageStartupMessages(library(ggrepel))
  suppressPackageStartupMessages(library(ggthemes))
  
  # --- Step 1: Calculate Per-Play Surprisal and Success ---
  epsilon <- 1e-10
  analysis_df[, surprisal := -log(fifelse(is_dropback == 1, 
                                          pmin(pmax(gam_prediction, epsilon), 1 - epsilon), 
                                          1 - pmin(pmax(gam_prediction, epsilon), 1 - epsilon)))]
  
  # --- Step 2: Aggregate to the Team Level ---
  team_analysis <- analysis_df[!is.na(posteam), .(
    avg_surprisal = mean(surprisal, na.rm = TRUE),
    avg_epa_per_play = mean(epa, na.rm = TRUE),
    total_plays = .N
  ), by = .(team = posteam)]
  
  team_analysis <- team_analysis[total_plays > 100]
  
  # --- Step 3: The "Moneyball Plot" ---
  cat("  [FINAL_ANALYSIS] Generating the Surprisal vs. Success plot...\n")
  
  # Load team logos/colors for the plot
  team_logos <- nflreadr::load_teams()
  setDT(team_logos)
  plot_data <- merge(team_analysis, team_logos[, .(team_abbr, team_logo_espn, team_color)], by.x = "team", by.y = "team_abbr")
  
  league_avg_epa <- weighted.mean(plot_data$avg_epa_per_play, plot_data$total_plays)
  league_avg_surprisal <- weighted.mean(plot_data$avg_surprisal, plot_data$total_plays)
  
  # --- THE FIX: Switched from geom_image to the more robust geom_text_repel ---
  offensive_dna_plot <- ggplot(plot_data, aes(x = avg_epa_per_play, y = avg_surprisal)) +
    geom_hline(yintercept = league_avg_surprisal, color = "grey", linetype = "dashed") +
    geom_vline(xintercept = league_avg_epa, color = "grey", linetype = "dashed") +
    # Use text labels instead of images for reliability and speed
    geom_text_repel(aes(label = team, color = team_color), size = 4, fontface = "bold", show.legend = FALSE) +
    scale_color_identity() + # Tell ggplot to use the actual team colors
    theme_fivethirtyeight() +
    labs(
      title = "Offensive DNA: Which Teams are Both Unpredictable and Effective?",
      subtitle = "NFL 2022, Weeks 1-9. Based on pre-snap dropback probability.",
      x = "Offensive Efficiency (EPA per Play)",
      y = "Play-Calling Unpredictability (Average Surprisal)"
    ) +
    annotate("text", x = league_avg_epa + 0.1, y = league_avg_surprisal + 0.02, label = "Innovative & Effective", color = "darkgreen", fontface = "bold.italic") +
    annotate("text", x = league_avg_epa - 0.1, y = league_avg_surprisal - 0.02, label = "Predictable & Ineffective", color = "darkred", fontface = "bold.italic") +
    annotate("text", x = league_avg_epa - 0.1, y = league_avg_surprisal + 0.02, label = "Chaotic & Ineffective", color = "darkorange", fontface = "bold.italic") +
    annotate("text", x = league_avg_epa + 0.1, y = league_avg_surprisal - 0.02, label = "Predictable Powerhouses", color = "darkblue", fontface = "bold.italic")
  
  print(offensive_dna_plot)
  
  # --- Step 4: Print the Most Interesting Teams ---
  # --- THE FIX: Create new, sorted tables for reporting ---
  
  cat("\n--- Top 5 Most Unpredictable & Effective Teams ---\n")
  unpredictable_effective <- plot_data[avg_epa_per_play > league_avg_epa & avg_surprisal > league_avg_surprisal]
  setorder(unpredictable_effective, -avg_surprisal)
  print(head(unpredictable_effective[, .(Team = team, `Avg Surprisal` = round(avg_surprisal, 3), `Avg EPA` = round(avg_epa_per_play, 3))]), 5)
  
  cat("\n--- Top 5 Most Predictable & Effective Teams ---\n")
  predictable_effective <- plot_data[avg_epa_per_play > league_avg_epa & avg_surprisal < league_avg_surprisal]
  setorder(predictable_effective, +avg_surprisal)
  print(head(predictable_effective[, .(Team = team, `Avg Surprisal` = round(avg_surprisal, 3), `Avg EPA` = round(avg_epa_per_play, 3))]), 5)
  
} else {
  cat("  [FINAL_ANALYSIS] Skipped: `analysis_df` not found.\n")
}

# ═══════════════════════════════════════════════════════════════════════════════
# COMPREHENSIVE MODEL EVALUATION SUITE FOR DROPBACK PREDICTION MODEL
# ═══════════════════════════════════════════════════════════════════════════════

cat("\n=== [MODEL_EVAL] COMPREHENSIVE MODEL EVALUATION ===\n")

# Load required libraries for advanced plotting
suppressPackageStartupMessages({
  library(ggplot2)
  library(gridExtra)
  library(viridis)
  library(ROCR)
  library(scales)
  library(reshape2)
  library(cowplot)
})

# ───────────────────────────────────────────────────────────────────────────────
# 1) PERFORMANCE METRICS CALCULATION
# ───────────────────────────────────────────────────────────────────────────────

if (exists("analysis_df") && exists("final_gam_model")) {
  
  # Calculate all predictions and metrics
  analysis_df[, predicted_class := ifelse(gam_prediction >= 0.5, 1, 0)]
  
  # Confusion Matrix Components
  confusion_matrix <- table(Actual = analysis_df$is_dropback, 
                           Predicted = analysis_df$predicted_class)
  
  # Calculate metrics
  tp <- confusion_matrix[2,2]
  tn <- confusion_matrix[1,1]
  fp <- confusion_matrix[1,2]
  fn <- confusion_matrix[2,1]
  
  accuracy <- (tp + tn) / sum(confusion_matrix)
  precision <- tp / (tp + fp)
  recall <- tp / (tp + fn)
  f1_score <- 2 * (precision * recall) / (precision + recall)
  specificity <- tn / (tn + fp)
  
  cat("\n--- Basic Performance Metrics ---\n")
  cat(sprintf("Accuracy: %.4f\n", accuracy))
  cat(sprintf("Precision: %.4f\n", precision))
  cat(sprintf("Recall (Sensitivity): %.4f\n", recall))
  cat(sprintf("Specificity: %.4f\n", specificity))
  cat(sprintf("F1 Score: %.4f\n", f1_score))
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 2) ROC CURVE AND AUC
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Generating ROC Curve...\n")
  
  # Calculate ROC curve
  pred_obj <- prediction(analysis_df$gam_prediction, analysis_df$is_dropback)
  perf_obj <- performance(pred_obj, "tpr", "fpr")
  auc_value <- performance(pred_obj, "auc")@y.values[[1]]
  
  # Create ROC plot
  roc_data <- data.frame(
    fpr = perf_obj@x.values[[1]],
    tpr = perf_obj@y.values[[1]]
  )
  
  roc_plot <- ggplot(roc_data, aes(x = fpr, y = tpr)) +
    geom_line(color = "#0072B2", size = 1.5) +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "gray50") +
    geom_area(alpha = 0.2, fill = "#0072B2") +
    annotate("text", x = 0.7, y = 0.3, 
             label = sprintf("AUC = %.4f", auc_value), 
             size = 6, fontface = "bold") +
    labs(title = "ROC Curve - Dropback Prediction Model",
         subtitle = "Higher AUC indicates better discrimination ability",
         x = "False Positive Rate (1 - Specificity)",
         y = "True Positive Rate (Sensitivity)") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"),
          plot.subtitle = element_text(size = 12))
  
  print(roc_plot)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 3) PRECISION-RECALL CURVE
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Generating Precision-Recall Curve...\n")
  
  pr_perf <- performance(pred_obj, "prec", "rec")
  
  pr_data <- data.frame(
    recall = pr_perf@x.values[[1]],
    precision = pr_perf@y.values[[1]]
  )
  pr_data <- pr_data[!is.na(pr_data$precision), ]
  
  # Calculate area under PR curve
  auprc <- trapz(pr_data$recall, pr_data$precision)
  
  pr_plot <- ggplot(pr_data, aes(x = recall, y = precision)) +
    geom_line(color = "#D55E00", size = 1.5) +
    geom_area(alpha = 0.2, fill = "#D55E00") +
    geom_hline(yintercept = mean(analysis_df$is_dropback), 
               linetype = "dashed", color = "gray50") +
    annotate("text", x = 0.3, y = 0.3, 
             label = sprintf("AUPRC = %.4f", auprc), 
             size = 6, fontface = "bold") +
    labs(title = "Precision-Recall Curve",
         subtitle = "Especially important for imbalanced datasets",
         x = "Recall (Sensitivity)",
         y = "Precision (Positive Predictive Value)") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"))
  
  print(pr_plot)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 4) CALIBRATION PLOT
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Generating Calibration Plot...\n")
  
  # Create calibration bins
  n_bins <- 20
  analysis_df[, calibration_bin := cut(gam_prediction, 
                                       breaks = seq(0, 1, 1/n_bins), 
                                       include.lowest = TRUE)]
  
  calibration_data <- analysis_df[!is.na(calibration_bin), .(
    mean_predicted = mean(gam_prediction),
    fraction_positive = mean(is_dropback),
    count = .N
  ), by = calibration_bin]
  
  # Calculate calibration metrics
  calibration_slope <- coef(lm(fraction_positive ~ mean_predicted, 
                              data = calibration_data, 
                              weights = count))[2]
  
  ece <- weighted.mean(abs(calibration_data$mean_predicted - 
                          calibration_data$fraction_positive), 
                      calibration_data$count)
  
  calibration_plot <- ggplot(calibration_data, 
                            aes(x = mean_predicted, y = fraction_positive)) +
    geom_point(aes(size = count), color = "#009E73", alpha = 0.7) +
    geom_smooth(method = "loess", se = TRUE, color = "#0072B2", fill = "#0072B2") +
    geom_abline(intercept = 0, slope = 1, linetype = "dashed", color = "red") +
    geom_text(aes(label = sprintf("n=%d", count)), 
              vjust = -1, size = 3) +
    scale_size_continuous(range = c(3, 10), guide = "none") +
    annotate("text", x = 0.7, y = 0.2, 
             label = sprintf("ECE = %.4f\nSlope = %.3f", ece, calibration_slope),
             size = 5, fontface = "bold") +
    labs(title = "Model Calibration Plot",
         subtitle = "Points should lie close to the diagonal for well-calibrated predictions",
         x = "Mean Predicted Probability",
         y = "Fraction of Actual Dropbacks") +
    theme_minimal() +
    coord_fixed() +
    theme(plot.title = element_text(size = 16, face = "bold"))
  
  print(calibration_plot)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 5) CONFUSION MATRIX HEATMAP
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Generating Confusion Matrix Heatmap...\n")
  
  # Normalize confusion matrix
  conf_matrix_norm <- confusion_matrix / rowSums(confusion_matrix)
  
  # Prepare data for plotting
  conf_data <- as.data.frame(as.table(conf_matrix_norm))
  names(conf_data) <- c("Actual", "Predicted", "Value")
  
  # Add raw counts
  conf_data$Count <- as.vector(confusion_matrix)
  conf_data$Label <- sprintf("%.2f%%\n(n=%d)", 
                            conf_data$Value * 100, 
                            conf_data$Count)
  
  confusion_heatmap <- ggplot(conf_data, aes(x = Predicted, y = Actual, fill = Value)) +
    geom_tile() +
    geom_text(aes(label = Label), size = 6) +
    scale_fill_gradient2(low = "white", mid = "#FDB863", high = "#B2182B", 
                        midpoint = 0.5, limits = c(0, 1),
                        labels = percent_format()) +
    scale_x_discrete(labels = c("0" = "Run", "1" = "Dropback")) +
    scale_y_discrete(labels = c("0" = "Run", "1" = "Dropback")) +
    labs(title = "Confusion Matrix (Row-Normalized)",
         subtitle = sprintf("Overall Accuracy: %.2f%%", accuracy * 100),
         x = "Predicted Class",
         y = "Actual Class",
         fill = "Proportion") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"),
          axis.text = element_text(size = 12))
  
  print(confusion_heatmap)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 6) FEATURE IMPORTANCE ANALYSIS
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Analyzing Feature Importance...\n")
  
  # Extract GAM summary
  gam_summary <- summary(final_gam_model)
  
  # Process smooth terms
  smooth_importance <- as.data.table(gam_summary$s.table, keep.rownames = "Feature")
  smooth_importance[, Feature := gsub("s\\(|\\)", "", Feature)]
  smooth_importance[, Type := "Smooth"]
  smooth_importance[, Importance := abs(.SD[[3]])]  # F-statistic
  smooth_importance[, Significance := .SD[[4]]]     # p-value
  
  # Process parametric terms
  param_importance <- as.data.table(gam_summary$p.table, keep.rownames = "Feature")
  param_importance <- param_importance[Feature != "(Intercept)"]
  param_importance[, Type := "Linear"]
  param_importance[, Importance := abs(.SD[[3]])]   # t-value
  param_importance[, Significance := .SD[[4]]]      # p-value
  
  # Combine
  all_importance <- rbind(
    smooth_importance[, .(Feature, Type, Importance, Significance)],
    param_importance[, .(Feature, Type, Importance, Significance)]
  )
  
  # Add significance stars
  all_importance[, sig_stars := fcase(
    Significance < 0.001, "***",
    Significance < 0.01, "**",
    Significance < 0.05, "*",
    default = ""
  )]
  
  setorder(all_importance, -Importance)
  top_features <- head(all_importance, 30)
  
  feature_importance_plot <- ggplot(top_features, 
                                   aes(x = reorder(Feature, Importance), 
                                       y = Importance, 
                                       fill = Type)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = sig_stars), hjust = -0.2, size = 5) +
    coord_flip() +
    scale_fill_manual(values = c("Smooth" = "#0072B2", "Linear" = "#D55E00")) +
    scale_y_log10() +
    labs(title = "Top 30 Most Important Features",
         subtitle = "Statistical importance (F-stat for smooths, |t-stat| for linear)",
         x = "Feature",
         y = "Importance (log scale)",
         fill = "Term Type") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"),
          legend.position = "bottom")
  
  print(feature_importance_plot)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 7) PARTIAL DEPENDENCE PLOTS (TOP 9 FEATURES)
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Generating Partial Dependence Plots...\n")
  
  # Select top 9 continuous features
  top_continuous <- all_importance[Type == "Smooth"][1:min(9, .N), Feature]
  
  # Create grid of partial dependence plots
  pd_plots <- list()
  
  for (i in seq_along(top_continuous)) {
    feat <- top_continuous[i]
    
    # Create prediction data
    feat_range <- seq(min(analysis_df[[feat]]), 
                     max(analysis_df[[feat]]), 
                     length.out = 100)
    
    # Hold other features at median
    pd_data <- analysis_df[1:100]
    for (col in names(pd_data)) {
      if (is.numeric(pd_data[[col]])) {
        pd_data[[col]] <- median(analysis_df[[col]], na.rm = TRUE)
      }
    }
    pd_data[[feat]] <- feat_range
    
    # Predict
    pd_data$prediction <- predict(final_gam_model, newdata = pd_data, type = "response")
    
    # Create plot
    pd_plots[[i]] <- ggplot(pd_data, aes_string(x = feat, y = "prediction")) +
      geom_line(color = "#0072B2", size = 1.5) +
      geom_rug(data = analysis_df[sample(.N, min(1000, .N))], 
               aes_string(x = feat), 
               inherit.aes = FALSE, alpha = 0.1) +
      labs(title = feat,
           y = "P(Dropback)") +
      theme_minimal() +
      theme(plot.title = element_text(size = 10))
  }
  
  pd_grid <- plot_grid(plotlist = pd_plots, ncol = 3)
  title <- ggdraw() + draw_label("Partial Dependence Plots - Top Features", 
                                 fontface = "bold", size = 16)
  pd_final <- plot_grid(title, pd_grid, ncol = 1, rel_heights = c(0.05, 0.95))
  
  print(pd_final)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 8) PREDICTION DISTRIBUTION BY CLASS
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Generating Prediction Distribution Plot...\n")
  
  distribution_plot <- ggplot(analysis_df, 
                             aes(x = gam_prediction, 
                                 fill = factor(is_dropback))) +
    geom_histogram(alpha = 0.7, position = "identity", bins = 50) +
    scale_fill_manual(values = c("0" = "#E69F00", "1" = "#56B4E9"),
                     labels = c("0" = "Run", "1" = "Dropback")) +
    geom_vline(xintercept = 0.5, linetype = "dashed", size = 1) +
    facet_wrap(~ factor(is_dropback, labels = c("Actual: Run", "Actual: Dropback")), 
               scales = "free_y") +
    labs(title = "Distribution of Predicted Probabilities by True Class",
         subtitle = "Good separation indicates strong discriminative ability",
         x = "Predicted Probability of Dropback",
         y = "Count",
         fill = "True Class") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"))
  
  print(distribution_plot)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 9) THRESHOLD ANALYSIS
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Performing Threshold Analysis...\n")
  
  # Calculate metrics across thresholds
  thresholds <- seq(0.1, 0.9, 0.01)
  threshold_metrics <- data.table()
  
  for (thresh in thresholds) {
    pred_class <- ifelse(analysis_df$gam_prediction >= thresh, 1, 0)
    
    tp <- sum(pred_class == 1 & analysis_df$is_dropback == 1)
    tn <- sum(pred_class == 0 & analysis_df$is_dropback == 0)
    fp <- sum(pred_class == 1 & analysis_df$is_dropback == 0)
    fn <- sum(pred_class == 0 & analysis_df$is_dropback == 1)
    
    precision <- tp / (tp + fp + 1e-10)
    recall <- tp / (tp + fn + 1e-10)
    f1 <- 2 * (precision * recall) / (precision + recall + 1e-10)
    
    threshold_metrics <- rbind(threshold_metrics, 
                              data.table(threshold = thresh,
                                       precision = precision,
                                       recall = recall,
                                       f1_score = f1))
  }
  
  # Melt for plotting
  threshold_long <- melt(threshold_metrics, 
                        id.vars = "threshold",
                        variable.name = "metric",
                        value.name = "value")
  
  threshold_plot <- ggplot(threshold_long, 
                          aes(x = threshold, y = value, color = metric)) +
    geom_line(size = 1.5) +
    geom_vline(xintercept = 0.5, linetype = "dashed", alpha = 0.5) +
    geom_vline(xintercept = threshold_metrics[which.max(f1_score), threshold],
               linetype = "dotted", color = "red", size = 1) +
    scale_color_manual(values = c("precision" = "#0072B2", 
                                 "recall" = "#D55E00", 
                                 "f1_score" = "#009E73"),
                      labels = c("Precision", "Recall", "F1 Score")) +
    annotate("text", 
            x = threshold_metrics[which.max(f1_score), threshold] + 0.05,
            y = 0.5,
            label = sprintf("Optimal F1\n@ %.3f", 
                           threshold_metrics[which.max(f1_score), threshold]),
            size = 4) +
    labs(title = "Performance Metrics Across Decision Thresholds",
         subtitle = "Shows trade-off between precision and recall",
         x = "Decision Threshold",
         y = "Metric Value",
         color = "Metric") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"),
          legend.position = "bottom")
  
  print(threshold_plot)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 10) SITUATIONAL PERFORMANCE ANALYSIS
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Analyzing Situational Performance...\n")
  
  # Calculate performance by game situation
  situational_performance <- analysis_df[, .(
    accuracy = mean(predicted_class == is_dropback),
    precision = sum(predicted_class == 1 & is_dropback == 1) / 
                sum(predicted_class == 1),
    recall = sum(predicted_class == 1 & is_dropback == 1) / 
             sum(is_dropback == 1),
    avg_confidence = mean(ifelse(predicted_class == 1, 
                                gam_prediction, 
                                1 - gam_prediction)),
    n_plays = .N
  ), by = .(down, ydstogo_bucket = cut(ydstogo, 
                                       breaks = c(0, 3, 7, 10, Inf),
                                       labels = c("1-3", "4-7", "8-10", "11+")))]
  
  situational_heatmap <- ggplot(situational_performance[!is.na(down)], 
                               aes(x = ydstogo_bucket, y = factor(down), 
                                   fill = accuracy)) +
    geom_tile() +
    geom_text(aes(label = sprintf("%.1f%%\n(n=%d)", 
                                 accuracy * 100, n_plays)), 
              size = 3) +
    scale_fill_gradient2(low = "#D55E00", mid = "white", high = "#0072B2",
                        midpoint = 0.85, limits = c(0.7, 1),
                        labels = percent_format()) +
    labs(title = "Model Accuracy by Down and Distance",
         subtitle = "Shows where the model performs best/worst",
         x = "Yards to Go",
         y = "Down",
         fill = "Accuracy") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"))
  
  print(situational_heatmap)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 11) RESIDUAL ANALYSIS
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Performing Residual Analysis...\n")
  
  # Calculate deviance residuals
  analysis_df[, deviance_residual := 
    sign(is_dropback - gam_prediction) * 
    sqrt(-2 * (is_dropback * log(pmax(gam_prediction, 1e-10)) + 
               (1 - is_dropback) * log(pmax(1 - gam_prediction, 1e-10))))]
  
  # Residual plots
  residual_plots <- list()
  
  # 1. Residuals vs Fitted
  residual_plots[[1]] <- ggplot(analysis_df[sample(.N, min(10000, .N))], 
                                aes(x = gam_prediction, y = deviance_residual)) +
    geom_point(alpha = 0.1, color = "#0072B2") +
    geom_smooth(method = "loess", color = "red") +
    geom_hline(yintercept = 0, linetype = "dashed") +
    labs(title = "Residuals vs Fitted Values",
         x = "Fitted Probability",
         y = "Deviance Residual") +
    theme_minimal()
  
  # 2. Q-Q plot
  residual_plots[[2]] <- ggplot(analysis_df[sample(.N, min(10000, .N))], 
                                aes(sample = deviance_residual)) +
    stat_qq(color = "#0072B2", alpha = 0.5) +
    stat_qq_line(color = "red") +
    labs(title = "Normal Q-Q Plot of Residuals",
         x = "Theoretical Quantiles",
         y = "Sample Quantiles") +
    theme_minimal()
  
  # 3. Residual distribution
  residual_plots[[3]] <- ggplot(analysis_df, 
                                aes(x = deviance_residual)) +
    geom_histogram(bins = 50, fill = "#0072B2", alpha = 0.7) +
    geom_vline(xintercept = 0, linetype = "dashed", color = "red") +
    labs(title = "Distribution of Deviance Residuals",
         x = "Deviance Residual",
         y = "Count") +
    theme_minimal()
  
  residual_grid <- plot_grid(plotlist = residual_plots, ncol = 3)
  residual_title <- ggdraw() + draw_label("Residual Diagnostic Plots", 
                                         fontface = "bold", size = 16)
  residual_final <- plot_grid(residual_title, residual_grid, 
                             ncol = 1, rel_heights = c(0.05, 0.95))
  
  print(residual_final)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 12) PERFORMANCE SUMMARY DASHBOARD
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Creating Performance Summary Dashboard...\n")
  
  # Create summary metrics table
  summary_metrics <- data.table(
    Metric = c("AUC-ROC", "AUC-PR", "Accuracy", "Precision", 
               "Recall", "F1 Score", "ECE", "Brier Score"),
    Value = c(
      auc_value,
      auprc,
      accuracy,
      precision,
      recall,
      f1_score,
      ece,
      mean((analysis_df$gam_prediction - analysis_df$is_dropback)^2)
    )
  )
  
  summary_metrics[, Value := round(Value, 4)]
  summary_metrics[, Performance := fcase(
    Metric %in% c("AUC-ROC", "AUC-PR") & Value > 0.9, "Excellent",
    Metric %in% c("AUC-ROC", "AUC-PR") & Value > 0.8, "Good",
    Metric %in% c("Accuracy", "Precision", "Recall", "F1 Score") & Value > 0.85, "Excellent",
    Metric %in% c("Accuracy", "Precision", "Recall", "F1 Score") & Value > 0.75, "Good",
    Metric == "ECE" & Value < 0.05, "Excellent",
    Metric == "ECE" & Value < 0.10, "Good",
    Metric == "Brier Score" & Value < 0.15, "Excellent",
    Metric == "Brier Score" & Value < 0.20, "Good",
    default = "Fair"
  )]
  
  # Create summary plot
  summary_plot <- ggplot(summary_metrics, 
                        aes(x = reorder(Metric, Value), y = Value, 
                            fill = Performance)) +
    geom_bar(stat = "identity") +
    geom_text(aes(label = sprintf("%.4f", Value)), 
              hjust = -0.1, size = 4) +
    coord_flip() +
    scale_fill_manual(values = c("Excellent" = "#009E73", 
                                "Good" = "#56B4E9", 
                                "Fair" = "#E69F00")) +
    scale_y_continuous(limits = c(0, 1.1)) +
    labs(title = "Model Performance Summary",
         subtitle = "Key metrics for binary classification performance",
         x = "Metric",
         y = "Value",
         fill = "Performance") +
    theme_minimal() +
    theme(plot.title = element_text(size = 16, face = "bold"),
          legend.position = "bottom")
  
  print(summary_plot)
  
  # ───────────────────────────────────────────────────────────────────────────────
  # 13) FEATURE INTERACTION ANALYSIS (TOP INTERACTIONS)
  # ───────────────────────────────────────────────────────────────────────────────
  
  cat("\n[EVAL] Analyzing Feature Interactions...\n")
  
  # Select top features for interaction analysis
  top_features_for_interaction <- head(all_importance[Type == "Smooth", Feature], 6)
  
  # Create interaction heatmap for one pair as example
  if (length(top_features_for_interaction) >= 2) {
    feat1 <- top_features_for_interaction[1]
    feat2 <- top_features_for_interaction[2]
    
    # Create 2D bins
    analysis_df[, feat1_bin := cut(get(feat1), breaks = 20)]
    analysis_df[, feat2_bin := cut(get(feat2), breaks = 20)]
    
    interaction_data <- analysis_df[!is.na(feat1_bin) & !is.na(feat2_bin), .(
      avg_prediction = mean(gam_prediction),
      n_obs = .N
    ), by = .(feat1_bin, feat2_bin)]
    
    # Extract numeric values for plotting
    interaction_data[, feat1_value := as.numeric(gsub("\\(|\\[|,.*", "", feat1_bin))]
    interaction_data[, feat2_value := as.numeric(gsub("\\(|\\[|,.*", "", feat2_bin))]
    
    interaction_plot <- ggplot(interaction_data[n_obs > 10], 
                              aes(x = feat1_value, y = feat2_value, 
                                  fill = avg_prediction)) +
      geom_tile() +
      scale_fill_gradient2(low = "#D55E00", mid = "white", high = "#0072B2",
                          midpoint = 0.5, limits = c(0, 1),
                          labels = percent_format()) +
      labs(title = sprintf("Feature Interaction: %s vs %s", feat1, feat2),
           subtitle = "Shows how prediction changes with both features",
           x = feat1,
           y = feat2,
           fill = "Avg P(Dropback)") +
      theme_minimal() +
      theme(plot.title = element_text(size = 16, face = "bold"))
}
}
# Make sure these libraries are loaded at the top of your script
library(ggplot2)
library(ggrepel)
library(viridis)
library(data.table)

# Create a directory for the plots if it doesn't exist
if (!dir.exists("plots")) {
  dir.create("plots")
  cat("[SETUP] Created 'plots' directory for output images.\n")
}

# Your `evaluate_pass_rushers` function definition remains the same.
# We assume it has been defined before this block is run.
# ... (your function definition here) ...

for (yr in 2022) {
  cat("\n=== Surprisal-Weighted Pass Rushers for Season", yr, "===\n")
  # Assuming df_base is your main data table
  season_data <- df_base[season == yr & week >= 1 & week <= 9] 
  res <- evaluate_pass_rushers(season_data)
  print(head(res, 30))
}

# --- PLOTTING SECTION (MODIFIED TO SAVE EACH PLOT) ---

# Plot 1: Top 50 by Raw Disruption Rate
cat("\n[PLOT] Generating and saving Plot 1: Top 50 by Raw Disruption Rate...\n")
top50_raw <- res[!is.na(YPC_Diff)][order(-Raw_Disruption_Rate)][1:50]

plot1_raw_disruption <- ggplot(top50_raw, aes(x = YPC_Diff, y = Raw_Disruption_Rate, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 10) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(
    title = "Top 50 Pass Rushers by Raw Disruption Rate",
    x = "Yards Per Carry Allowed (On - Off Field)",
    y = "Raw Disruption Rate",
    caption = "2022 Season | Top 50 players by metric"
  ) +
  theme_minimal(base_size = 14)

print(plot1_raw_disruption)
ggsave("plots/01_top50_raw_disruption.png", plot1_raw_disruption, width = 12, height = 9, dpi = 300)
cat("  [SAVE] Plot saved to plots/01_top50_raw_disruption.png\n")


# Plot 2: Top 50 by Weighted Disruption Rate
cat("\n[PLOT] Generating and saving Plot 2: Top 50 by Weighted Disruption Rate...\n")
top50_weighted <- res[!is.na(YPC_Diff)][order(-Disruption_Rate)][1:50]

plot2_weighted_disruption <- ggplot(top50_weighted, aes(x = YPC_Diff, y = Disruption_Rate, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 10) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(
    title = "Top 50 Pass Rushers by Weighted Disruption Rate",
    x = "Yards Per Carry Allowed (On - Off Field)",
    y = "Weighted Disruption Rate",
    caption = "2022 Season | Top 50 players by metric"
  ) +
  theme_minimal(base_size = 14)

print(plot2_weighted_disruption)
ggsave("plots/02_top50_weighted_disruption.png", plot2_weighted_disruption, width = 12, height = 9, dpi = 300)
cat("  [SAVE] Plot saved to plots/02_top50_weighted_disruption.png\n")


# Plot 3: Top 50 "Overachievers" by Disruption Rate Difference
cat("\n[PLOT] Generating and saving Plot 3: Top 50 Overachievers...\n")
top50_overachievers <- res[!is.na(YPC_Diff)][order(-Disruption_Rate_Diff)][1:50]

plot3_overachievers <- ggplot(top50_overachievers, aes(x = YPC_Diff, y = Disruption_Rate_Diff, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 10) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(
    title = "Top 50 Pass Rush Overachievers",
    subtitle = "Players whose weighted disruption rate is highest relative to their raw rate",
    x = "Yards Per Carry Allowed (On - Off Field)",
    y = "Disruption Rate Difference (Weighted - Raw)",
    caption = "2022 Season | Top 50 players by metric"
  ) +
  theme_minimal(base_size = 14)

print(plot3_overachievers)
ggsave("plots/03_top50_overachievers.png", plot3_overachievers, width = 12, height = 9, dpi = 300)
cat("  [SAVE] Plot saved to plots/03_top50_overachievers.png\n")


# Plot 4: Top 50 "Underachievers" by Disruption Rate Difference
cat("\n[PLOT] Generating and saving Plot 4: Top 50 Underachievers...\n")
top50_underachievers <- res[!is.na(YPC_Diff)][order(Disruption_Rate_Diff)][1:50]

plot4_underachievers <- ggplot(top50_underachievers, aes(x = YPC_Diff, y = Disruption_Rate_Diff, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 10) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(
    title = "Top 50 Pass Rush Underachievers",
    subtitle = "Players whose production came on the most predictable downs",
    x = "Yards Per Carry Allowed (On - Off Field)",
    y = "Disruption Rate Difference (Weighted - Raw)",
    caption = "2022 Season | Bottom 50 players by metric"
  ) +
  theme_minimal(base_size = 14)

print(plot4_underachievers)
ggsave("plots/04_top50_underachievers.png", plot4_underachievers, width = 12, height = 9, dpi = 300)
cat("  [SAVE] Plot saved to plots/04_top50_underachievers.png\n")


# Plot 5: Disruption Rate vs. Average Run Expectation
cat("\n[PLOT] Generating and saving Plot 5: Disruption Rate vs. Run Expectation...\n")
plot_data_context <- res[!is.na(YPC_Diff)]

# Fit linear model to find the trend
lm_fit <- lm(Raw_Disruption_Rate ~ Avg_Run, data = plot_data_context)
plot_data_context$fitted <- predict(lm_fit)
plot_data_context$residual <- plot_data_context$Raw_Disruption_Rate - plot_data_context$fitted

# Identify the top 15 players with the largest positive residuals (biggest overperformers)
top_performers <- plot_data_context[order(-residual)][1:15]

plot5_context <- ggplot(plot_data_context, aes(x = Avg_Run, y = Raw_Disruption_Rate)) +
  geom_point(aes(color = Pass_Rush_Snaps), alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black", linetype = "dashed", linewidth = 1.0) +
  geom_text_repel(data = top_performers, aes(label = Player), size = 4.0, max.overlaps = 15) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  scale_x_continuous(labels = scales::percent) +
  scale_y_continuous(labels = scales::percent) +
  labs(
    title = "Disruption Rate vs. Situational Context",
    subtitle = "Players above the line generate more pressure than expected for their role",
    x = "Average Pre-Snap Run Probability Faced",
    y = "Raw Disruption Rate"
  ) +
  theme_minimal(base_size = 14)

print(plot5_context)
ggsave("plots/05_disruption_vs_context.png", plot5_context, width = 12, height = 9, dpi = 300)
cat("  [SAVE] Plot saved to plots/05_disruption_vs_context.png\n")

# ───────────────────────────────────────────────────────────────────────────────
# 9) ADVANCED PASS RUSHER ANALYSIS & VISUALIZATION
# ───────────────────────────────────────────────────────────────────────────────

# This function encapsulates all of your custom evaluation logic
run_advanced_pass_rusher_analysis <- function(analysis_data, season_year = 2022) {
  
  cat(paste0("\n=== [ADV_EVAL] Running Advanced Pass Rusher Analysis for Season ", season_year, " ===\n"))
  
  # --- Step 1: Prepare the Data ---
  # The input `analysis_data` is the master `analysis_df` from the main script.
  data_season <- copy(analysis_data)
  
  # --- Step 2: Calculate Surprisal ---
  epsilon <- 1e-10
  # Using the robust `is_dropback` as the ground truth for intent
  data_season[, surprisal := -log(fifelse(is_dropback == 1, 
                                          pmin(pmax(gam_prediction, epsilon), 1 - epsilon), 
                                          1 - pmin(pmax(gam_prediction, epsilon), 1 - epsilon)))]
  
  # --- Step 3: Calculate Disruption Metrics ---
  cat("  [ADV_EVAL] Calculating weighted and raw disruption metrics...\n")
  
  dropback_snaps <- data_season[is_dropback == 1]
  dropback_snaps[, defense_players := str_trim(gsub(";+$", "", defense_players))]
  def_players_long <- dropback_snaps[!is.na(defense_players) & defense_players != "", .(
    gsis_id = unlist(strsplit(defense_players, ";")),
    surprisal = rep(surprisal, lengths(strsplit(defense_players, ";"))),
    gam_prediction = rep(gam_prediction, lengths(strsplit(defense_players, ";")))
  ), by = .(old_game_id, play_id)]
  
  player_surprisal_exposure <- def_players_long[, .(weighted_pass_rush_snaps = sum(surprisal, na.rm = TRUE)), by = gsis_id]
  
  # Note: nflverse PBP does not have half_sack_player_id, so this logic is simplified
  sacks_full <- dropback_snaps[sack == 1 & !is.na(sack_player_id), .(gsis_id = sack_player_id, weight = 1, surprisal)]
  sacks_weighted <- sacks_full[, .(weighted_sacks = sum(weight * surprisal, na.rm = TRUE)), by = gsis_id]
  sacks_raw <- sacks_full[, .(raw_sacks = .N), by = gsis_id]
  
  qb_hits <- dropback_snaps[qb_hit == 1 & !is.na(qb_hit_1_player_id), .(gsis_id = qb_hit_1_player_id, surprisal)]
  qb_hits_weighted <- qb_hits[, .(weighted_qb_hits = sum(surprisal, na.rm = TRUE)), by = gsis_id]
  qb_hits_raw <- qb_hits[, .(raw_qb_hits = .N), by = gsis_id]
  
  player_snap_counts <- def_players_long[, .(raw_pass_rush_snaps = .N), by = gsis_id]
  
  # --- Step 4: Combine into a Summary Table ---
  cat("  [ADV_EVAL] Combining all metrics into a summary table...\n")
  summary_list <- list(player_surprisal_exposure, sacks_weighted, qb_hits_weighted, player_snap_counts, sacks_raw, qb_hits_raw)
  disruption_summary <- Reduce(function(x, y) merge(x, y, by = "gsis_id", all = TRUE), summary_list)
  
  for(col in names(disruption_summary)) {
    if(is.numeric(disruption_summary[[col]])) set(disruption_summary, which(is.na(disruption_summary[[col]])), col, 0)
  }
  
  disruption_summary[, disruption_rate := (weighted_sacks + weighted_qb_hits) / weighted_pass_rush_snaps]
  disruption_summary[, raw_disruption_rate := (raw_sacks + raw_qb_hits) / raw_pass_rush_snaps]
  # Your original logic was raw - weighted. Let's keep that.
  disruption_summary[, disruption_rate_diff := raw_disruption_rate - disruption_rate]
  
  # --- Step 5: Calculate Contextual Metrics ---
  cat("  [ADV_EVAL] Calculating contextual metrics (Avg Run Prob, YPC Diff)...\n")
  
  avg_run_prob_by_player <- def_players_long[, .(avg_run_prob = mean(1 - gam_prediction, na.rm = TRUE)), by = gsis_id]
  disruption_summary <- merge(disruption_summary, avg_run_prob_by_player, by = "gsis_id", all.x = TRUE)
  
  # YPC Diff Calculation (preserved from your logic)
  run_plays <- data_season[is_dropback == 0 & !is.na(defense_players) & defense_players != ""]
  # (Your complex and correct YPC_Diff logic is assumed to be here)
  
  # --- Step 6: Finalize and Filter the Leaderboard ---
  rosters <- nflreadr::load_rosters(season_year)
  setDT(rosters)
  disruption_summary <- merge(disruption_summary, rosters[, .(gsis_id, full_name, position, team)], by = "gsis_id", all.x = TRUE)
  
  pass_rusher_positions <- c("DE", "DT", "EDGE", "OLB", "ILB", "LB", "NT", "DL")
  disruption_summary <- disruption_summary[position %in% pass_rusher_positions & raw_pass_rush_snaps >= 100]
  
  setorder(disruption_summary, -disruption_rate)
  
  # Return the full summary table for plotting, formatted as you requested
  return(disruption_summary[, .(
    Player = full_name, Team = team, Position = position,
    Weighted_Sacks = round(weighted_sacks, 3),
    Weighted_QB_Hits = round(weighted_qb_hits, 3),
    Pass_Rush_Snaps = raw_pass_rush_snaps,
    Weighted_Pass_Rush_Snaps = round(weighted_pass_rush_snaps, 3),
    Disruption_Rate = round(disruption_rate, 4),
    Raw_Disruption_Rate = round(raw_disruption_rate, 4),
    Disruption_Rate_Diff = round(disruption_rate_diff, 4),
    YPC_Diff = NA_real_, # Placeholder, as your YPC code is separate
    Avg_Run = round(avg_run_prob, 5)
  )])
}


# --- SCRIPT EXECUTION ---

# Create a directory for plots
if (!dir.exists("plots")) dir.create("plots")

# Run the evaluation to get the final results table
res <- run_advanced_pass_rusher_analysis(analysis_df, season_year = 2022)

cat("\n--- Top 30 Pass Rushers by Weighted Disruption Rate ---\n")
print(head(res, 30))

# --- PLOTTING SECTION (Adapted to save each plot) ---

# Plot 1: Top 50 by Raw Disruption Rate
cat("\n[PLOT] Generating and saving Plot 1...\n")
top50_raw <- res[!is.na(YPC_Diff)][order(-Raw_Disruption_Rate)][1:50]
plot1 <- ggplot(top50_raw, aes(x = YPC_Diff, y = Raw_Disruption_Rate, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 4.5, max.overlaps = 100) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(title = "Top 50 by Raw Disruption Rate", x = "YPC Allowed (On-Off)", y = "Disruption Rate") +
  theme_minimal(base_size = 14)
print(plot1)
ggsave("plots/01_top50_raw_disruption.png", plot1, width = 12, height = 9, dpi = 300)

# (All other plots follow the same pattern)

# ───────────────────────────────────────────────────────────────────────────────
# NEW SECTION: PRINCIPAL COMPONENT ANALYSIS (PCA) OF MODEL FEATURES
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [PCA_ANALYSIS] Visualizing Feature Structure with PCA ===\n")

if (exists("features_m3") && exists("keeper_linear_features") && exists("keeper_smooth_features")) {
  
  # --- Step 1: Prepare Data for PCA ---
  cat("  [PCA] Preparing data for PCA (using the 34 significant features)...\n")
  
  # Combine the lists of keeper features
  pca_features <- c(keeper_linear_features, keeper_smooth_features)
  
  # Subset the main data table to only these features
  # The `..` tells data.table to use the variable `pca_features` to select columns
  pca_data <- features_m3[, ..pca_features]
  
  # --- Step 2: Run the PCA ---
  # The `prcomp` function is the standard for PCA in R.
  # `scale. = TRUE` is ESSENTIAL. It standardizes all features to have the same scale.
  cat("  [PCA] Running Principal Component Analysis...\n")
  pca_result <- prcomp(pca_data, scale. = TRUE)
  
  # --- Step 3: Extract Results for Plotting ---
  # The 'x' component contains the new coordinates (scores) for each play (PC1, PC2, etc.)
  pca_scores <- as.data.table(pca_result$x)
  
  # The 'rotation' component contains the loadings, which are the directions of the original features.
  pca_loadings <- as.data.table(pca_result$rotation, keep.rownames = "Feature")
  
  # Add the true outcome to the scores data for coloring the plot
  pca_scores[, is_dropback := factor(features_m3$is_dropback, labels = c("Designed Run", "Dropback"))]
  
  # Get the variance explained by the top two components
  variance_explained <- summary(pca_result)$importance[2, 1:2]
  cat(sprintf("  [PCA] PC1 explains %.1f%% of variance.\n", variance_explained[1] * 100))
  cat(sprintf("  [PCA] PC2 explains %.1f%% of variance.\n", variance_explained[2] * 100))
  
  # --- Step 4: Create the PCA Biplot ---
  cat("  [PCA] Generating PCA biplot...\n")
  
  # This scaling factor makes the feature arrows visible on the same plot as the scores
  arrow_scale <- max(abs(range(pca_scores[, .(PC1, PC2)]))) * 0.4
  
  pca_plot <- ggplot() +
    # Plot the individual plays (scores), colored by their true outcome
    geom_point(data = pca_scores, aes(x = PC1, y = PC2, color = is_dropback), alpha = 0.25) +
    
    # Add the feature vectors (loadings) as arrows from the center
    geom_segment(data = pca_loadings, 
                 aes(x = 0, y = 0, xend = PC1 * arrow_scale, yend = PC2 * arrow_scale),
                 arrow = arrow(length = unit(0.2, "cm")), color = "black") +
    
    # Add labels to the feature arrows
    ggrepel::geom_text_repel(data = pca_loadings, 
                             aes(x = PC1 * arrow_scale, y = PC2 * arrow_scale, label = Feature),
                             color = "black", size = 3.0, point.padding = 0.5) +
    
    scale_color_manual(values = c("Designed Run" = "#D55E00", "Dropback" = "#0072B2")) +
    labs(
      title = "Principal Component Analysis of Significant GAM Features",
      subtitle = "How pre-snap features separate Designed Runs from Dropbacks",
      x = sprintf("Principal Component 1 (%.1f%% Variance)", variance_explained[1] * 100),
      y = sprintf("Principal Component 2 (%.1f%% Variance)", variance_explained[2] * 100),
      color = "Actual Play Intent"
    ) +
    theme_minimal(base_size = 14) +
    coord_equal() # Ensures the axes are scaled equally, preserving the true angles of the features
  
  print(pca_plot)
  ggsave("plots/pca_features_biplot.png", pca_plot, width = 14, height = 11, dpi = 300)
  cat("  [SAVE] Saved PCA plot to plots/pca_features_biplot.png\n")
  
} else {
  cat("  [PCA_ANALYSIS] Skipped: `features_m3` or keeper lists not found.\n")
  
}

# ───────────────────────────────────────────────────────────────────────────────
# NEW SECTION: PRINCIPAL COMPONENT ANALYSIS (PCA) OF MODEL FEATURES
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [PCA_ANALYSIS] Visualizing Feature Structure with PCA ===\n")

if (exists("features_m3") && exists("keeper_linear_features") && exists("keeper_smooth_features")) {
  
  # --- Step 1: Prepare Data for PCA ---
  cat("  [PCA] Preparing data for PCA (using the 34 significant features)...\n")
  
  # Combine the lists of keeper features
  pca_features <- c(keeper_linear_features, keeper_smooth_features)
  
  # Subset the main data table to only these features
  # The `..` tells data.table to use the variable `pca_features` to select columns
  pca_data <- features_m3[, ..pca_features]
  
  # --- Step 2: Run the PCA ---
  # The `prcomp` function is the standard for PCA in R.
  # `scale. = TRUE` is ESSENTIAL. It standardizes all features to have the same scale.
  cat("  [PCA] Running Principal Component Analysis...\n")
  pca_result <- prcomp(pca_data, scale. = TRUE)
  
  # --- Step 3: Extract Results for Plotting ---
  # The 'x' component contains the new coordinates (scores) for each play (PC1, PC2, etc.)
  pca_scores <- as.data.table(pca_result$x)
  
  # The 'rotation' component contains the loadings, which are the directions of the original features.
  pca_loadings <- as.data.table(pca_result$rotation, keep.rownames = "Feature")
  
  # Add the true outcome to the scores data for coloring the plot
  pca_scores[, is_dropback := factor(features_m3$is_dropback, labels = c("Designed Run", "Dropback"))]
  
  # Get the variance explained by the top two components
  variance_explained <- summary(pca_result)$importance[2, 1:2]
  cat(sprintf("  [PCA] PC1 explains %.1f%% of variance.\n", variance_explained[1] * 100))
  cat(sprintf("  [PCA] PC2 explains %.1f%% of variance.\n", variance_explained[2] * 100))
  
  # --- Step 4: Create the PCA Biplot ---
  cat("  [PCA] Generating PCA biplot...\n")
  
  # This scaling factor makes the feature arrows visible on the same plot as the scores
  arrow_scale <- max(abs(range(pca_scores[, .(PC1, PC2)]))) * 0.4
  
  pca_plot <- ggplot() +
    # Plot the individual plays (scores), colored by their true outcome
    geom_point(data = pca_scores, aes(x = PC1, y = PC2, color = is_dropback), alpha = 0.25) +
    
    # Add the feature vectors (loadings) as arrows from the center
    geom_segment(data = pca_loadings, 
                 aes(x = 0, y = 0, xend = PC1 * arrow_scale, yend = PC2 * arrow_scale),
                 arrow = arrow(length = unit(0.2, "cm")), color = "black") +
    
    # Add labels to the feature arrows
    ggrepel::geom_text_repel(data = pca_loadings, 
                             aes(x = PC1 * arrow_scale, y = PC2 * arrow_scale, label = Feature),
                             color = "black", size = 3.0, point.padding = 0.5) +
    
    scale_color_manual(values = c("Designed Run" = "#D55E00", "Dropback" = "#0072B2")) +
    labs(
      title = "Principal Component Analysis of Significant GAM Features",
      subtitle = "How pre-snap features separate Designed Runs from Dropbacks",
      x = sprintf("Principal Component 1 (%.1f%% Variance)", variance_explained[1] * 100),
      y = sprintf("Principal Component 2 (%.1f%% Variance)", variance_explained[2] * 100),
      color = "Actual Play Intent"
    ) +
    theme_minimal(base_size = 14) +
    coord_equal() # Ensures the axes are scaled equally, preserving the true angles of the features
  
  print(pca_plot)
  ggsave("plots/pca_features_biplot.png", pca_plot, width = 14, height = 11, dpi = 300)
  cat("  [SAVE] Saved PCA plot to plots/pca_features_biplot.png\n")
  
} else {
  cat("  [PCA_ANALYSIS] Skipped: `features_m3` or keeper lists not found.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# NEW SECTION: PRINCIPAL COMPONENT ANALYSIS (PCA) OF TOP 10 FEATURES
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [PCA_ANALYSIS] Visualizing Feature Structure with PCA (Top 10 Features) ===\n")

if (exists("features_m3") && exists("keeper_linear_features") && exists("keeper_smooth_features")) {
  
  # --- Step 1 & 2: Prepare and Run PCA (on all 34 significant features) ---
  cat("  [PCA] Preparing data and running PCA...\n")
  pca_features <- c(keeper_linear_features, keeper_smooth_features)
  pca_data <- features_m3[, ..pca_features]
  pca_result <- prcomp(pca_data, scale. = TRUE)
  
  # --- Step 3: Extract Results and Select Top 10 Features ---
  pca_scores <- as.data.table(pca_result$x)
  pca_loadings <- as.data.table(pca_result$rotation, keep.rownames = "Feature")
  pca_scores[, is_dropback := factor(features_m3$is_dropback, labels = c("Designed Run", "Dropback"))]
  variance_explained <- summary(pca_result)$importance[2, 1:2]
  
  # --- THE FIX: Select only the top 10 features based on their impact on the plot ---
  cat("  [PCA] Identifying the top 10 most influential features for visualization...\n")
  
  # Calculate a combined importance score (vector length in the PC1-PC2 plane)
  pca_loadings[, importance_in_plot := sqrt(PC1^2 + PC2^2)]
  
  # Order by this new score and take the top 10
  setorder(pca_loadings, -importance_in_plot)
  top_10_loadings <- head(pca_loadings, 10)
  
  # --- Step 4: Create the Cleaned-Up PCA Biplot ---
  cat("  [PCA] Generating PCA biplot for TOP 10 features...\n")
  
  arrow_scale <- 10
  
  pca_top10_plot <- ggplot() +
    # Plot all the individual plays (scores) in the background
    geom_point(data = pca_scores, aes(x = PC1, y = PC2, color = is_dropback), alpha = 0.2) +
    
    # Add arrows and labels for ONLY the top 10 features
    geom_segment(data = top_10_loadings, 
                 aes(x = 0, y = 0, xend = PC1 * arrow_scale, yend = PC2 * arrow_scale),
                 arrow = arrow(length = unit(0.2, "cm")), color = "black") +
    
    ggrepel::geom_text_repel(data = top_10_loadings, 
                             aes(x = PC1 * arrow_scale, y = PC2 * arrow_scale, label = Feature),
                             color = "black", size = 4.0, # Made text slightly larger
                             point.padding = 0.6,
                             max.overlaps = Inf) +
    
    scale_color_manual(values = c("Designed Run" = "#D55E00", "Dropback" = "#0072B2")) +
    labs(
      title = "PCA of Top 10 Significant GAM Features",
      subtitle = "How the most important pre-snap features separate Runs from Dropbacks",
      x = sprintf("Principal Component 1 (%.1f%% Variance)", variance_explained[1] * 100),
      y = sprintf("Principal Component 2 (%.1f%% Variance)", variance_explained[2] * 100),
      color = "Actual Play Intent"
    ) +
    theme_minimal(base_size = 14) +
    coord_equal()
  
  print(pca_top10_plot)
  ggsave("plots/pca_top10_features_biplot.png", pca_top10_plot, width = 14, height = 11, dpi = 300)
  cat("  [SAVE] Saved Top 10 PCA plot to plots/pca_top10_features_biplot.png\n")
  
} else {
  cat("  [PCA_ANALYSIS] Skipped: `features_m3` or keeper lists not found.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 9) FINAL EVALUATION: DISRUPTION RATE & RISERS/FALLERS ANALYSIS
# ───────────────────────────────────────────────────────────────────────────────

cat("\n=== [FINAL_EVAL] CALCULATING DISRUPTION RATE & FINDING RISERS/FALLERS ===\n")

if (exists("analysis_df")) {
  
  # --- Step 1: Prepare the Master Snap-Level Data ---
  cat("  [FINAL_EVAL] Preparing master data table for analysis...\n")
  
  # We need the detailed player-level outcomes from player_play.csv
  player_play_df <- fread("player_play.csv")
  
  # We only need the prediction from our main analysis table
  play_level_predictions <- analysis_df[, .(gameId, play_id, is_dropback, gam_prediction)]
  setnames(play_level_predictions, "play_id", "playId")
  
  # Start with all pass rush snaps for actual dropback plays
  pass_rush_snaps <- player_play_df[
    wasInitialPassRusher == TRUE & 
      playId %in% play_level_predictions[is_dropback == 1, playId]
  ]
  
  # Merge in our model's predictions
  pass_rush_snaps <- merge(pass_rush_snaps, play_level_predictions, by = c("gameId", "playId"))
  
  # --- Step 2: Calculate Surprisal for Every Snap ---
  epsilon <- 1e-10
  pass_rush_snaps[, surprisal := -log(fifelse(is_dropback == 1, 
                                              pmin(pmax(gam_prediction, epsilon), 1 - epsilon), 
                                              1 - pmin(pmax(gam_prediction, epsilon), 1 - epsilon)))]
  
  # --- Step 3: Calculate Weighted & Raw Disruptions for Every Snap ---
  # We create two new columns: one for the raw value (1, 0.5, 0) and one for the weighted value.
  pass_rush_snaps[, raw_disruption := fifelse(sackYardsAsDefense < 0, 1.0, # Full Sack
                                              fifelse(halfSackYardsAsDefense < 0, 0.5, # Half Sack
                                                      fifelse(quarterbackHit == 1, 1.0, # QB Hit
                                                              0.0)))]
  
  pass_rush_snaps[, weighted_disruption := raw_disruption * surprisal]
  
  # --- Step 4: Aggregate to the Player Level ---
  cat("  [FINAL_EVAL] Aggregating performance to the player level...\n")
  
  player_summary <- pass_rush_snaps[, .(
    total_weighted_disruption = sum(weighted_disruption, na.rm = TRUE),
    total_raw_disruption = sum(raw_disruption, na.rm = TRUE),
    weighted_opportunity = sum(surprisal, na.rm = TRUE), # This is the denominator from the image
    raw_snaps = .N
  ), by = .(nflId)]
  
  # --- Step 5: Small Sample Size Adjustment (Empirical Bayes Smoothing) ---
  cat("  [FINAL_EVAL] Applying smoothing to adjust for small sample sizes...\n")
  
  # Calculate the league average weighted and raw rates
  league_avg_weighted_rate <- sum(player_summary$total_weighted_disruption) / sum(player_summary$weighted_opportunity)
  league_avg_raw_rate <- sum(player_summary$total_raw_disruption) / sum(player_summary$raw_snaps)
  
  # 'k' is our smoothing factor. 100 is a robust, standard choice.
  # It's like adding 100 "league average" snaps to every player's totals.
  k_weighted <- 100 
  k_raw <- 100
  
  player_summary[, smoothed_weighted_rate := (total_weighted_disruption + k_weighted * league_avg_weighted_rate) / (weighted_opportunity + k_weighted)]
  player_summary[, smoothed_raw_rate := (total_raw_disruption + k_raw * league_avg_raw_rate) / (raw_snaps + k_raw)]
  
  # --- Step 6: Calculate Risers & Fallers ---
  player_summary[, rate_difference := smoothed_weighted_rate - smoothed_raw_rate]
  
  # --- Step 7: Finalize the Leaderboard ---
  cat("  [FINAL_EVAL] Finalizing leaderboard and identifying archetypes...\n")
  
  bdb_roster <- bdb_data$players[, .(nflId, displayName, position)]
  nflreadr_roster <- nflreadr::load_rosters(2022)
  setDT(nflreadr_roster)
  roster_teams <- unique(nflreadr_roster[, .(full_name, team)], by = "full_name")
  
  leaderboard <- merge(player_summary, bdb_roster, by = "nflId")
  setnames(leaderboard, "displayName", "full_name")
  leaderboard <- merge(leaderboard, roster_teams, by = "full_name", all.x = TRUE)
  
  leaderboard <- leaderboard[raw_snaps >= 150] # Filter for a meaningful number of snaps
  
  # --- Step 8: Generate and Print the Final Lists ---
  
  # Top 5 Overall (by the final, smoothed Disruption Rate)
  setorder(leaderboard, -smoothed_weighted_rate)
  top_5_overall <- head(leaderboard, 5)[, .(
    Rank = .I, Player = full_name, Team = team, Position = position,
    `Disruption Rate` = round(smoothed_weighted_rate, 4),
    `Raw Rate` = round(smoothed_raw_rate, 4),
    `Rush Snaps` = raw_snaps
  )]
  
  # Top 5 Risers (biggest positive difference)
  setorder(leaderboard, -rate_difference)
  top_5_risers <- head(leaderboard, 5)[, .(
    Rank = .I, Player = full_name, Team = team,
    `Weighted Rate` = round(smoothed_weighted_rate, 4),
    `Raw Rate` = round(smoothed_raw_rate, 4),
    `Improvement` = paste0("+", round(rate_difference, 4))
  )]
  
  # Top 5 Fallers (biggest negative difference)
  setorder(leaderboard, rate_difference)
  top_5_fallers <- head(leaderboard, 5)[, .(
    Rank = .I, Player = full_name, Team = team,
    `Weighted Rate` = round(smoothed_weighted_rate, 4),
    `Raw Rate` = round(smoothed_raw_rate, 4),
    `Drop` = round(rate_difference, 4)
  )]
  
  cat("\n\n--- Top 5 Pass Rushers Overall (by Smoothed Weighted Disruption Rate) ---\n")
  print(top_5_overall)
  
  cat("\n\n--- Top 5 Risers (Biggest Improvement from Raw to Weighted Rate) ---\n")
  cat("These players excel at creating pressure on unexpected dropbacks (e.g., play-action).\n")
  print(top_5_risers)
  
  cat("\n\n--- Top 5 Fallers (Biggest Drop from Raw to Weighted Rate) ---\n")
  cat("These players' production is concentrated on obvious, 'unsurprising' passing downs.\n")
  print(top_5_fallers)
  
} else {
  cat("  [FINAL_EVAL] Skipped: `analysis_df` object not found.\n")
}

library(png)

# ───────────────────────────────────────────────────────────────────────────────
# 10) FINAL VISUALIZATION: TOP 5 PLAYER LEADERBOARD (BASE R)
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [VISUALIZATION] CREATING TOP 5 LEADERBOARD WITH BASE R GRAPHICS ===\n")

if (exists("leaderboard")) {
  
  # --- Step 1: Prepare the Data ---
  suppressPackageStartupMessages({
    library(grid) # The core graphics engine
    library(png)  # For reading PNG images from URLs
  })
  
  # Get the full 2022 roster to find headshot URLs
  rosters_2022_with_headshots <- nflreadr::load_rosters(2022)
  setDT(rosters_2022_with_headshots)
  
  # Isolate the top 5 players and merge headshot URLs
  top_5_data <- head(leaderboard, 5)
  top_5_data <- merge(top_5_data,
                      rosters_2022_with_headshots[, .(full_name, headshot_url)],
                      by = "full_name", all.x = TRUE)
  setorder(top_5_data, Rank)
  
  # --- Step 2: Create the Red-to-Green Color Scale ---
  # Create a function that maps a value from 0 to 1 to a color
  color_palette <- colorRampPalette(c("#e65c5c", "white", "#63be7b"))(100)
  
  # Normalize the disruption rates to be between 0 and 1
  rates <- top_5_data$smoothed_weighted_rate
  norm_rates <- (rates - min(rates)) / (max(rates) - min(rates))
  cell_colors <- color_palette[ceiling(norm_rates * 99) + 1]
  
  # --- Step 3: Set up the PNG file and Grid Layout ---
  cat("  [VISUALIZATION] Saving final leaderboard to PNG file...\n")
  
  png("plots/top_5_leaderboard_base.png", width = 800, height = 500, res = 100)
  
  # Clear the canvas and set up a grid: 6 rows (title + 5 players), 5 columns
  grid.newpage()
  layout <- grid.layout(
    nrow = 6, 
    ncol = 5,
    widths = unit(c(0.5, 1, 3, 1.5, 1.5), "null"), # Relative column widths
    heights = unit(c(1, 1, 1, 1, 1, 1), "null")   # Relative row heights
  )
  pushViewport(viewport(layout = layout))
  
  # --- Step 4: Draw the Header and Title ---
  # Main Title
  grid.text("Top 5 Pass Rushers of 2022", vp = viewport(layout.pos.row = 1, layout.pos.col = 1:5),
            gp = gpar(fontsize = 20, fontface = "bold"))
  
  # Column Headers
  headers <- c("Rank", "", "Player", "Team/Pos", "Disruption Rate")
  for (i in seq_along(headers)) {
    grid.text(headers[i], vp = viewport(layout.pos.row = 2, layout.pos.col = i),
              gp = gpar(fontsize = 10, fontface = "bold"))
  }
  
  # --- Step 5: Loop Through Players and Draw Each Row ---
  for (i in 1:nrow(top_5_data)) {
    player <- top_5_data[i, ]
    row_pos <- i + 2 # Start drawing from the 3rd row of the grid
    
    # Col 1: Rank
    grid.text(player$Rank, vp = viewport(layout.pos.row = row_pos, layout.pos.col = 1))
    
    # Col 2: Headshot
    if (!is.na(player$headshot_url)) {
      tryCatch({
        img <- readPNG(RCurl::getURLContent(player$headshot_url))
        grid.raster(img, vp = viewport(layout.pos.row = row_pos, layout.pos.col = 2))
      }, error = function(e) {
        grid.text("No Img", vp = viewport(layout.pos.row = row_pos, layout.pos.col = 2))
      })
    }
    
    # Col 3: Player Name
    grid.text(player$full_name, vp = viewport(layout.pos.row = row_pos, layout.pos.col = 3), just = "left")
    
    # Col 4: Team & Position
    grid.text(paste(player$team, player$position, sep = " / "), 
              vp = viewport(layout.pos.row = row_pos, layout.pos.col = 4))
    
    # Col 5: Disruption Rate (Cell + Text)
    # Draw the colored background rectangle
    grid.rect(vp = viewport(layout.pos.row = row_pos, layout.pos.col = 5),
              gp = gpar(fill = cell_colors[i], col = "grey"))
    # Draw the text on top
    grid.text(sprintf("%.4f", player$smoothed_weighted_rate), 
              vp = viewport(layout.pos.row = row_pos, layout.pos.col = 5))
  }
  
  # --- Step 6: Close the PNG device to save the file ---
  dev.off()
  
  cat("  [SAVE] Final leaderboard saved to 'plots/top_5_leaderboard_base.png'\n")
  
} else {
  cat("  [VISUALIZATION] Skipped: `leaderboard` object not found.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 11) VISUALIZATION: TOP 5 RISERS & FALLERS
# ───────────────────────────────────────────────────────────────────────────────
cat("\n=== [VISUALIZATION] CREATING RISERS & FALLERS LEADERBOARDS ===\n")

if (exists("leaderboard")) {
  
  # --- Step 1: Prepare Data (Headshots) ---
  # We can reuse the roster data from the previous section
  if (!exists("rosters_2022_with_headshots")) {
    rosters_2022_with_headshots <- nflreadr::load_rosters(2022)
    setDT(rosters_2022_with_headshots)
  }
  
  # --- Step 2: Create the "Risers" Table ---
  cat("  [VISUALIZATION] Identifying and building the Top 5 Risers table...\n")
  
  # Sort by the rate_difference in descending order to find the biggest gainers
  setorder(leaderboard, -rate_difference)
  top_5_risers <- head(leaderboard, 5)
  top_5_risers[, Rank := .I] # Add a rank column
  
  # Merge headshot URLs
  top_5_risers_with_headshots <- merge(
    top_5_risers,
    rosters_2022_with_headshots[, .(full_name, headshot_url)],
    by = "full_name", all.x = TRUE
  )
  
  # Select and arrange columns
  risers_table_data <- top_5_risers_with_headshots %>%
    select(Rank, headshot_url, full_name, team, smoothed_weighted_rate, smoothed_raw_rate, rate_difference) %>%
    arrange(Rank)
  
  # Build the gt table
  risers_table <- risers_table_data %>%
    gt() %>%
    tab_header(
      title = md("**Top 5 Risers**"),
      subtitle = "Players with the biggest improvement from Raw to Weighted Disruption Rate"
    ) %>%
    gt_img_rows(columns = headshot_url, height = 45) %>%
    # Color the "Improvement" column with a green scale
    data_color(
      columns = rate_difference,
      colors = scales::col_numeric(
        palette = c("white", "#63be7b"), # White -> Green
        domain = range(risers_table_data$rate_difference)
      )
    ) %>%
    fmt_number(columns = where(is.numeric), decimals = 4) %>%
    # Use fmt_sprintf to add a '+' sign to the improvement
    fmt_sprintf(columns = rate_difference, fmt = "+%.4f") %>%
    cols_label(
      headshot_url = "", full_name = "Player", team = "Team",
      smoothed_weighted_rate = "Weighted Rate",
      smoothed_raw_rate = "Raw Rate",
      rate_difference = "Improvement"
    ) %>%
    tab_source_note(md("Metric rewards pressure on unexpected dropbacks (e.g., play-action).")) %>%
    gt_theme_538()
  
  # Save and print
  gtsave(risers_table, "plots/top_5_risers.png", vwidth = 900, vheight = 650)
  cat("  [SAVE] Risers leaderboard saved to 'plots/top_5_risers.png'\n")
  print(risers_table)
  
  # --- Step 3: Create the "Fallers" Table ---
  cat("\n  [VISUALIZATION] Identifying and building the Top 5 Fallers table...\n")
  
  # Sort by the rate_difference in ASCENDING order to find the biggest drops
  setorder(leaderboard, rate_difference)
  top_5_fallers <- head(leaderboard, 5)
  top_5_fallers[, Rank := .I]
  
  top_5_fallers_with_headshots <- merge(
    top_5_fallers,
    rosters_2022_with_headshots[, .(full_name, headshot_url)],
    by = "full_name", all.x = TRUE
  )
  
  fallers_table_data <- top_5_fallers_with_headshots %>%
    select(Rank, headshot_url, full_name, team, smoothed_weighted_rate, smoothed_raw_rate, rate_difference) %>%
    arrange(Rank)
  
  # Build the gt table
  fallers_table <- fallers_table_data %>%
    gt() %>%
    tab_header(
      title = md("**Top 5 Fallers**"),
      subtitle = "Players with the biggest drop from Raw to Weighted Disruption Rate"
    ) %>%
    gt_img_rows(columns = headshot_url, height = 45) %>%
    # Color the "Drop" column with a red scale
    data_color(
      columns = rate_difference,
      colors = scales::col_numeric(
        palette = c("#e65c5c", "white"), # Red -> White
        domain = range(fallers_table_data$rate_difference)
      )
    ) %>%
    fmt_number(columns = where(is.numeric), decimals = 4) %>%
    cols_label(
      headshot_url = "", full_name = "Player", team = "Team",
      smoothed_weighted_rate = "Weighted Rate",
      smoothed_raw_rate = "Raw Rate",
      rate_difference = "Drop"
    ) %>%
    tab_source_note(md("Metric penalizes production on obvious, 'unsurprising' passing downs.")) %>%
    gt_theme_538()
  
  # Save and print
  gtsave(fallers_table, "plots/top_5_fallers.png", vwidth = 900, vheight = 650)
  cat("  [SAVE] Fallers leaderboard saved to 'plots/top_5_fallers.png'\n")
  print(fallers_table)
  
} else {
  cat("  [VISUALIZATION] Skipped: `leaderboard` object not found.\n")
}

# ───────────────────────────────────────────────────────────────────────────────
# 9) FINAL EVALUATION: ADVANCED PASS RUSHER METRICS & VISUALIZATION
# ───────────────────────────────────────────────────────────────────────────────

# This is the definitive, corrected, and adapted version of your evaluation function.
evaluate_pass_rushers <- function(analysis_data, pbp_data, season_year) {
  
  cat(paste0("\n=== [ADV_EVAL] Running Advanced Analysis for Season ", season_year, " ===\n"))
  
  # --- Step 1: Prepare the Data ---
  data_season <- copy(analysis_data)
  
  # --- Step 2: Calculate Surprisal ---
  epsilon <- 1e-10
  data_season[, surprisal := -log(fifelse(is_dropback == 1, 
                                          pmin(pmax(gam_prediction, epsilon), 1 - epsilon), 
                                          1 - pmin(pmax(gam_prediction, epsilon), 1 - epsilon)))]
  
  # --- Step 3: Calculate Disruption Metrics ---
  cat("  [ADV_EVAL] Calculating weighted and raw disruption metrics...\n")
  
  dropback_snaps <- data_season[is_dropback == 1]
  dropback_snaps[, defense_players := str_trim(gsub(";+$", "", defense_players))]
  def_players_long <- dropback_snaps[!is.na(defense_players) & defense_players != "", .(
    gsis_id = unlist(strsplit(defense_players, ";")),
    surprisal = rep(surprisal, lengths(strsplit(defense_players, ";"))),
    gam_prediction = rep(gam_prediction, lengths(strsplit(defense_players, ";")))
  ), by = .(old_game_id, play_id)]
  
  player_surprisal_exposure <- def_players_long[, .(weighted_pass_rush_snaps = sum(surprisal, na.rm = TRUE)), by = gsis_id]
  
  sacks_full <- dropback_snaps[sack == 1 & !is.na(sack_player_id), .(gsis_id = sack_player_id, weight = 1.0, surprisal)]
  sacks_weighted <- sacks_full[, .(weighted_sacks = sum(weight * surprisal, na.rm = TRUE)), by = gsis_id]
  sacks_raw <- sacks_full[, .(raw_sacks = sum(weight)), by = gsis_id]
  
  qb_hits <- dropback_snaps[qb_hit == 1 & !is.na(qb_hit_1_player_id), .(gsis_id = qb_hit_1_player_id, surprisal)]
  qb_hits_weighted <- qb_hits[, .(weighted_qb_hits = sum(surprisal, na.rm = TRUE)), by = gsis_id]
  qb_hits_raw <- qb_hits[, .(raw_qb_hits = .N), by = gsis_id]
  
  player_snap_counts <- def_players_long[, .(raw_pass_rush_snaps = .N), by = gsis_id]
  
  # --- Step 4: Combine into a Summary Table ---
  summary_list <- list(player_surprisal_exposure, sacks_weighted, qb_hits_weighted, player_snap_counts, sacks_raw, qb_hits_raw)
  disruption_summary <- Reduce(function(x, y) merge(x, y, by = "gsis_id", all = TRUE), summary_list)
  
  for(col in names(disruption_summary)) {
    if(is.numeric(disruption_summary[[col]])) set(disruption_summary, which(is.na(disruption_summary[[col]])), col, 0)
  }
  
  disruption_summary[, disruption_rate := (weighted_sacks + weighted_qb_hits) / weighted_pass_rush_snaps]
  disruption_summary[, raw_disruption_rate := (raw_sacks + raw_qb_hits) / raw_pass_rush_snaps]
  disruption_summary[, disruption_rate_diff := raw_disruption_rate - disruption_rate]
  
  # --- Step 5: Calculate Contextual Metrics ---
  cat("  [ADV_EVAL] Calculating contextual metrics (Avg Run Prob, YPC Diff)...\n")
  
  avg_run_prob_by_player <- def_players_long[, .(avg_run_prob = mean(1 - gam_prediction, na.rm = TRUE)), by = gsis_id]
  disruption_summary <- merge(disruption_summary, avg_run_prob_by_player, by = "gsis_id", all.x = TRUE)
  
  # --- YPC Diff Calculation (CORRECTED) ---
  # THE FIX: Use `data_season` (which is `analysis_df`) as the source for run plays.
  run_plays <- data_season[is_dropback == 0 & !is.na(defense_players) & defense_players != ""]
  
  # We still need `yards_gained`, which lives in `pbp_data`. Merge it in.
  run_plays <- merge(run_plays, pbp_data[, .(old_game_id, play_id, yards_gained)], by = c("old_game_id", "play_id"), all.x = TRUE)
  
  run_def_long <- run_plays[!is.na(yards_gained), .(gsis_id = unlist(strsplit(defense_players, ";"))), by = .(old_game_id, play_id, yards_gained)]
  
  all_players <- unique(def_players_long$gsis_id)
  run_play_ids <- unique(run_plays[, .(old_game_id, play_id, yards_gained)])
  player_on_plays <- run_def_long[, .(on_play = TRUE), by = .(gsis_id, old_game_id, play_id)]
  
  ypc_diff_dt <- rbindlist(lapply(all_players, function(player_id) {
    if (!player_id %in% player_on_plays$gsis_id) return(NULL)
    joined <- merge(run_play_ids, player_on_plays[gsis_id == player_id], by = c("old_game_id", "play_id"), all.x = TRUE)
    joined[is.na(on_play), on_play := FALSE]
    if (sum(joined$on_play) < 100) return(NULL) # Min 100 run snaps
    ypc_on <- joined[on_play == TRUE, mean(yards_gained, na.rm = TRUE)]
    ypc_off <- joined[on_play == FALSE, mean(yards_gained, na.rm = TRUE)]
    list(gsis_id = player_id, ypc_on = ypc_on, ypc_off = ypc_off, ypc_diff = ypc_on - ypc_off)
  }))
  
  if (nrow(ypc_diff_dt) > 0) {
    disruption_summary <- merge(disruption_summary, ypc_diff_dt[, .(gsis_id, ypc_diff)], by = "gsis_id", all.x = TRUE)
  } else {
    disruption_summary[, ypc_diff := NA_real_]
  }
  
  # --- Step 6: Finalize and Filter ---
  rosters <- nflreadr::load_rosters(season_year)
  setDT(rosters)
  disruption_summary <- merge(disruption_summary, rosters[, .(gsis_id, full_name, position, team)], by = "gsis_id", all.x = TRUE)
  
  pass_rusher_positions <- c("DE", "DT", "EDGE", "OLB", "ILB", "LB", "NT", "DL")
  disruption_summary <- disruption_summary[position %in% pass_rusher_positions & raw_pass_rush_snaps >= 100]
  
  setorder(disruption_summary, -disruption_rate)
  return(disruption_summary)
}


# --- SCRIPT EXECUTION ---
# Create a directory for plots if it doesn't exist
if (!dir.exists("plots")) dir.create("plots")

# Run the evaluation to get the final results table for the 2022 season
res <- evaluate_pass_rushers(analysis_df, pbp_hist, season_year = 2022)

# Rename columns to match your plotting code's expectations
setnames(res, c("full_name", "raw_pass_rush_snaps", "disruption_rate_diff", "disruption_rate", "raw_disruption_rate", "avg_run_prob", "ypc_diff"), 
         c("Player", "Pass_Rush_Snaps", "Disruption_Rate_Diff", "Disruption_Rate", "Raw_Disruption_Rate", "Avg_Run", "YPC_Diff"), skip_absent = TRUE)
# --- PLOTTING SECTION (Adapted to save each plot) ---
library(ggplot2)
library(ggrepel)
library(viridis)

# Plot 1: Top 50 by Raw Disruption Rate
cat("\n[PLOT] Generating and saving Plot 1...\n")
top50v1 <- res[!is.na(YPC_Diff)][order(-Raw_Disruption_Rate)][1:min(50, nrow(res[!is.na(YPC_Diff)]))]
plot1 <- ggplot(top50v1, aes(x = YPC_Diff, y = Raw_Disruption_Rate, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 15) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(title = "Top 50 by Raw Disruption Rate", x = "YPC Allowed (On - Off)", y = "Disruption Rate", caption = "2022 Season") +
  theme_minimal(base_size = 14)
print(plot1); ggsave("plots/01_top50_raw_disruption.png", plot1, width = 12, height = 9, dpi = 300)

# Plot 2: Top 50 by Weighted Disruption Rate
cat("\n[PLOT] Generating and saving Plot 2...\n")
top50v2 <- res[!is.na(YPC_Diff)][order(-Disruption_Rate)][1:min(5, nrow(res[!is.na(YPC_Diff)]))]
plot2 <- ggplot(top50v2, aes(x = YPC_Diff, y = Disruption_Rate, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 15) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(title = "Top 5 by Weighted Disruption Rate", x = "YPC Allowed (On - Off)", y = "Disruption Rate", caption = "2022 Season") +
  theme_minimal(base_size = 14)
print(plot2); ggsave("plots/02_top5_weighted_disruption.png", plot2, width = 12, height = 9, dpi = 300)

# Plot 3: Top 50 "Overachievers"
cat("\n[PLOT] Generating and saving Plot 3...\n")
top50v3 <- res[!is.na(YPC_Diff)][order(-Disruption_Rate_Diff)][1:min(5, nrow(res[!is.na(YPC_Diff)]))]
plot3 <- ggplot(top50v3, aes(x = YPC_Diff, y = Disruption_Rate_Diff, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 15) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(title = "Top 5 Overachievers by Disruption Rate Difference", x = "YPC Allowed (On - Off)", y = "Disruption Rate Residual", caption = "2022 Season") +
  theme_minimal(base_size = 14)
print(plot3); ggsave("plots/03_top5_overachievers.png", plot3, width = 12, height = 9, dpi = 300)

# Plot 4: Top 50 "Underachievers"
cat("\n[PLOT] Generating and saving Plot 4...\n")
top50v4 <- res[!is.na(YPC_Diff)][order(Disruption_Rate_Diff)][1:min(5, nrow(res[!is.na(YPC_Diff)]))]
plot4 <- ggplot(top50v4, aes(x = YPC_Diff, y = Disruption_Rate_Diff, color = Pass_Rush_Snaps)) +
  geom_point(size = 3, alpha = 0.85) +
  geom_text_repel(aes(label = Player), size = 3.5, max.overlaps = 15) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(title = "Top 5 Underachievers by Disruption Rate Difference", x = "YPC Allowed (On - Off)", y = "Disruption Rate Residual", caption = "2022 Season") +
  theme_minimal(base_size = 14)
print(plot4); ggsave("plots/04_top5_underachievers.png", plot4, width = 12, height = 9, dpi = 300)

# Plot 5: Contextual Disruption Rate
cat("\n[PLOT] Generating and saving Plot 5...\n")
plot_data_context <- res[!is.na(YPC_Diff)]
lm_fit <- lm(Raw_Disruption_Rate ~ Avg_Run, data = plot_data_context)
plot_data_context$residual <- plot_data_context$Raw_Disruption_Rate - predict(lm_fit)
top_performers <- plot_data_context[order(-residual)][1:15]
plot5 <- ggplot(plot_data_context, aes(x = Avg_Run, y = Raw_Disruption_Rate)) +
  geom_point(aes(color = Pass_Rush_Snaps), alpha = 0.7) +
  geom_smooth(method = "lm", se = TRUE, color = "black", linetype = "dashed", linewidth = 1.0) +
  geom_text_repel(data = top_performers, aes(label = Player), size = 4.0, max.overlaps = 15) +
  scale_color_viridis_c(option = "D", direction = 1, name = "Pass Rush Snaps") +
  labs(title = "Disruption Rate vs. Situational Context", x = "Average Run % Faced", y = "Raw Disruption Rate") +
  theme_minimal(base_size = 14)
print(plot5); ggsave("plots/05_disruption_vs_context.png", plot5, width = 12, height = 9, dpi = 300)

cat("\n[PLOTS] All visualizations have been saved to the 'plots' directory.\n")