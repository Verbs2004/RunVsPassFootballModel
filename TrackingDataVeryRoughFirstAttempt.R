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
})

# Performance setup
n_cores <- min(detectCores() - 1,12)
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
  
  df <- pbp_data[play_type %in% c("run", "pass") & !is.na(down) & down %in% 1:4 & !is.na(ydstogo) & ydstogo >= 1 & ydstogo <= 50 & !is.na(yardline_100) & yardline_100 >= 1 & yardline_100 <= 99 & !is.na(old_game_id) & !is.na(play_id)]
  df[, season := as.integer(substr(old_game_id, 1, 4))]
  cat("  [FEAT_M1] After initial filtering:", nrow(df), "plays retained.\n")
  
  df[, is_pass := as.numeric(play_type == "pass")]
  
  df[, `:=`(shotgun = fcoalesce(as.numeric(shotgun), 0), no_huddle = fcoalesce(as.numeric(no_huddle), 0), wp = fcoalesce(wp, 0.5), score_differential = fcoalesce(score_differential, 0), qtr = fcoalesce(qtr, 1), goal_to_go = fcoalesce(as.numeric(goal_to_go), 0), quarter_seconds_remaining = fcoalesce(quarter_seconds_remaining, 900), epa = fcoalesce(epa, 0))]
  cat("  [FEAT_M1] Base situational features created.\n")
  
  df[, `:=`(third_down = as.numeric(down == 3), fourth_down = as.numeric(down == 4), short_yardage = as.numeric(ydstogo <= 3), long_yardage = as.numeric(ydstogo >= 8), red_zone = as.numeric(yardline_100 <= 20), two_minute_warning = as.numeric((qtr == 2 | qtr == 4) & quarter_seconds_remaining <= 120), trailing = as.numeric(score_differential < 0), leading = as.numeric(score_differential > 0), close_game = as.numeric(abs(score_differential) <= 7), score_diff_x_time = score_differential * quarter_seconds_remaining, wp_leverage = abs(wp - 0.5), expected_points_added = epa, time_pressure = as.numeric(quarter_seconds_remaining <= 300 & abs(score_differential) <= 10), garbage_time = as.numeric(abs(score_differential) > 21))]
  cat("  [FEAT_M1] Advanced situational features created.\n")
  
  df[, `:=`(down_x_distance = down * ydstogo, third_and_long = as.numeric(down == 3 & ydstogo >= 7), shotgun_x_down = shotgun * down, wp_x_score_diff = wp * score_differential, red_zone_x_down = red_zone * down, short_yardage_x_down = short_yardage * down, time_x_score = quarter_seconds_remaining * abs(score_differential), leverage_x_distance = wp_leverage * ydstogo)]
  cat("  [FEAT_M1] Interaction features created.\n")
  
  feature_cols <- c("down", "ydstogo", "yardline_100", "qtr", "shotgun", "no_huddle", "wp", "score_differential", "goal_to_go", "third_down", "fourth_down", "short_yardage", "long_yardage", "red_zone", "two_minute_warning", "trailing", "leading", "close_game", "down_x_distance", "third_and_long", "shotgun_x_down", "wp_x_score_diff", "score_diff_x_time", "wp_leverage", "expected_points_added", "time_pressure", "garbage_time", "red_zone_x_down", "short_yardage_x_down", "time_x_score", "leverage_x_distance")
  
  id_cols_to_keep <- c("old_game_id", "play_id", "is_pass", "season")
  if ("gameId" %in% names(df)) { id_cols_to_keep <- c(id_cols_to_keep, "gameId") }
  
  final_df <- df[, c(id_cols_to_keep, feature_cols), with = FALSE]
  
  for (col in feature_cols) { if (col %in% names(final_df)) set(final_df, which(is.na(final_df[[col]])), col, 0) }
  cat("  [FEAT_M1] Final feature set cleaned and prepared.\n")
  return(final_df)
}

create_model2_features <- function(pbp_data, participation_data) {
  cat("--- [FEAT_M2] Creating Model 2 features ---\n")
  df <- create_model1_features(pbp_data)
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
    cat("  [FEAT_M3_WARN] No BDB data, returning empty table.\n")
    return(data.table()) 
  }
  
  plays_bdb <- copy(bdb_data$plays)
  games_bdb <- bdb_data$games
  tracking_bdb <- bdb_data$tracking
  
  plays_bdb[, old_game_id := as.character(gameId)]
  
  if (!is.null(games_bdb) && nrow(games_bdb) > 0) {
    games_bdb[, gameId := as.integer(gameId)]
    plays_bdb[, gameId := as.integer(gameId)]
    plays_bdb <- merge(plays_bdb, games_bdb[, .(gameId, week)], by = "gameId", all.x = TRUE)
    cat("  [FEAT_M3] Merged with games data. Week column range:", range(plays_bdb$week, na.rm = TRUE), "\n")
  } else { 
    plays_bdb[, week := 1L]
    cat("  [FEAT_M3_WARN] No games data available, using default week 1.\n")
  }
  
  plays_bdb[, label := ifelse(!is.na(isDropback), as.numeric(isDropback), as.numeric(pff_passCoverage != "" | passResult != ""))]
  
  cat("  [FEAT_M3] Merging BDB plays with PBP context...\n")
  pbp_context <- pbp_hist[, .(old_game_id, play_id, play_type, score_differential, wp, qtr, shotgun, no_huddle, epa, quarter_seconds_remaining, down, ydstogo, yardline_100, goal_to_go)]
  model_df <- merge(plays_bdb, pbp_context, by.x=c("old_game_id", "playId"), by.y=c("old_game_id", "play_id"), all.x=TRUE)
  
  if (!"week" %in% names(model_df)) {
    model_df[, week := 1L]
  } else {
    model_df[, week := as.integer(week)]
    model_df[is.na(week), week := 1L]
  }
  
  cat("  [FEAT_M3] Resolving column name conflicts from merge...\n")
  model_df[, down := fcoalesce(as.integer(down.x), as.integer(down.y))]
  model_df[, ydstogo := fcoalesce(as.integer(yardsToGo), as.integer(ydstogo))]
  model_df[, c("down.x", "down.y", "yardsToGo") := NULL]
  setnames(model_df, "playId", "play_id", skip_absent=TRUE)
  
  label_week_col <- model_df[, .(old_game_id, play_id, label, week)]
  
  cat("  [FEAT_M3] Calling create_model2_features to generate base feature set...\n")
  model_df <- create_model2_features(model_df, participation_data)
  
  model_df <- merge(model_df, label_week_col, by = c("old_game_id", "play_id"), all.x = TRUE)
  
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

# Create features for each model
features_m1 <- create_model1_features(pbp_hist)
features_m2 <- create_model2_features(pbp_modern, participation_data)
features_m3 <- create_model3_features(bdb_data, participation_data, pbp_hist)

# Add Model 4: BDB features without stacking
features_m4 <- copy(features_m3)
m4_feature_cols <- setdiff(names(features_m4), c("old_game_id", "play_id", "label", "is_pass", "gameId", "week"))


# Define feature columns
m1_feature_cols <- setdiff(names(features_m1), c("old_game_id", "play_id", "is_pass", "gameId"))
m2_feature_cols <- setdiff(names(features_m2), c("old_game_id", "play_id", "is_pass", "gameId"))
m3_base_feature_cols <- setdiff(names(features_m3), c("old_game_id", "play_id", "label", "is_pass", "gameId", "week"))

cat("  [FEATURES] M1 features:", length(m1_feature_cols), "columns\n")
cat("  [FEATURES] M2 features:", length(m2_feature_cols), "columns\n")
cat("  [FEATURES] M3 base features:", length(m3_base_feature_cols), "columns\n")
cat("  [FEATURES] M4 features:", length(m4_feature_cols), "columns\n")

# ───────────────────────────────────────────────────────────────────────────────
# 4) CROSS-VALIDATION SETUP WITH PROPER DATA SEPARATION
# ───────────────────────────────────────────────────────────────────────────────

# Identify BDB weeks (2022 weeks 1-9) to exclude from M1 and M2 training
bdb_weeks_2022 <- unique(features_m3$week)
bdb_weeks_2022 <- bdb_weeks_2022[!is.na(bdb_weeks_2022)]
cat("  [CV_SETUP] BDB weeks identified:", paste(bdb_weeks_2022, collapse=", "), "\n")
# Training wrapper function
train_xgb_model_wrapper <- function(data, feature_cols, target_col, model_name) {
  cat("  [TRAIN] Training", model_name, "with", length(feature_cols), "features on", nrow(data), "samples...\n")
  
  X <- as.matrix(data[, feature_cols, with = FALSE])
  y <- data[[target_col]]
  
  # XGBoost parameters
  params <- list(
    objective = "binary:logistic",
    eval_metric = "auc",
    max_depth = 6,
    eta = 0.1,
    subsample = 0.8,
    colsample_bytree = 0.8,
    min_child_weight = 1,
    gamma = 0,
    alpha = 0.1,
    lambda = 1,
    nthread = n_cores
  )
  
  # Train model
  model <- xgboost(
    data = X,
    label = y,
    params = params,
    nrounds = 1000,
    verbose = 0,
    early_stopping_rounds = 50,
    maximize = TRUE
  )
  
  cat("  [TRAIN]", model_name, "training complete.\n")
  return(model)
}
# ───────────────────────────────────────────────────────────────────────────────
# 5) CROSS-VALIDATION FOR EACH MODEL ON ITS DOMAIN
# ───────────────────────────────────────────────────────────────────────────────

cat("\n=== [CV] CROSS-VALIDATION FOR EACH MODEL ON ITS DOMAIN ===\n")

# Model 1 CV: Time-based splits on historical data
cat("\n--- [CV_M1] Model 1 Cross-Validation (Historical Data) ---\n")
m1_cv_results <- list()
m1_years <- unique(features_m1$season)
m1_years <- m1_years[!is.na(m1_years) & m1_years >= 2020]  # Use recent years for CV

if (length(m1_years) >= 2) {
  for (holdout_year in m1_years) {
    cat("  [CV_M1] Fold: Holding out year", holdout_year, "\n")
    
    m1_train_fold <- features_m1[season != holdout_year | is.na(season)]
    m1_test_fold <- features_m1[season == holdout_year]
    
    if (nrow(m1_train_fold) > 0 && nrow(m1_test_fold) > 0) {
      model_m1_fold <- train_xgb_model_wrapper(m1_train_fold, m1_feature_cols, "is_pass", "M1_CV")
      predictions <- predict(model_m1_fold, as.matrix(m1_test_fold[, m1_feature_cols, with = FALSE]))
      
      fold_auc <- pROC::auc(m1_test_fold$is_pass, predictions, quiet = TRUE)
      m1_cv_results[[length(m1_cv_results) + 1]] <- list(
        year = holdout_year,
        auc = as.numeric(fold_auc),
        n_test = nrow(m1_test_fold)
      )
      cat("  [CV_M1] Year", holdout_year, "AUC:", round(fold_auc, 4), "\n")
    }
  }
}

# Model 2 CV: Exclude BDB weeks, use remaining modern data
cat("\n--- [CV_M2] Model 2 Cross-Validation (Modern Data, Excluding BDB) ---\n")
m2_cv_results <- list()

# Get BDB game IDs to exclude
if (nrow(features_m3) > 0) {
  bdb_game_ids <- unique(features_m3$old_game_id)
  bdb_game_ids <- bdb_game_ids[!is.na(bdb_game_ids)]
  m2_clean_data <- features_m2[!old_game_id %in% bdb_game_ids]
  cat("  [CV_M2] Excluded", length(bdb_game_ids), "BDB games from M2 CV\n")
} else {
  m2_clean_data <- features_m2
}

m2_years <- unique(m2_clean_data$season)
m2_years <- m2_years[!is.na(m2_years) & m2_years >= 2020]

if (length(m2_years) >= 2) {
  for (holdout_year in m2_years) {
    cat("  [CV_M2] Fold: Holding out year", holdout_year, "\n")
    
    m2_train_fold <- m2_clean_data[season != holdout_year | is.na(season)]
    m2_test_fold <- m2_clean_data[season == holdout_year]
    
    if (nrow(m2_train_fold) > 0 && nrow(m2_test_fold) > 0) {
      model_m2_fold <- train_xgb_model_wrapper(m2_train_fold, m2_feature_cols, "is_pass", "M2_CV")
      predictions <- predict(model_m2_fold, as.matrix(m2_test_fold[, m2_feature_cols, with = FALSE]))
      
      fold_auc <- pROC::auc(m2_test_fold$is_pass, predictions, quiet = TRUE)
      m2_cv_results[[length(m2_cv_results) + 1]] <- list(
        year = holdout_year,
        auc = as.numeric(fold_auc),
        n_test = nrow(m2_test_fold)
      )
      cat("  [CV_M2] Year", holdout_year, "AUC:", round(fold_auc, 4), "\n")
    }
  }
}

# Model 3 CV: Week-based splits on BDB data
cat("\n--- [CV_M3] Model 3 Cross-Validation (BDB Data) ---\n")
m3_cv_results <- list()
cv_weeks <- unique(features_m3$week)
cv_weeks <- cv_weeks[!is.na(cv_weeks)]

if (length(cv_weeks) >= 2) {
  # Prepare base models trained on non-BDB data for M3 stacking
  if (nrow(features_m3) > 0) {
    bdb_game_ids <- unique(features_m3$old_game_id)
    m1_base_train <- features_m1[!old_game_id %in% bdb_game_ids]
    m2_base_train <- features_m2[!old_game_id %in% bdb_game_ids]
  } else {
    m1_base_train <- features_m1
    m2_base_train <- features_m2
  }
  
  # Train base models once for M3 stacking
  base_model_1 <- train_xgb_model_wrapper(m1_base_train, m1_feature_cols, "is_pass", "M1_Base_for_M3")
  base_model_2 <- train_xgb_model_wrapper(m2_base_train, m2_feature_cols, "is_pass", "M2_Base_for_M3")
  
  for (holdout_week in cv_weeks) {
    cat("  [CV_M3] Fold: Holding out week", holdout_week, "\n")
    
    m3_train_fold <- features_m3[week != holdout_week]
    m3_test_fold <- features_m3[week == holdout_week]
    
    if (nrow(m3_train_fold) > 0 && nrow(m3_test_fold) > 0) {
      # Add prediction features
      m3_train_fold[, m1_pred_feat := predict(base_model_1, as.matrix(m3_train_fold[, m1_feature_cols, with = FALSE]))]
      m3_train_fold[, m2_pred_feat := predict(base_model_2, as.matrix(m3_train_fold[, m2_feature_cols, with = FALSE]))]
      
      m3_test_fold[, m1_pred_feat := predict(base_model_1, as.matrix(m3_test_fold[, m1_feature_cols, with = FALSE]))]
      m3_test_fold[, m2_pred_feat := predict(base_model_2, as.matrix(m3_test_fold[, m2_feature_cols, with = FALSE]))]
      
      # Train stacked model
      m3_stacked_cols <- c(m3_base_feature_cols, "m1_pred_feat", "m2_pred_feat")
      # Fix:
      model_m3_fold <- train_xgb_model_wrapper(m3_train_fold, m3_stacked_cols, "is_pass", "M3_CV")
      
      # Calculate feature importance for the current fold
      importance_matrix <- xgb.importance(feature_names = m3_stacked_cols, model = model_m3_fold)
      cat("    [CV_M3] Top 94 features for week", holdout_week, "holdout:\n")
      print(head(importance_matrix, 94))
      
      # Predict
      predictions <- predict(model_m3_fold, as.matrix(m3_test_fold[, m3_stacked_cols, with = FALSE]))
      # --- START: Added code for more metrics ---
      
      # Define a threshold to convert probabilities to classes
      threshold <- 0.5
      predicted_classes <- factor(ifelse(predictions > threshold, 1, 0), levels = c(0, 1))
      actual_classes <- factor(m3_test_fold$is_pass, levels = c(0, 1))
      
      # Calculate Confusion Matrix and its statistics (Accuracy, Sensitivity, etc.)
      # Note: Ensure both factors have the same levels to avoid errors.
      cm <- caret::confusionMatrix(predicted_classes, actual_classes)
      
      # Calculate LogLoss
      log_loss <- MLmetrics::LogLoss(y_pred = predictions, y_true = m3_test_fold$is_pass)
      
      # --- END: Added code for more metrics ---
      
      # Calculate AUC
      fold_auc <- pROC::auc(m3_test_fold$is_pass, predictions, quiet = TRUE)
      
      # Store all results for the fold
      m3_cv_results[[length(m3_cv_results) + 1]] <- list(
        week = holdout_week,
        auc = as.numeric(fold_auc),
        log_loss = log_loss,
        confusion_matrix = cm, # The full confusion matrix object
        feature_importance = importance_matrix,
        n_test = nrow(m3_test_fold)
      )
      
      # Print a summary for the fold
      cat("    [CV_M3] Week", holdout_week, "AUC:", round(fold_auc, 4), 
          "| Accuracy:", round(cm$overall['Accuracy'], 4),
          "| LogLoss:", round(log_loss, 4), "\n")
    }
  }
}

# Model 4 CV: Week-based splits on BDB data (no stacking)
cat("\n--- [CV_M4] Model 4 Cross-Validation (BDB Data, No Stacking) ---\n")
m4_cv_results <- list()

if (length(cv_weeks) >= 2 && nrow(features_m4) > 0) {
  for (holdout_week in cv_weeks) {
    cat("  [CV_M4] Fold: Holding out week", holdout_week, "\n")
    
    m4_train_fold <- features_m4[week != holdout_week]
    m4_test_fold <- features_m4[week == holdout_week]
    
    if (nrow(m4_train_fold) > 0 && nrow(m4_test_fold) > 0) {
      # Train non-stacked model
      model_m4_fold <- train_xgb_model_wrapper(m4_train_fold, m4_feature_cols, "is_pass", "M4_CV")
      
      # Predict
      predictions <- predict(model_m4_fold, as.matrix(m4_test_fold[, m4_feature_cols, with = FALSE]))
      
      fold_auc <- pROC::auc(m4_test_fold$is_pass, predictions, quiet = TRUE)
      m4_cv_results[[length(m4_cv_results) + 1]] <- list(
        week = holdout_week,
        auc = as.numeric(fold_auc),
        n_test = nrow(m4_test_fold)
      )
      cat("  [CV_M4] Week", holdout_week, "AUC:", round(fold_auc, 4), "\n")
    }
  }
}

# Print CV summaries
cat("\n--- [CV_SUMMARY] Cross-Validation Results ---\n")
if (length(m1_cv_results) > 0) {
  m1_aucs <- sapply(m1_cv_results, function(x) x$auc)
  cat("  [CV_M1] Mean AUC:", round(mean(m1_aucs), 4), "±", round(sd(m1_aucs), 4), "\n")
}
if (length(m2_cv_results) > 0) {
  m2_aucs <- sapply(m2_cv_results, function(x) x$auc)
  cat("  [CV_M2] Mean AUC:", round(mean(m2_aucs), 4), "±", round(sd(m2_aucs), 4), "\n")
}
if (length(m3_cv_results) > 0) {
  m3_aucs <- sapply(m3_cv_results, function(x) x$auc)
  cat("  [CV_M3] Mean AUC:", round(mean(m3_aucs), 4), "±", round(sd(m3_aucs), 4), "\n")
}
if (length(m4_cv_results) > 0) {
  m4_aucs <- sapply(m4_cv_results, function(x) x$auc)
  cat("  [CV_M4] Mean AUC:", round(mean(m4_aucs), 4), "±", round(sd(m4_aucs), 4), "\n")
}