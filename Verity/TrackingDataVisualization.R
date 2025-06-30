library(readr)
library(dplyr)
library(stringr)
library(plotly)

# 4) Load data
games    <- read_csv("games.csv")
plays    <- read_csv("plays.csv")
tracking <- read_csv("tracking_week_1.csv")

# 1) Full team-name lookup
team_full <- c(
  ARI="Arizona Cardinals", ATL="Atlanta Falcons", BAL="Baltimore Ravens",
  BUF="Buffalo Bills", CAR="Carolina Panthers", CHI="Chicago Bears",
  CIN="Cincinnati Bengals", CLE="Cleveland Browns", DAL="Dallas Cowboys",
  DEN="Denver Broncos", DET="Detroit Lions", GB="Green Bay Packers",
  HOU="Houston Texans", IND="Indianapolis Colts", JAX="Jacksonville Jaguars",
  KC="Kansas City Chiefs", LA="Los Angeles Rams", LAC="Los Angeles Chargers",
  LV="Las Vegas Raiders", MIA="Miami Dolphins", MIN="Minnesota Vikings",
  NE="New England Patriots", NO="New Orleans Saints", NYG="New York Giants",
  NYJ="New York Jets", PHI="Philadelphia Eagles", PIT="Pittsburgh Steelers",
  SEA="Seattle Seahawks", SF="San Francisco 49ers", TB="Tampa Bay Buccaneers",
  TEN="Tennessee Titans", WAS="Washington Commanders"
)

# 2) Ordinal-suffix helper
ord_suffix <- function(x) {
  if (x %% 10 == 1 && x %% 100 != 11) return("st")
  if (x %% 10 == 2 && x %% 100 != 12) return("nd")
  if (x %% 10 == 3 && x %% 100 != 13) return("rd")
  "th"
}

# 3) Team colours + halo colours
team_cols <- list(
  ARI=c("#97233F","#000000"), ATL=c("#A71930","#000000"), BAL=c("#241773","#000000"),
  BUF=c("#00338D","#C60C30"), CAR=c("#0085CA","#101820"), CHI=c("#0B162A","#C83803"),
  CIN=c("#FB4F14","#000000"), CLE=c("#311D00","#FF3C00"), DAL=c("#003594","#041E42"),
  DEN=c("#FB4F14","#002244"), DET=c("#0076B6","#B0B7BC"), GB =c("#203731","#FFB612"),
  HOU=c("#03202F","#A71930"), IND=c("#002C5F","#A2AAAD"), JAX=c("#101820","#D7A22A"),
  KC =c("#E31837","#FFB81C"), LA =c("#003594","#FFA300"), LAC=c("#0080C6","#FFC20E"),
  LV =c("#000000","#A5ACAF"), MIA=c("#008E97","#FC4C02"), MIN=c("#4F2683","#FFC62F"),
  NE =c("#002244","#C60C30"), NO =c("#101820","#D3BC8D"), NYG=c("#0B2265","#A71930"),
  NYJ=c("#125740","#000000"), PHI=c("#004C54","#A5ACAF"), PIT=c("#FFB612","#101820"),
  SEA=c("#002244","#69BE28"), SF =c("#AA0000","#B3995D"), TB =c("#D50A0A","#FF7900"),
  TEN=c("#0C2340","#4B92DB","#C8102E"), WAS=c("#5A1414","#FFB612"),
  football=c("#CBB67C","#663831")
)

# 5) Pick play
gid <- 2022091200
pid <- 286

# 6) Filter & compute speed
df <- tracking %>%
  filter(gameId == gid, playId == pid) %>%
  mutate(speed_mph = s * 2.236936)
if (!nrow(df)) stop("No tracking data for that play!")

# Identify football club to hide its legend entry
ball_club <- df %>%
  filter(displayName == "football") %>%
  pull(club) %>%
  unique()

# 7) Pull metadata
gm       <- games %>% filter(gameId == gid)
pm       <- plays %>% filter(gameId == gid, playId == pid)
los      <- pm$absoluteYardlineNumber[1]
yards_to <- pm$yardsToGo[1]
dir_vec  <- df$playDirection[!is.na(df$playDirection)]
play_dir <- if (length(dir_vec)) dir_vec[1] else "right"
fd_raw   <- if (play_dir == "right") los + yards_to else los - yards_to
down     <- pm$down[1]
qtr      <- pm$quarter[1]
clock    <- pm$gameClock[1]

# 8) Build title
away_abbr <- gm$visitorTeamAbbr[1]
home_abbr <- gm$homeTeamAbbr[1]
away_full <- team_full[away_abbr]
home_full <- team_full[home_abbr]
q_label   <- paste0(qtr, ord_suffix(qtr), " Q")
score_lbl <- sprintf("%d–%d", gm$visitorFinalScore[1], gm$homeFinalScore[1])
dd_lbl    <- paste0(down, ord_suffix(down), " & ", yards_to)
top_title <- paste(
  sprintf("%s @ %s", away_full, home_full),
  "│", q_label, clock,
  "│ Score:", score_lbl,
  "│", dd_lbl
)

# 9) Field shapes & yard-lines
yard_lines      <- seq(10,110,by=10)
yard_lines_shft <- yard_lines + 10
mid_lines_shft  <- yard_lines_shft - 5

shapes <- c(
  lapply(seq(1,119), function(xv) list(type="line", xref="x", yref="y",
                                       x0=xv, x1=xv, y0=23, y1=24,
                                       line=list(color="white", width=1), layer="below")),
  lapply(seq(1,119), function(xv) list(type="line", xref="x", yref="y",
                                       x0=xv, x1=xv, y0=29, y1=30,
                                       line=list(color="white", width=1), layer="below")),
  lapply(yard_lines_shft, function(xv) list(type="line", xref="x", yref="y",
                                            x0=xv-0.5, x1=xv+0.5, y0=24, y1=24,
                                            line=list(color="white", width=1), layer="below")),
  lapply(yard_lines_shft, function(xv) list(type="line", xref="x", yref="y",
                                            x0=xv-0.5, x1=xv+0.5, y0=29, y1=29,
                                            line=list(color="white", width=1), layer="below")),
  lapply(mid_lines_shft, function(xv) list(type="line", xref="x", yref="y",
                                           x0=xv, x1=xv, y0=0, y1=53.5,
                                           line=list(color="white", width=1), layer="below")),
  lapply(yard_lines_shft, function(xv) list(type="line", xref="x", yref="y",
                                            x0=xv, x1=xv, y0=0, y1=53.5,
                                            line=list(color="white", width=2), layer="below")),
  list(
    list(type="line", x0=los,   x1=los,   y0=0, y1=53.5,
         line=list(color="blue", dash="dash", width=2), layer="below"),
    list(type="line", x0=fd_raw, x1=fd_raw, y0=0, y1=53.5,
         line=list(color="yellow", dash="dash", width=2), layer="below")
  ),
  list(
    list(type="line", x0=10,  x1=10,  y0=0, y1=53.5,
         line=list(color="white", width=4), layer="below"),
    list(type="line", x0=110, x1=110, y0=0, y1=53.5,
         line=list(color="white", width=4), layer="below")
  ),
  list(
    list(type="rect", x0=0,   x1=10,  y0=0, y1=53.5,
         fillcolor=team_cols[[home_abbr]][1], line=list(width=0), layer="below"),
    list(type="rect", x0=110, x1=120, y0=0, y1=53.5,
         fillcolor=team_cols[[away_abbr]][1], line=list(width=0), layer="below")
  )
)

# 10) Yard-numbers & endzones
num_labels <- data.frame(
  x     = yard_lines_shft,
  y_top = 48.5, y_bot = 5,
  raw   = c("10","20","30","40","50","40","30","20","10","","")
) %>% filter(raw!="") %>% mutate(
  bottom_lbl = ifelse(x<60, paste0("\u25C2",raw),
                      ifelse(x>60,paste0(raw,"\u25B8"), raw)),
  top_lbl    = ifelse(x<60, paste0(raw,"\u25B8"),
                      ifelse(x>60,paste0("\u25C2",raw), raw))
)

endzone_labels <- data.frame(
  x     = c(5,115),
  y     = rep(53.5/2,2),
  text  = c(home_abbr, away_abbr),
  angle = c(90,-90)
)

# 11) Build plot, hiding the football legend
clubs <- unique(df$club)
fig <- plot_ly()
for (cl in clubs) {
  sub     <- df %>% filter(club == cl)
  fill    <- team_cols[[cl]][1]
  halo    <- team_cols[[cl]][2]
  is_ball <- cl == ball_club
  
  fig <- fig %>%
    add_trace(
      data       = sub,
      x          = ~x, y = ~y, frame = ~frameId,
      type       = "scatter", mode = "markers",
      marker     = list(size = 12, color = fill,
                        line = list(width = 2, color = halo)),
      name       = if (cl == "football") "Football" else cl,
      showlegend = !is_ball,
      hoverinfo  = "text",
      text       = ~paste0("nflId: ", nflId,
                           "<br>Name: ", displayName,
                           "<br>Speed: ", round(speed_mph,1), " MPH")
    )
}

# 12) Layout & animation controls (no axis labels or ticks, title in black)
fig <- fig %>%
  layout(
    paper_bgcolor = "#DEDEDF",
    plot_bgcolor  = "#00B140",
    margin        = list(t=140, r=200, b=60),
    legend        = list(x=1.05, y=0.8),
    title         = list(
      text   = top_title,
      x      = 0.5, xanchor="center",
      y      = 1.15, yanchor="top",
      font   = list(size=18, color="black")   # <-- title color set to black
    ),
    xaxis = list(
      range          = c(0,120),
      showgrid       = FALSE,
      zeroline       = FALSE,
      showticklabels = FALSE,
      title          = ""
    ),
    yaxis = list(
      range          = c(0,53.5),
      showgrid       = FALSE,
      zeroline       = FALSE,
      showticklabels = FALSE,
      title          = ""
    ),
    shapes = shapes
  ) %>%
  add_annotations(
    data      = num_labels, x=~x, y=~y_top, text=~top_lbl,
    textangle = 180, font=list(color="white", size=20, family="Arial Narrow"),
    showarrow = FALSE, layer="below"
  ) %>%
  add_annotations(
    data      = num_labels, x=~x, y=~y_bot, text=~bottom_lbl,
    font      = list(color="white", size=20, family="Arial Narrow"),
    showarrow = FALSE, layer="below"
  ) %>%
  add_annotations(
    data      = endzone_labels, x=~x, y=~y, text=~text, textangle=~angle,
    font      = list(size=32, color="white", family="Arial Narrow"), showarrow=FALSE
  ) %>%
  add_annotations(
    x         = fd_raw, y=53.5, text=down,
    font      = list(size=16, color="black"), showarrow=FALSE,
    ay        = -25, bordercolor="black",
    borderwidth=2, borderpad=4, bgcolor="#FF7300"
  ) %>%
  animation_opts(frame=100, transition=0, redraw=FALSE) %>%
  animation_slider(currentvalue=list(prefix="Frame "))

# 13) Show
fig
