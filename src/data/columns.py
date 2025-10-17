KEEP = [
    "game_date",             # Date of the game (useful for temporal splits and tracking)
    "game_pk",               # Unique game identifier (PK = primary key for game)
    "pitcher",               # Pitcher ID (MLBAM player identifier)
    "batter",                # Batter ID (MLBAM player identifier)
    "pitch_type",            # Type of pitch thrown (e.g., FF = Four-seam Fastball, SL = Slider, CU = Curveball)
    "release_speed",         # Pitch velocity at release, in miles per hour (mph)
    "release_pos_x",         # Horizontal release position of the pitcher's hand (in feet, from catcher’s perspective)
    "release_pos_z",         # Vertical release height of the pitcher's hand (in feet)
    "pfx_x",                 # Horizontal movement of the pitch in inches (positive = toward catcher’s right)
    "pfx_z",                 # Vertical movement of the pitch in inches (positive = upward “rise” relative to gravity)
    "spin_rate_deprecated",  # Spin rate of the pitch in revolutions per minute (RPM) — labeled “deprecated” but still common
    "p_throws",              # Pitcher’s throwing hand (‘R’ = right, ‘L’ = left)
    "stand",                 # Batter’s stance (‘R’ = right, ‘L’ = left)
    "balls",                 # Current count of balls (0–3)
    "strikes",               # Current count of strikes (0–2)
    "inning",                # Current inning number of the game
    "outs_when_up",          # Number of outs when this pitch occurred (0–2)
    "on_1b",                 # Runner ID on first base (NaN if no runner)
    "on_2b",                 # Runner ID on second base (NaN if no runner)
    "on_3b",                 # Runner ID on third base (NaN if no runner)
    "px",                    # Horizontal location where the pitch crossed the plate (in feet; 0 = center)
    "pz",                    # Vertical location where the pitch crossed the plate (in feet; ~1.5–3.5 typical strike zone)
    "description",           # Descriptive text of the pitch outcome (e.g., 'hit_into_play', 'called_strike')
    "events",                # Result of the pitch (e.g., 'single', 'home_run', 'strikeout')
    "hc_x",                  # x-coordinate of batted ball contact point or landing spot (for balls in play)
    "hc_y",                  # y-coordinate of batted ball contact point or landing spot (for balls in play)
]


# Post-contact (DO NOT use these for pre-contact modeling). Keep only for EDA sanity.
POST_CONTACT = [
    "launch_speed","launch_angle","hit_distance_sc","barrel","babip_value"
]

# Mapping Statcast events to our outcome classes (you’ll refine later)
EVENT_TO_OUTCOME = {
    "single":"1B","double":"2B","triple":"3B","home_run":"HR",
    "field_error":"ROE","fielders_choice":"OUT","force_out":"OUT",
    "grounded_into_double_play":"OUT","double_play":"OUT",
    "sac_fly":"OUT","sac_bunt":"OUT",
    "field_out":"OUT","flyout":"OUT","lineout":"OUT","pop_out":"OUT",
}
