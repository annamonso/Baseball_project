# Data Dictionary

| Column | Type | Non-null % | Description |
|---|---|---:|---|
| game_date | object | 100.0% |  |
| game_pk | Int64 | 100.0% |  |
| pitcher | Int64 | 100.0% |  |
| batter | Int64 | 100.0% |  |
| pitch_type | category | 100.0% |  |
| release_speed | Float64 | 100.0% | Pitch velocity (mph). |
| release_pos_x | Float64 | 100.0% | Release point horizontal (ft). |
| release_pos_z | Float64 | 100.0% | Release point height (ft). |
| pfx_x | Float64 | 100.0% | Horizontal break of pitch at 40ft (ft). |
| pfx_z | Float64 | 100.0% | Vertical break of pitch at 40ft (ft). |
| spin_rate_deprecated | Int64 | 0.0% | Legacy spin rate column; prefer spin_rate if present. |
| p_throws | category | 100.0% | Pitcher throwing hand (R/L). |
| stand | category | 100.0% | Batter stance (R/L). |
| balls | Int64 | 100.0% | Balls in count before pitch. |
| strikes | Int64 | 100.0% | Strikes in count before pitch. |
| inning | Int64 | 100.0% | Inning number. |
| outs_when_up | Int64 | 100.0% | Number of outs when batter came up. |
| on_1b | Int64 | 31.4% | Runner on 1st (id) or NaN. |
| on_2b | Int64 | 19.3% | Runner on 2nd (id) or NaN. |
| on_3b | Int64 | 9.8% | Runner on 3rd (id) or NaN. |
| description | category | 100.0% | Text description of the pitch event. |
| events | category | 27.3% | Play outcome for the pitch/plate appearance. |
| hc_x | Float64 | 18.3% | Hit coordinate x on field diagram. |
| hc_y | Float64 | 18.3% | Hit coordinate y on field diagram. |
| runner_on_1b | int64 | 100.0% |  |
| runner_on_2b | int64 | 100.0% |  |
| runner_on_3b | int64 | 100.0% |  |
