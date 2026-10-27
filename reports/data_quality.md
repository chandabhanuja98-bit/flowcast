# Data Quality Report

Written by `src/clean.py` every time the cleaning runs.

## What was found and done

| Check | Result |
|---|---|
| traffic_sensor_log.csv: exact duplicate rows removed | 1767 |
| weather_observations.csv: exact duplicate rows removed | 0 |
| calendar_events.csv: exact duplicate rows removed | 0 |
| Traffic: missing windows before cleaning | 4499 |
| Traffic: duplicate road/timestamp rows removed | 0 |
| weather_observations.csv: duplicate key rows removed | 0 |
| calendar_events.csv: duplicate key rows removed | 0 |
| Weather: different condition labels before / after | 13 / 5 |
| traffic_sensor_log.csv: impossible values set to NaN | 704 |
| weather_observations.csv: impossible values set to NaN | 0 |
| traffic_sensor_log.csv: missing windows added | 4499 |
| traffic_sensor_log.csv: statistical outliers capped (3 x IQR) | 0 |
| traffic_sensor_log.csv: missing sensor values before filling | 45232 |
| traffic_sensor_log.csv: rows quarantined as unrecoverable | 0 |
| weather_observations.csv: missing weather values before filling | 278 |
| traffic_sensor_log.csv: congestion levels derived | 31123 |
| traffic_sensor_log.csv: existing labels that disagree with the rule | 574 |
| Merge: rows with no weather match | 0 |
| Merge: rows with no calendar match | 0 |
| Final: rows (25 roads x every 30 minute window) | 181200 |
| Final: columns | 27 |
| Final: nulls in modelling columns | 0 |

## How each known defect was handled

| Defect | Handling |
|---|---|
| Missing sensor windows | Missing rows added, gaps up to 2 hours interpolated, longer gaps use the road median for that time of day |
| Duplicate log rows | Exact duplicates dropped, then one row kept per road and timestamp |
| Impossible values | Negative values, speed above 200 and occupancy above 100 set to NaN and then filled |
| Hourly weather vs 30 minute traffic | Each traffic window takes the weather of its hour |
| Inconsistent weather labels | Lowercased, `rainy` set to `rain`, `foggy` set to `fog` |
| Mixed date formats | Each file parsed with its own format into one tz-aware timestamp |
| Missing congestion level | Derived from traffic volume divided by road capacity |

## Notes

- Congestion limits (0.25, 0.40, 0.50) were read from the rows that already had a label.
- Travel time was not capped by IQR because the high values are real heavy congestion.
- `vehicle_type_dist` is empty on the added windows, `holiday_name` and `event_name` are empty on normal days.
