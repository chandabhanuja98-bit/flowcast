"""Run the FlowCast pipeline on an uploaded traffic file and predict with the saved models.

Steps: validate and load (ingest), clean and merge (clean), build features (features),
predict with the winning saved models (ml_models). Nothing is retrained and the original
data, report and predictions are never overwritten. Results go to data/uploads and
data/processed/predictions_upload_<version>.csv.
"""
import contextlib
import io
import json
import shutil
import threading
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import pandas as pd

import clean
import features
import ingest
import ml_models
from features import DATASET_VERSION, PROCESSED_DIRECTORY
from ingest import CALENDAR_FILE, PROJECT_ROOT, REQUIRED_COLUMNS, TIMEZONE, TRAFFIC_FILE, WEATHER_FILE
from utils import model_path

UPLOAD_DIRECTORY = PROJECT_ROOT / "data" / "uploads"
UPLOAD_PREDICTIONS_FILE = PROCESSED_DIRECTORY / f"predictions_upload_{DATASET_VERSION}.csv"
UPLOAD_INFO_FILE = UPLOAD_PREDICTIONS_FILE.with_suffix(".json")
TEMPLATE_DIRECTORY = PROJECT_ROOT / "data" / "templates"
TEMPLATE_FILES = {
    TRAFFIC_FILE: "traffic_sensor_log_template.csv",
    WEATHER_FILE: "weather_observations_template.csv",
    CALENDAR_FILE: "calendar_events_template.csv",
}
MIN_DAYS = 8
KEEP_RUNS = 3
RUN_LOCK = threading.Lock()


class UploadError(ValueError):
    """A problem with the uploaded data that the user can fix."""


@contextlib.contextmanager
def use_run_directories(run_directory: Path):
    """Point the ingest module at this run's own raw and quarantine folders."""
    saved = (ingest.RAW_DATA_DIRECTORY, ingest.QUARANTINE_DIRECTORY)
    ingest.RAW_DATA_DIRECTORY = run_directory / "raw"
    ingest.QUARANTINE_DIRECTORY = run_directory / "quarantine"
    try:
        yield
    finally:
        ingest.RAW_DATA_DIRECTORY, ingest.QUARANTINE_DIRECTORY = saved


def save_inputs(run_directory: Path, uploads: dict[str, bytes | None]) -> None:
    """Write the uploaded tables into the run folder. Tables not uploaded are copied from data/raw."""
    raw = run_directory / "raw"
    raw.mkdir(parents=True)
    for name in REQUIRED_COLUMNS:
        if uploads.get(name) is not None:
            (raw / name).write_bytes(uploads[name])
        else:
            shutil.copy(ingest.RAW_DATA_DIRECTORY / name, raw / name)


def check_time_span(traffic: pd.DataFrame) -> None:
    span = traffic["timestamp"].max() - traffic["timestamp"].min()
    if span < pd.Timedelta(days=MIN_DAYS - 1, hours=23):
        raise UploadError(
            f"The file covers {span.days} days. The features use the previous week of readings, "
            f"so it needs at least {MIN_DAYS} days of data for every road."
        )


def load_tables() -> dict[str, pd.DataFrame]:
    """Validate and load the run's three tables, turning errors into messages the user can act on."""
    try:
        tables = ingest.load_and_validate_tables()
    except pd.errors.ParserError as error:
        raise UploadError(f"The file could not be read as a CSV: {error}") from error
    except pd.errors.EmptyDataError as error:
        raise UploadError("The file is empty.") from error
    except (ValueError, FileNotFoundError) as error:
        raise UploadError(str(error)) from error
    check_time_span(tables[TRAFFIC_FILE])
    return tables


def clean_tables(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    clean.quality_report.clear()
    try:
        return clean.clean_and_merge(tables)
    except ValueError as error:
        if "Nulls found" in str(error):
            raise UploadError(
                "After joining, some rows have no weather or calendar data. The weather and calendar tables "
                "must cover every date in the traffic file. Upload matching weather and calendar files as well. "
                f"Detail: {error}"
            ) from error
        raise UploadError(f"The cleaned data failed its final checks: {error}") from error


def build_feature_table(merged: pd.DataFrame, run_directory: Path) -> pd.DataFrame:
    """Write the merged table, read it back as the offline pipeline does, and build the features."""
    merged_file = run_directory / "merged.csv"
    merged.to_csv(merged_file, index=False)
    try:
        prepared, _, _ = features.prepare_features(features.load_merged(merged_file))
    except ValueError as error:
        raise UploadError(f"The feature checks failed: {error}") from error
    if prepared.empty:
        raise UploadError(f"No rows are left after removing the first week of history. Upload at least {MIN_DAYS} days of data.")
    return prepared


def read_model_cards(winners: dict[str, str]) -> dict[str, dict]:
    return {target: json.loads(model_path(name, target, "json").read_text(encoding="utf-8")) for target, name in winners.items()}


def add_missing_weather_columns(prepared: pd.DataFrame, cards: dict[str, dict]) -> pd.DataFrame:
    """A weather label missing from the upload becomes an all-zero column, as the models expect it."""
    for card in cards.values():
        for column in card["feature_columns"]:
            if column in prepared.columns:
                continue
            if not column.startswith("weather_"):
                raise UploadError(f"The models need the feature {column}, which could not be built from this file.")
            prepared[column] = 0
    return prepared


def predict_upload(prepared: pd.DataFrame, winners: dict[str, str]) -> pd.DataFrame:
    """Predict with the saved winners. Intervals and accident calibration come from the original validation window."""
    if not ml_models.PREDICTIONS_FILE.exists():
        raise UploadError("The original predictions file is missing. Run: python src/ml_models.py predict")
    baseline = pd.read_csv(ml_models.PREDICTIONS_FILE)
    for column in ["timestamp", "forecast_for"]:
        baseline[column] = pd.to_datetime(baseline[column], utc=True).dt.tz_convert(TIMEZONE)

    uploaded = ml_models.predict_windows(prepared, winners).assign(split="upload")
    combined = pd.concat([baseline, uploaded], ignore_index=True)
    combined = ml_models.calibrate_accident_risk(ml_models.add_prediction_intervals(combined))
    return combined[combined["split"] == "upload"].reset_index(drop=True)


def count_quarantined(run_directory: Path) -> dict[str, int]:
    folder = run_directory / "quarantine"
    return {file.stem: len(pd.read_csv(file)) for file in sorted(folder.glob("*.csv"))} if folder.exists() else {}


def prune_old_runs() -> None:
    runs = sorted(path for path in UPLOAD_DIRECTORY.iterdir() if path.is_dir())
    for old in runs[:-KEEP_RUNS]:
        shutil.rmtree(old, ignore_errors=True)


def run_upload(uploads: dict[str, bytes | None], file_names: dict[str, str], step=print) -> dict:
    """Run the full pipeline on the uploaded tables and save the predictions and a lineage record.

    uploads maps a table file name to its bytes (None = use the original table from data/raw).
    step is called with a short progress message before each stage.
    """
    if uploads.get(TRAFFIC_FILE) is None:
        raise UploadError("Upload a traffic sensor log first.")
    with RUN_LOCK:
        run_id = datetime.now(ZoneInfo(TIMEZONE)).strftime("%Y%m%d_%H%M%S")
        run_directory = UPLOAD_DIRECTORY / run_id
        log = io.StringIO()
        try:
            save_inputs(run_directory, uploads)
            with use_run_directories(run_directory), contextlib.redirect_stdout(log):
                step("1/4 Checking the file: columns, number formats, dates and ranges")
                tables = load_tables()
                rows_uploaded = len(tables[TRAFFIC_FILE])
                step("2/4 Cleaning: duplicates, impossible values, gaps, labels, joining weather and calendar")
                merged = clean_tables(tables)
                step("3/4 Building the features (lags, rolling statistics, time and weather features)")
                prepared = build_feature_table(merged, run_directory)
                step("4/4 Predicting with the saved models")
                try:
                    winners = ml_models.choose_winners()
                except SystemExit as error:
                    raise UploadError("No trained models were found. Train the models first.") from error
                cards = read_model_cards(winners)
                predictions = predict_upload(add_missing_weather_columns(prepared, cards), winners)
        except Exception:
            shutil.rmtree(run_directory, ignore_errors=True)
            raise

        PROCESSED_DIRECTORY.mkdir(parents=True, exist_ok=True)
        predictions.to_csv(UPLOAD_PREDICTIONS_FILE, index=False)
        info = {
            "run_id": run_id,
            "created": datetime.now(ZoneInfo(TIMEZONE)).isoformat(timespec="seconds"),
            "files": {name: file_names.get(name) or "original table from data/raw" for name in REQUIRED_COLUMNS},
            "rows_uploaded": rows_uploaded,
            "rows_predicted": len(predictions),
            "roads": int(predictions["road_id"].nunique()),
            "forecast_from": str(predictions["forecast_for"].min()),
            "forecast_to": str(predictions["forecast_for"].max()),
            "dataset_version": DATASET_VERSION,
            "models": {target: {"model": name, "trained_at": cards[target]["trained_at"], "training_window_to": cards[target]["training_window"]["to"]} for target, name in winners.items()},
            "cleaning_summary": {key: str(value) for key, value in clean.quality_report.items()},
            "quarantined_rows": count_quarantined(run_directory),
        }
        UPLOAD_INFO_FILE.write_text(json.dumps(info, indent=2), encoding="utf-8")
        (run_directory / "run_log.txt").write_text(log.getvalue(), encoding="utf-8")
        prune_old_runs()
        return info


def load_last_run() -> dict | None:
    """The lineage record of the most recent upload run, if there is one."""
    if not (UPLOAD_INFO_FILE.exists() and UPLOAD_PREDICTIONS_FILE.exists()):
        return None
    return json.loads(UPLOAD_INFO_FILE.read_text(encoding="utf-8"))
