import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

COMPOUNDS = ("SOFT", "MEDIUM", "HARD")
FEATURES = ["age", "progress", "soft", "medium", "hard"]


def features(age, compound, progress):
    rows = pd.DataFrame({"age": age, "progress": progress, "compound": compound})
    for name in COMPOUNDS:
        rows[name.lower()] = (rows["compound"] == name).astype(int)
    return rows[FEATURES]


def clean_laps(laps, distance):
    valid = (
        laps["IsAccurate"].eq(True)
        & laps["TrackStatus"].eq("1")
        & laps["PitInTime"].isna()
        & laps["PitOutTime"].isna()
        & laps["Compound"].isin(COMPOUNDS)
        & laps["TyreLife"].ge(1)
        & laps["LapNumber"].gt(1)
        & laps["LapTime"].notna()
    )
    clean = laps.loc[valid]
    rows = features(clean["TyreLife"], clean["Compound"], clean["LapNumber"] / distance)
    rows["seconds"] = clean["LapTime"]
    return rows.loc[rows["seconds"] > 0].reset_index(drop=True)


def train(train_rows, test_rows):
    if len(train_rows) < 50 or test_rows.empty:
        raise ValueError("Too few usable laps to train or evaluate this model.")
    for name in COMPOUNDS:
        if not train_rows[name.lower()].any():
            raise ValueError(f"No usable {name} training laps.")
    model = GradientBoostingRegressor(max_depth=3, random_state=42)
    model.fit(train_rows[FEATURES], train_rows["seconds"])
    error = mean_absolute_error(test_rows["seconds"], model.predict(test_rows[FEATURES]))
    return model, error


def compare(model, completed, distance, age, current, fresh, pit_loss):
    """Evaluate stops at the end of lap p, including the just-completed lap."""
    laps = np.arange(completed + 1, distance + 1)
    old_times = model.predict(features(age + laps - completed, current, laps / distance))
    stay_time = float(old_times.sum())
    elapsed = np.r_[0.0, np.cumsum(old_times)]
    plans = []
    for pit_lap in range(completed, distance):
        after = laps[laps > pit_lap]
        new_times = model.predict(features(after - pit_lap, fresh, after / distance))
        total = elapsed[pit_lap - completed] + pit_loss + new_times.sum()
        plans.append((pit_lap, total, stay_time - total))
    return stay_time, pd.DataFrame(plans, columns=["Pit lap", "Time (s)", "Gain vs staying out (s)"])
