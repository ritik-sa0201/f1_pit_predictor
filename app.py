from pathlib import Path

import pandas as pd
import streamlit as st

from core import COMPOUNDS, clean_laps, compare, train

st.set_page_config(page_title="F1 Pit Stop Predictor")
st.title("F1 Pit Stop Predictor")
st.write("Bahrain Grand Prix · Dry, green-flag running")


@st.cache_resource(show_spinner=False)
def prepare_model():
    path = Path(__file__).resolve().parent / "data" / "bahrain_laps.csv"
    laps = pd.read_csv(path, dtype={"TrackStatus": str})
    seasons = {}
    for year in (2022, 2023, 2024):
        race = laps.loc[laps["Year"] == year]
        distance = int(race["LapNumber"].max())
        seasons[year] = clean_laps(race, distance), distance
    earlier = pd.concat([seasons[2022][0], seasons[2023][0]], ignore_index=True)
    latest, distance = seasons[2024]
    model, error = train(earlier, latest)
    return model, error, distance, len(earlier), len(latest)


try:
    with st.spinner("Loading Bahrain race laps and training the model…"):
        model, error, distance, n_train, n_test = prepare_model()
except Exception as exc:
    st.error(f"Could not prepare the race data. {exc}")
    st.caption("Check that data/bahrain_laps.csv was extracted alongside the app.")
    if st.button("Retry"):
        st.rerun()
    st.stop()

st.caption(f"Trained on {n_train:,} laps from 2022–23. Average absolute lap-time error "
           f"on {n_test:,} held-out laps from 2024: {error:.2f} s.")
completed = st.number_input("Lap just completed", 1, distance - 1, min(25, distance - 1))
age = st.number_input("Laps already run on this tyre set", 1, 100, 12)
current = st.selectbox("Current tyres", COMPOUNDS, index=1)
fresh = st.selectbox("New tyres if you stop", COMPOUNDS, index=2)
loss = st.number_input("Assumed pit-stop time loss (seconds)", 0.0, 90.0, 22.0, 0.5)

stay, plans = compare(model, completed, distance, age, current, fresh, loss)
best = plans.loc[plans["Time (s)"].idxmin()]
gain = best["Gain vs staying out (s)"]
if gain > 0:
    st.success(f"Pit at the end of lap {int(best['Pit lap'])} for {fresh.lower()} tyres. "
               f"Estimated gain over staying out: {gain:.2f} s.")
else:
    st.success(f"Stay out. The fastest one-stop plan costs an extra {-gain:.2f} s.")

st.subheader("Predicted time to the finish")
chart = plans.set_index("Pit lap")[["Time (s)"]].rename(columns={"Time (s)": "One stop"})
chart["Stay out"] = stay
st.line_chart(chart, x_label="Pit at end of lap", y_label="Remaining time (seconds)")
st.caption("Each stop uses one fresh set. The comparison assumes the driver can legally "
           "finish on either plan. Traffic, safety cars and future weather are outside the model.")
