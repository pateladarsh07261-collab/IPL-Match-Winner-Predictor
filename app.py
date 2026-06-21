import streamlit as st
import pandas as pd
import joblib

# Load model
rf_pipeline = joblib.load("rf_pipeline.pkl")

# Load dataset
mdf = pd.read_csv("matches_final.csv")
st.title("🏏 IPL Match Winner Predictor")

teams = sorted(mdf["team1"].unique())

team1 = st.selectbox("Select Team 1", teams)

team2 = st.selectbox(
    "Select Team 2",
    [t for t in teams if t != team1]
)

city = st.selectbox(
    "Select City",
    sorted(mdf["city"].dropna().unique())
)

venue = st.selectbox(
    "Select Venue",
    sorted(mdf["venue"].unique())
)

toss_winner = st.selectbox(
    "Select Toss Winner",
    [team1, team2]
)

toss_decision = st.selectbox(
    "Toss Decision",
    ["bat", "field"]
)

season = st.text_input("Enter Season")

match_type = st.selectbox(
    "Match Type",
    sorted(mdf["match_type"].unique())
)

if st.button("Predict Winner"):

    team1_avg_score = mdf[mdf["team1"] == team1]["team1_avg_score"].mean()
    team2_avg_score = mdf[mdf["team2"] == team2]["team2_avg_score"].mean()

    team1_avg_wickets = mdf[mdf["team1"] == team1]["team1_avg_wickets"].mean()
    team2_avg_wickets = mdf[mdf["team2"] == team2]["team2_avg_wickets"].mean()

    team1_win_pct = mdf[mdf["team1"] == team1]["team1_win_pct"].mean()
    team2_win_pct = mdf[mdf["team2"] == team2]["team2_win_pct"].mean()

    sample_match = pd.DataFrame({
        "season":[season],
        "city":[city],
        "match_type":[match_type],
        "team1":[team1],
        "team2":[team2],
        "toss_winner":[toss_winner],
        "toss_decision":[toss_decision],
        "venue":[venue],
        "team1_avg_score":[team1_avg_score],
        "team2_avg_score":[team2_avg_score],
        "team1_avg_wickets":[team1_avg_wickets],
        "team2_avg_wickets":[team2_avg_wickets],
        "team1_win_pct":[team1_win_pct],
        "team2_win_pct":[team2_win_pct]
    })

    prediction = rf_pipeline.predict(sample_match)

    probs = rf_pipeline.predict_proba(sample_match)[0]

    winner_index = probs.argmax()

    confidence = probs[winner_index] * 100

    st.success(f"🏆 Predicted Winner: {prediction[0]}")
    st.info(f"📊 Confidence: {confidence:.2f}%")
    