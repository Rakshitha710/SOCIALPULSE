import streamlit as st
import pandas as pd
import joblib
import re
from pathlib import Path

# PAGE CONFIGURATION
st.set_page_config(
    page_title="SOCIALPULSE",
    page_icon="📊",
    layout="wide"
)

# PROJECT PATHS
PROJECT_PATH = Path(__file__).resolve().parent.parent
DATA_PATH = PROJECT_PATH / "dataset" / "twitter_training_cleaned.csv"
MODEL_PATH = PROJECT_PATH / "models"

# LOAD DATASET
@st.cache_data
def load_data():
    return pd.read_csv(DATA_PATH)

df = load_data()

# SIDEBAR
st.sidebar.title("📊 SOCIALPULSE")
st.sidebar.metric("Total Tweets", f"{len(df):,}")

# DASHBOARD STYLING
st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #f5f7fb;
    color: #243447;
}

/* MAIN HEADINGS */
.stApp h1,
.stApp h2,
.stApp h3 {
    color: #1f3c88 !important;
}

/* NORMAL TEXT */
.stApp p,
.stApp label {
    color: #243447 !important;
}

/* MAIN METRIC CARDS */
[data-testid="stMetric"] {
    background-color: white;
    padding: 15px;
    border-radius: 10px;
    box-shadow: 0 2px 6px rgba(0,0,0,0.08);
}

/* MAIN METRIC TEXT */
[data-testid="stMetricLabel"],
[data-testid="stMetricValue"],
[data-testid="stMetricDelta"] {
    color: #243447 !important;
}

/* SIDEBAR */
[data-testid="stSidebar"] {
    background-color: #262730;
}

/* SIDEBAR TITLE */
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2 {
    color: white !important;
}

/* SIDEBAR TEXT */
[data-testid="stSidebar"] p,
[data-testid="stSidebar"] label {
    color: white !important;
}

/* SIDEBAR METRIC - REMOVE WHITE BOX */
[data-testid="stSidebar"] [data-testid="stMetric"] {
    background-color: transparent !important;
    box-shadow: none !important;
    padding: 10px 0;
}

/* SIDEBAR METRIC TEXT */
[data-testid="stSidebar"] [data-testid="stMetricLabel"],
[data-testid="stSidebar"] [data-testid="stMetricValue"] {
    color: white !important;
}

/* TWEET INPUT BOX */
[data-testid="stTextArea"] textarea {
    background-color: #262730 !important;
    color: white !important;
    -webkit-text-fill-color: white !important;
    caret-color: white !important;
}

/* TWEET INPUT PLACEHOLDER */
[data-testid="stTextArea"] textarea::placeholder {
    color: #bfc7d5 !important;
    opacity: 1;
}

/* ANALYZE SENTIMENT BUTTON */
.stButton > button {
    background-color: #7c3aed !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}

.stButton > button:hover {
    background-color: #6d28d9 !important;
    color: white !important;
}

/* DOWNLOAD BUTTON */
.stDownloadButton > button {
    background-color: #059669 !important;
    color: white !important;
    border-radius: 8px;
    border: none;
    padding: 10px 20px;
}

.stDownloadButton > button:hover {
    background-color: #047857 !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)


# MAIN TITLE
st.title("📊 SOCIALPULSE")
st.subheader("Social Media Sentiment Analysis Dashboard")


# OVERALL DATASET OVERVIEW
st.markdown("### Overall Dataset Overview")

col1, col2, col3, col4 = st.columns(4)

col1.metric("Total Tweets", f"{len(df):,}")
col2.metric("Positive Tweets", f"{(df['Sentiment'] == 'Positive').sum():,}")
col3.metric("Negative Tweets", f"{(df['Sentiment'] == 'Negative').sum():,}")
col4.metric("Topics", f"{df['Topic'].nunique():,}")

st.metric("Model Accuracy", "96.6%")

st.divider()


# SENTIMENT DISTRIBUTION
st.markdown("### Sentiment Distribution")

sentiment_counts = df["Sentiment"].value_counts().reset_index()
sentiment_counts.columns = ["Sentiment", "Tweets"]

st.bar_chart(
    sentiment_counts,
    x="Sentiment",
    y="Tweets"
)


# SENTIMENT PIE CHART
st.markdown("### Sentiment Distribution Pie Chart")

st.vega_lite_chart(
    sentiment_counts,
    {
        "mark": {
            "type": "arc",
            "innerRadius": 0
        },
        "encoding": {
            "theta": {
                "field": "Tweets",
                "type": "quantitative"
            },
            "color": {
                "field": "Sentiment",
                "type": "nominal"
            },
            "tooltip": [
                {
                    "field": "Sentiment",
                    "type": "nominal"
                },
                {
                    "field": "Tweets",
                    "type": "quantitative"
                }
            ]
        }
    },
    use_container_width=True
)

st.divider()


# SENTIMENT COMPARISON CHART
st.markdown("### Sentiment Comparison")

sentiment_comparison = (
    df.groupby(["Topic", "Sentiment"])
    .size()
    .reset_index(name="Tweet Count")
)

st.bar_chart(
    sentiment_comparison,
    x="Topic",
    y="Tweet Count",
    color="Sentiment"
)

st.divider()


# TOP 10 TRENDING TOPICS
st.markdown("### Top 10 Trending Topics")

top_topics = df["Topic"].value_counts().head(10)

st.dataframe(
    top_topics.rename("Tweet Count").reset_index(),
    use_container_width=True
)

st.bar_chart(
    top_topics,
    horizontal=True
)

st.divider()


# TOPIC-WISE SENTIMENT HEATMAP
st.markdown("### Topic-wise Sentiment Heatmap")

heatmap_data = df.pivot_table(
    index="Topic",
    columns="Sentiment",
    values="Tweet_ID",
    aggfunc="count",
    fill_value=0
)

styled_heatmap = heatmap_data.style.background_gradient(
    cmap="YlGnBu"
).format("{:.0f}")

st.dataframe(
    styled_heatmap,
    use_container_width=True
)
# TOPIC-WISE SENTIMENT ANALYSIS
st.markdown("### Topic-wise Sentiment Analysis")

topics = sorted(df["Topic"].dropna().unique())

selected_topic = st.selectbox(
    "Select a topic",
    topics
)

topic_data = df[df["Topic"] == selected_topic]

topic_sentiment = (
    topic_data["Sentiment"]
    .value_counts()
    .reset_index()
)

topic_sentiment.columns = ["Sentiment", "Tweets"]

st.bar_chart(
    topic_sentiment,
    x="Sentiment",
    y="Tweets"
)

st.divider()


# EXPLORE TWEETS
st.markdown("### Explore Tweets")

selected_sentiment = st.selectbox(
    "Filter by sentiment",
    ["All"] + sorted(df["Sentiment"].dropna().unique().tolist())
)

if selected_sentiment == "All":
    filtered_data = df
else:
    filtered_data = df[df["Sentiment"] == selected_sentiment]

st.dataframe(
    filtered_data[["Topic", "Sentiment", "Tweet"]],
    use_container_width=True
)

st.divider()


# TEXT PREPROCESSING
def clean_text(text):
    text = str(text)
    text = text.lower()
    text = re.sub(r"http\S+|www\S+", "", text)
    text = re.sub(r"@\w+", "", text)
    text = re.sub(r"#", "", text)
    text = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


# LOAD TRAINED MODEL
@st.cache_resource
def load_model():
    model = joblib.load(
        MODEL_PATH / "best_sentiment_model.pkl"
    )

    vectorizer = joblib.load(
        MODEL_PATH / "best_tfidf_vectorizer.pkl"
    )

    return model, vectorizer


model, vectorizer = load_model()


# TWEET SENTIMENT PREDICTION
st.header("🤖 Tweet Sentiment Prediction")

st.write("Enter a tweet to predict its sentiment.")

user_tweet = st.text_area(
    "Enter your tweet",
    placeholder="Example: I absolutely love this game!"
)

if st.button("Analyze Sentiment"):

    if user_tweet.strip():

        cleaned_tweet = clean_text(user_tweet)

        transformed_tweet = vectorizer.transform(
            [cleaned_tweet]
        )

        prediction = model.predict(
            transformed_tweet
        )[0]

        st.success(
            f"Predicted Sentiment: {prediction}"
        )

    else:

        st.warning(
            "Please enter a tweet first."
        )


# DOWNLOAD SENTIMENT ANALYSIS REPORT
st.header("📥 Download Sentiment Analysis Report")

report = (
    df.groupby("Sentiment")
    .size()
    .reset_index(name="Tweet Count")
)

csv = report.to_csv(
    index=False
).encode("utf-8")

st.download_button(
    label="Download Sentiment Report",
    data=csv,
    file_name="sentiment_analysis_report.csv",
    mime="text/csv"
)


# FOOTER
st.caption("SOCIALPULSE")