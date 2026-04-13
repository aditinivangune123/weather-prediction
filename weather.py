import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="Weather Pro App", layout="wide")

# -------------------------------
# 🌈 PREMIUM CSS
# -------------------------------
st.markdown("""
<style>

/* Background */
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #020617, #0f172a);
    color: white;
    font-family: 'Segoe UI', sans-serif;
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: linear-gradient(180deg, #020617, #1e293b);
}

/* Headings */
h1 {
    color: #38bdf8;
    text-align: center;
    font-size: 40px;
}
h2, h3 {
    color: #7dd3fc;
}

/* Text visibility */
p, label {
    color: #e2e8f0 !important;
    font-size: 16px;
}

/* Glass card effect */
.block-container {
    background: rgba(255, 255, 255, 0.05);
    padding: 2rem;
    border-radius: 20px;
}

/* Buttons */
.stButton>button {
    background: linear-gradient(90deg, #38bdf8, #0ea5e9);
    color: white;
    border-radius: 12px;
    height: 3em;
    font-size: 16px;
    border: none;
}

/* Input boxes */
input {
    background-color: #1e293b !important;
    color: white !important;
}

</style>
""", unsafe_allow_html=True)

# -------------------------------
# TITLE
# -------------------------------
st.title("🌦️ Weather Prediction Pro Dashboard")
st.markdown("### AI Powered Weather Intelligence System")

# -------------------------------
# SIDEBAR NAVIGATION
# -------------------------------
menu = st.sidebar.radio("📌 Navigation", ["Home", "Data Analysis", "Prediction"])

file = st.sidebar.file_uploader("Upload Dataset", type=["csv"])

# -------------------------------
# LOAD DATA
# -------------------------------
if file is not None:
    df = pd.read_csv(file)
else:
    df = pd.read_csv("weather.csv")

df = df.dropna()

if "Date" in df.columns:
    df = df.drop("Date", axis=1)

le = LabelEncoder()
df["Condition"] = le.fit_transform(df["Condition"])

X = df.drop("Condition", axis=1)
y = df["Condition"]

# -------------------------------
# MODEL
# -------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier()
model.fit(X_train, y_train)

accuracy = accuracy_score(y_test, model.predict(X_test))

# -------------------------------
# 🏠 HOME PAGE
# -------------------------------
if menu == "Home":

    st.subheader("📊 Model Performance")

    c1, c2, c3 = st.columns(3)

    c1.metric("Accuracy", f"{accuracy:.2f}")
    c2.metric("Training Data", len(X_train))
    c3.metric("Testing Data", len(X_test))

    st.markdown("---")

    st.subheader("📌 Dataset Preview")
    st.dataframe(df.head())

# -------------------------------
# 📊 DATA ANALYSIS PAGE
# -------------------------------
elif menu == "Data Analysis":

    st.subheader("📈 Visual Insights")

    col1, col2 = st.columns(2)

    with col1:
        plt.figure()
        sns.histplot(df['Temp_C'], kde=True)
        st.pyplot(plt)

    with col2:
        plt.figure()
        sns.boxplot(x=y, y=df['Humidity_%'])
        st.pyplot(plt)

    st.markdown("### 🔥 Correlation Heatmap")

    plt.figure(figsize=(8,5))
    sns.heatmap(df.corr(), cmap='coolwarm')
    st.pyplot(plt)

# -------------------------------
# 🌍 PREDICTION PAGE
# -------------------------------
elif menu == "Prediction":

    st.subheader("🌍 Predict Weather")

    input_cols = st.columns(len(X.columns))
    input_data = []

    for i, col in enumerate(X.columns):
        val = input_cols[i].number_input(col, value=0.0)
        input_data.append(val)

    if st.button("🔮 Predict Now"):
        prediction = model.predict([input_data])

        if prediction[0] == 1:
            st.error("🌧️ Rainy Weather Expected")
        else:
            st.success("☀️ Sunny Weather Expected")

    st.markdown("---")
    st.info("💡 Tip: Enter realistic weather values for better prediction")