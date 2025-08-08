# app.py
import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="Tips Dataset EDA",
    page_icon="📊",
    layout="wide"
)

# ----------------- HEADER -----------------
st.title("📊 Exploratory Data Analysis — Tips Dataset")
st.markdown("**By: Shah Ahmad — Data Scientist**")
st.caption("This project was built with the assistance of ChatGPT")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# ----------------- DATASET INFO -----------------
st.subheader("📄 Dataset Preview")
st.write(df.head())

# Show dataset shape
st.markdown(f"**Rows:** {df.shape[0]} | **Columns:** {df.shape[1]}")

# ----------------- DESCRIPTIVE STATS -----------------
st.subheader("📈 Descriptive Statistics")
st.write(df.describe())

# ----------------- DATA VISUALIZATIONS -----------------
st.subheader("📊 Data Visualizations")

# Sidebar filters
st.sidebar.header("Filter Options")
day_filter = st.sidebar.multiselect("Select Day(s):", df['day'].unique(), default=df['day'].unique())
sex_filter = st.sidebar.multiselect("Select Gender(s):", df['sex'].unique(), default=df['sex'].unique())

filtered_df = df[(df['day'].isin(day_filter)) & (df['sex'].isin(sex_filter))]

# Total Bill vs Tip Scatterplot
st.markdown("### 💵 Total Bill vs Tip")
fig, ax = plt.subplots()
sns.scatterplot(x="total_bill", y="tip", hue="sex", size="size", data=filtered_df, ax=ax)
st.pyplot(fig)

# Average Tip by Day
st.markdown("### 📅 Average Tip by Day")
fig, ax = plt.subplots()
sns.barplot(x="day", y="tip", data=filtered_df, estimator="mean", ci=None, ax=ax)
st.pyplot(fig)

# Correlation Heatmap
st.markdown("### 🔥 Correlation Heatmap")
fig, ax = plt.subplots()
sns.heatmap(filtered_df.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
st.pyplot(fig)

# ----------------- CONCLUSION -----------------
st.subheader("📌 Insights")
st.markdown("""
- **Saturday & Sunday** tend to have higher total bills and tips — possibly due to weekend dining.
- Male customers generally tip slightly more than females in this dataset.
- Total bill has a **positive correlation** with tip amount.
""")

# ----------------- FOOTER -----------------
st.markdown("---")
st.markdown("💡 *EDA helps us understand patterns, trends, and relationships in the data before building any model.*")
