import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import io

# ----------------- PAGE CONFIG -----------------
st.set_page_config(
    page_title="EDA on Tips Dataset",
    page_icon="📊",
    layout="wide"
)

# ----------------- TITLE -----------------
st.markdown("<h1 style='text-align: center; color: #4CAF50;'>📊 EDA on Tips Dataset</h1>", unsafe_allow_html=True)
st.write("Welcome to an **interactive EDA** (Exploratory Data Analysis) dashboard built using Streamlit. "
         "This app lets you explore the classic Seaborn **Tips Dataset** in an engaging way.")

# ----------------- LOAD DATA -----------------
@st.cache_data
def load_data():
    return sns.load_dataset("tips")

df = load_data()

# ----------------- SIDEBAR -----------------
st.sidebar.header("🔍 Navigation")
section = st.sidebar.radio("Go to", ["About Dataset", "Data Preview", "EDA Summary", "Visualizations", "Correlation Heatmap"])

st.sidebar.subheader("⚙️ Filter Data")
days = st.sidebar.multiselect("Select Days", options=df['day'].unique(), default=df['day'].unique())
sex_filter = st.sidebar.multiselect("Select Gender", options=df['sex'].unique(), default=df['sex'].unique())
df_filtered = df[(df['day'].isin(days)) & (df['sex'].isin(sex_filter))]

# ----------------- ABOUT DATASET -----------------
if section == "About Dataset":
    st.subheader("📌 About Dataset")
    st.write("""
    The **Tips dataset** contains information about tips received by restaurant waiters.  
    It is commonly used for practicing **data visualization & statistical analysis**.

    **Features:**
    - `total_bill`: Total bill in USD.
    - `tip`: Tip given to the waiter in USD.
    - `sex`: Gender of the person paying the bill.
    - `smoker`: Whether the person was a smoker.
    - `day`: Day of the week.
    - `time`: Lunch or Dinner.
    - `size`: Number of people at the table.
    """)

# ----------------- DATA PREVIEW -----------------
elif section == "Data Preview":
    st.subheader("📋 Data Preview")
    st.dataframe(df_filtered)

    buffer = io.StringIO()
    df_filtered.info(buf=buffer)
    st.text("**Dataset Info:**")
    st.text(buffer.getvalue())

# ----------------- EDA SUMMARY -----------------
elif section == "EDA Summary":
    st.subheader("📈 EDA Summary Statistics")
    st.write(df_filtered.describe())

# ----------------- VISUALIZATIONS -----------------
elif section == "Visualizations":
    st.subheader("📊 Visualizations")

    col1, col2 = st.columns(2)

    with col1:
        st.write("💰 **Total Bill Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(df_filtered['total_bill'], kde=True, ax=ax, color='skyblue')
        st.pyplot(fig)

    with col2:
        st.write("💵 **Tip Distribution**")
        fig, ax = plt.subplots()
        sns.histplot(df_filtered['tip'], kde=True, ax=ax, color='orange')
        st.pyplot(fig)

    st.write("🍽 **Average Tip by Day**")
    fig, ax = plt.subplots()
    sns.barplot(data=df_filtered, x='day', y='tip', ci=None, palette="viridis")
    st.pyplot(fig)

# ----------------- CORRELATION -----------------
elif section == "Correlation Heatmap":
    st.subheader("🔗 Correlation Heatmap")
    fig, ax = plt.subplots(figsize=(8, 5))
    sns.heatmap(df_filtered.corr(numeric_only=True), annot=True, cmap="coolwarm", ax=ax)
    st.pyplot(fig)

# ----------------- FOOTER -----------------
st.markdown("<hr>", unsafe_allow_html=True)
st.markdown(
    "<p style='text-align: center; color: gray;'>"
    "Created with ❤️ by a Data Scientist & AI Engineer</p>",
    unsafe_allow_html=True
)
