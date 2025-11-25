import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Streamlit Page Setup
st.set_page_config(page_title="Movie Ratings Dashboard", layout="wide")

# Title
st.title("Movie Ratings Analysis Dashboard")
st.write("Interactive dashboard built with Streamlit + Seaborn")

# Load Data
movies = pd.read_csv("//Users/mahidharreddy/Downloads/Movie-Rating.csv")

st.subheader("Dataset Preview")
st.dataframe(movies.head())

# Sidebar
st.sidebar.header("Filters")

all_genres = movies["Genre"].unique()
selected_genre = st.sidebar.selectbox("Select Genre", ["All"] + list(all_genres))

# Filter Data
if selected_genre != "All":
    filtered = movies[movies["Genre"] == selected_genre]
else:
    filtered = movies

st.subheader(f"Showing Data For Genre: {selected_genre}")
st.dataframe(filtered)

# Distribution Plot
st.subheader("Audience Ratings Distribution")

fig1, ax1 = plt.subplots(figsize=(8,5))
sns.histplot(filtered["AudienceRating"], kde=True, bins=20, ax=ax1)
st.pyplot(fig1)

# Rotten Tomatoes Distribution
st.subheader("CriticsRating")

fig2, ax2 = plt.subplots(figsize=(8,5))
sns.histplot(filtered["CriticsRating"], kde=True, bins=20, ax=ax2)
st.pyplot(fig2)

# Countplot of Genres
st.subheader("Genre Counts (Full Dataset)")

fig3, ax3 = plt.subplots(figsize=(10,5))
sns.countplot(data=movies, x="Genre", ax=ax3)
plt.xticks(rotation=45)
st.pyplot(fig3)

# KDE by Genre
st.subheader("Audience Rating Distribution by Genre (KDE)")

fig4, ax4 = plt.subplots(figsize=(10,6))
sns.kdeplot(data=movies, x="AudienceRating", hue="Genre", fill=True, ax=ax4)
st.pyplot(fig4)

# Relationship Plot
st.subheader("Relationship: Rotten Tomatoes vs Audience Rating")

fig5, ax5 = plt.subplots(figsize=(8,5))
sns.regplot(data=movies, x="CriticsRating", y="AudienceRating", ax=ax5)
st.pyplot(fig5)

st.success("Dashboard Loaded Successfully")
