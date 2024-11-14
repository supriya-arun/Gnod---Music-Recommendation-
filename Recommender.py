import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import os

# Load the song data
df = pd.read_csv('C:/Users/ngaut/Documents/Ironhack/Week10/spotify_final1.csv') 

# Clean the 'title' column to remove extra spaces
df['title'] = df['title'].str.strip()

# Select features for clustering
features = ['danceability', 'energy', 'valence', 'tempo']
X = df[features]

# Scale the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Apply KMeans clustering
kmeans = KMeans(n_clusters=2, random_state=42)
df['Cluster'] = kmeans.fit_predict(X_scaled)

# Define the recommendation function
def recommend_songs(input_song, df):
    # Clean the input song title for comparison (strip spaces and convert to lowercase)
    input_song_cleaned = input_song.strip().lower()
    
    # Find the cluster of the input song
    song_cluster = df[df['title'].str.lower() == input_song_cleaned]['Cluster'].values

    # If the song is not found, return None
    if len(song_cluster) == 0:
        return None  # Song not found
    song_cluster = song_cluster[0]  # Get the cluster of the input song
    
    # Recommend songs from the same cluster, excluding the input song itself
    recommended_songs = df[df['Cluster'] == song_cluster]
    recommended_songs = recommended_songs[recommended_songs['title'].str.lower() != input_song_cleaned]
    
    return recommended_songs[['title', 'artist', 'Cluster']].head(5) 

# Streamlit user interface
st.title("Song Recommender System")
image_path = os.path.join(os.getcwd(), 'Spotify_img.png')
st.image(image_path, width=700)
st.write("Enter a song name to get recommendations based on similarity:")

# Input song name from user
input_song = st.text_input("Song Name", "")

# Only make the recommendation when there's input
if input_song:
    # Clean and lower case the input song
    input_song_cleaned = input_song.strip().lower()

    # Check if the input song exists in the dataframe (case-insensitive check)
    if input_song_cleaned in df['title'].str.strip().str.lower().values:
        recommendations = recommend_songs(input_song, df)
        if recommendations is not None:
            st.write(f"Songs similar to '{input_song}':")
            st.write(recommendations)
        else:
            st.write("Song not found. Please try again with a different song name.")
    else:
        st.write("Song not found. Please try again with a different song name.")

# Now, display the global hits section after the recommendations
file_path = 'C:/Users/ngaut/Documents/Ironhack/Week10/df_global.csv'
df_global = pd.read_csv(file_path)

# Display the global hits message after recommendations
st.write("Check out what the globe grooves to:")

# Select the top 5 rows and only the 'title' and 'artist' columns
top_5_songs = df_global[['title', 'artist']].head(5)

# Display the top 5 songs
st.write(top_5_songs)
