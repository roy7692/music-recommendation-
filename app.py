import pickle
import streamlit as st
import spotipy
from spotipy.oauth2 import SpotifyClientCredentials
from typing import List, Tuple, Dict
import random
import time
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics.pairwise import cosine_similarity

# --- Page config and styling ---
st.set_page_config(
    page_title="🎵 BeatMatch Recommender", 
    page_icon="🎧", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with animations and modern styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Montserrat:wght@400;600;700&family=Poppins:wght@300;400;500;600&display=swap');
    
    * {
        font-family: 'Poppins', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 20px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 10px 30px rgba(0,0,0,0.2);
        animation: fadeIn 1s ease-in;
    }
    
    .title {
        font-family: 'Montserrat', sans-serif;
        font-size: 3rem;
        font-weight: 700;
        color: white;
        margin: 0;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .subtitle {
        color: rgba(255,255,255,0.9);
        font-size: 1.2rem;
        margin-top: 0.5rem;
        font-weight: 300;
    }
    
    .stats-card {
        background: white;
        border-radius: 12px;
        padding: 1.5rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.1);
        border-left: 4px solid #667eea;
        margin-bottom: 1rem;
    }
    
    .stats-value {
        font-size: 2rem;
        font-weight: 700;
        color: #667eea;
        margin: 0;
    }
    
    .stats-label {
        font-size: 0.9rem;
        color: #666;
        margin: 0;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .metric-badge {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 0.3rem 0.8rem;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-block;
        margin: 0.2rem;
    }
    
    .similarity-score {
        background: #e8f4fd;
        padding: 0.5rem;
        border-radius: 8px;
        margin: 0.5rem 0;
        text-align: center;
    }
    
    .feature-bar {
        height: 8px;
        background: #e0e0e0;
        border-radius: 4px;
        margin: 0.3rem 0;
        overflow: hidden;
    }
    
    .feature-fill {
        height: 100%;
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        border-radius: 4px;
    }
    
    .pulse {
        animation: pulse 2s infinite;
    }
    
    @keyframes pulse {
        0% { transform: scale(1); }
        50% { transform: scale(1.05); }
        100% { transform: scale(1); }
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(-20px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    .card {
        border-radius: 16px;
        box-shadow: 0 8px 25px rgba(0,0,0,0.1);
        padding: 0;
        text-align: center;
        background: white;
        transition: all 0.3s ease;
        border: 1px solid rgba(255,255,255,0.2);
        height: 100%;
        display: flex;
        flex-direction: column;
        overflow: hidden;
    }
    
    .card:hover {
        transform: translateY(-10px);
        box-shadow: 0 15px 35px rgba(0,0,0,0.15);
    }
    
    .card-content {
        padding: 1.5rem;
        flex-grow: 1;
        display: flex;
        flex-direction: column;
        justify-content: space-between;
    }
    
    .album-img {
        border-radius: 0;
        width: 100%;
        height: 250px;
        object-fit: cover;
        transition: all 0.3s ease;
    }
    
    .album-img:hover {
        transform: scale(1.03);
    }
    
    .song-title {
        font-weight: 600;
        margin-top: 0;
        margin-bottom: 0.5rem;
        font-size: 1.1rem;
        color: #2d3748;
        line-height: 1.3;
    }
    
    .artist-name {
        color: #718096;
        margin: 0;
        font-size: 0.9rem;
        line-height: 1.2;
    }
    
    .spotify-btn {
        background: #1DB954;
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        font-weight: 600;
        margin-top: 1rem;
        cursor: pointer;
        transition: all 0.3s ease;
        text-decoration: none;
        display: inline-block;
        width: 100%;
        text-align: center;
    }
    
    .spotify-btn:hover {
        background: #1ed760;
        transform: scale(1.05);
        color: white;
        text-decoration: none;
    }
    
    .selected-song {
        background: linear-gradient(135deg, #ffecd2 0%, #fcb69f 100%);
        padding: 1.5rem;
        border-radius: 15px;
        margin: 1rem 0;
        border-left: 5px solid #667eea;
    }
    
    .recommendation-counter {
        background: #667eea;
        color: white;
        padding: 0.5rem 1rem;
        border-radius: 50px;
        display: inline-block;
        margin-bottom: 1rem;
        font-weight: 600;
    }
    
    .music-notes {
        font-size: 1.5rem;
        margin: 0 0.5rem;
    }
    
    .stProgress > div > div > div > div {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
    }
    
    .sidebar .sidebar-content {
        background: linear-gradient(180deg, #f8f9fa 0%, #e9ecef 100%);
    }
    
    /* Remove default Streamlit spacing */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    
    .st-emotion-cache-1y4p8pa {
        padding: 2rem 1rem;
    }
    
    /* Audio player styling */
    .stAudio {
        margin-top: 0.5rem;
        margin-bottom: 0;
    }
    
    .audio-preview {
        margin: 0.5rem 0 0 0;
    }
    
    .no-preview {
        color: #999;
        font-size: 0.8rem;
        margin: 0.5rem 0 0 0;
        font-style: italic;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# --- Spotify credentials ---
CLIENT_ID = "70a9fb89662f4dac8d07321b259eaad7"
CLIENT_SECRET = "4d6710460d764fbbb8d8753dc094d131"

# Initialize SpotiPy client
@st.cache_resource
def init_spotify(client_id: str, client_secret: str) -> spotipy.Spotify:
    client_credentials_manager = SpotifyClientCredentials(client_id=client_id, client_secret=client_secret)
    return spotipy.Spotify(client_credentials_manager=client_credentials_manager)

sp = init_spotify(CLIENT_ID, CLIENT_SECRET)

# --- Enhanced helper functions with statistics ---
@st.cache_data
def get_song_album_cover_and_link(song_name: str, artist_name: str) -> Tuple[str, str, str]:
    """Return (album_cover_url, spotify_track_url, preview_url)."""
    search_query = f"track:{song_name} artist:{artist_name}"
    try:
        results = sp.search(q=search_query, type="track", limit=1)
        items = results.get("tracks", {}).get("items", [])
        if items:
            track = items[0]
            album_cover_url = track["album"]["images"][0]["url"] if track["album"]["images"] else ""
            spotify_url = track.get("external_urls", {}).get("spotify", "")
            preview_url = track.get("preview_url", "")
            return album_cover_url or "https://i.postimg.cc/0QNxYz4V/social.png", spotify_url, preview_url
    except Exception as e:
        print("Spotify lookup failed:", e)
    return "https://i.postimg.cc/0QNxYz4V/social.png", "", ""

def recommend_with_stats(song: str, music_df, similarity_matrix, n_recs: int = 5) -> Tuple[List[str], List[str], List[str], List[str], List[float], Dict]:
    """Return lists: song names, album image urls, spotify urls, preview urls, similarity scores, and statistics."""
    try:
        index = music_df[music_df['song'] == song].index[0]
    except IndexError:
        return [], [], [], [], [], {}

    distances = sorted(list(enumerate(similarity_matrix[index])), reverse=True, key=lambda x: x[1])
    recommended_names, recommended_images, recommended_links, preview_urls, similarity_scores = [], [], [], [], []

    for i in distances[1:n_recs+1]:
        idx = i[0]
        similarity_score = i[1]
        row = music_df.iloc[idx]
        artist = row['artist']
        title = row['song']
        img_url, spotify_url, preview_url = get_song_album_cover_and_link(title, artist)
        recommended_names.append(f"{title}")
        recommended_images.append(img_url)
        recommended_links.append(spotify_url)
        preview_urls.append(preview_url)
        similarity_scores.append(similarity_score)

    # Calculate statistics
    stats = calculate_recommendation_stats(index, distances, music_df, n_recs)
    
    return recommended_names, recommended_images, recommended_links, preview_urls, similarity_scores, stats

def calculate_recommendation_stats(seed_index: int, distances: List, music_df, n_recs: int) -> Dict:
    """Calculate comprehensive statistics about the recommendations."""
    stats = {}
    
    # Basic similarity stats
    similarity_scores = [dist[1] for dist in distances[1:n_recs+1]]
    stats['avg_similarity'] = np.mean(similarity_scores)
    stats['max_similarity'] = np.max(similarity_scores)
    stats['min_similarity'] = np.min(similarity_scores)
    stats['similarity_std'] = np.std(similarity_scores)
    
    # Genre analysis
    seed_genres = set(music_df.iloc[seed_index].get('genre', '').split(', ') if 'genre' in music_df.columns else [])
    rec_genres = []
    for i in distances[1:n_recs+1]:
        idx = i[0]
        genres = set(music_df.iloc[idx].get('genre', '').split(', ') if 'genre' in music_df.columns else [])
        rec_genres.extend(list(genres))
    
    stats['genre_diversity'] = len(set(rec_genres)) / len(rec_genres) if rec_genres else 0
    stats['genre_overlap'] = len(seed_genres.intersection(set(rec_genres))) / len(seed_genres) if seed_genres else 0
    
    # Feature analysis (if features exist in dataframe)
    numeric_columns = music_df.select_dtypes(include=[np.number]).columns
    if len(numeric_columns) > 0:
        seed_features = music_df.iloc[seed_index][numeric_columns].values
        rec_features = []
        for i in distances[1:n_recs+1]:
            idx = i[0]
            rec_features.append(music_df.iloc[idx][numeric_columns].values)
        
        rec_features = np.array(rec_features)
        stats['feature_variance'] = np.mean(np.var(rec_features, axis=0))
    
    return stats

def get_model_statistics(music_df, similarity_matrix) -> Dict:
    """Get overall model statistics."""
    stats = {}
    
    stats['total_songs'] = len(music_df)
    stats['total_artists'] = music_df['artist'].nunique()
    
    # Calculate average similarity across all songs
    sample_size = min(100, len(music_df))
    sample_indices = random.sample(range(len(music_df)), sample_size)
    avg_similarities = []
    
    for idx in sample_indices:
        similarities = similarity_matrix[idx]
        # Get top 5 similarities (excluding self)
        top_similarities = sorted(similarities, reverse=True)[1:6]
        avg_similarities.extend(top_similarities)
    
    stats['avg_model_similarity'] = np.mean(avg_similarities) if avg_similarities else 0
    stats['model_confidence'] = stats['avg_model_similarity'] * 100  # Convert to percentage
    
    # Genre stats if available
    if 'genre' in music_df.columns:
        all_genres = []
        for genres in music_df['genre'].dropna():
            all_genres.extend([g.strip() for g in genres.split(',')])
        stats['total_genres'] = len(set(all_genres))
        stats['most_common_genre'] = max(set(all_genres), key=all_genres.count) if all_genres else "Unknown"
    else:
        stats['total_genres'] = "N/A"
        stats['most_common_genre'] = "N/A"
    
    return stats

# --- Load data ---
@st.cache_data
def load_data() -> Tuple:
    music = pickle.load(open('df.pkl','rb'))
    similarity = pickle.load(open('similarity.pkl','rb'))
    return music, similarity

music, similarity = load_data()

# --- Calculate model statistics ---
model_stats = get_model_statistics(music, similarity)

# --- Header with animated elements ---
st.markdown(
    """
    <div class="main-header pulse">
        <h1 class="title">🎵 BeatMatch Recommender</h1>
        <p class="subtitle">Discover your next favorite song with AI-powered recommendations</p>
        <div>
            <span class="music-notes">🎸</span>
            <span class="music-notes">🎹</span>
            <span class="music-notes">🎤</span>
            <span class="music-notes">🥁</span>
            <span class="music-notes">🎧</span>
        </div>
    </div>
    """,
    unsafe_allow_html=True
)

# --- Sidebar with enhanced UI ---
with st.sidebar:
    st.markdown("## 🎛️ Control Panel")
    
    # Song selection with search
    music_list = music['song'].values
    selected_song = st.selectbox("**Search or select a song**", music_list, 
                                help="Start typing to find your favorite song")
    
    # Show current selection details
    if selected_song:
        song_artist = music[music['song'] == selected_song]['artist'].values[0]
        st.markdown(f"""
        <div class="selected-song">
            <strong>Selected:</strong><br>
            🎵 {selected_song}<br>
            👤 {song_artist}
        </div>
        """, unsafe_allow_html=True)
    
    st.markdown("---")
    
    # Recommendation settings
    st.markdown("### ⚙️ Settings")
    n_recs = st.slider("**Number of recommendations**", min_value=3, max_value=12, value=6)
    
    # Additional filters
    st.markdown("### 🎚️ Filters")
    min_similarity = st.slider("**Minimum similarity score**", 0.0, 1.0, 0.5, 0.1)
    
    # Statistics display
    st.markdown("### 📊 Model Overview")
    st.markdown(f"""
    <div class="stats-card">
        <p class="stats-value">{model_stats['total_songs']}</p>
        <p class="stats-label">Songs in Database</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-card">
        <p class="stats-value">{model_stats['total_artists']}</p>
        <p class="stats-label">Unique Artists</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown(f"""
    <div class="stats-card">
        <p class="stats-value">{model_stats['model_confidence']:.1f}%</p>
        <p class="stats-label">Model Confidence</p>
    </div>
    """, unsafe_allow_html=True)
    
    st.markdown("---")
    st.markdown("*Powered by Spotify API & Machine Learning*")

# --- Main content area ---
col1, col2 = st.columns([1, 3])

with col1:
    st.markdown("### 🎯 Your Pick")
    if selected_song:
        # Get selected song details
        artist = music[music['song'] == selected_song]['artist'].values[0]
        img_url, spotify_url, preview_url = get_song_album_cover_and_link(selected_song, artist)
        
        st.image(img_url, use_container_width=True)
        st.markdown(f"**{selected_song}**")
        st.markdown(f"*by {artist}*")
        
        if spotify_url:
            st.markdown(f'<a href="{spotify_url}" target="_blank" class="spotify-btn">🎵 Open in Spotify</a>', 
                       unsafe_allow_html=True)

with col2:
    st.markdown("### 🚀 Ready to Discover?")
    
    # Animated recommendation button
    if st.button('🎧 Generate Recommendations', use_container_width=True, type="primary"):
        with st.spinner("🎶 Scanning the music universe for perfect matches..."):
            # Progress bar for visual feedback
            progress_bar = st.progress(0)
            for i in range(100):
                time.sleep(0.01)
                progress_bar.progress(i + 1)
            
            # Get recommendations with statistics
            rec_names, rec_images, rec_links, preview_urls, similarity_scores, rec_stats = recommend_with_stats(
                selected_song, music, similarity, n_recs
            )
        
        if not rec_names:
            st.error("❌ Couldn't find recommendations for this song. Try another one!")
        else:
            # Success message with counter
            st.markdown(f'<div class="recommendation-counter">🎉 Found {len(rec_names)} perfect matches!</div>', 
                       unsafe_allow_html=True)
            
            # Display recommendation statistics
            st.markdown("### 📈 Recommendation Insights")
            stat_col1, stat_col2, stat_col3, stat_col4 = st.columns(4)
            
            with stat_col1:
                st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-value">{rec_stats['avg_similarity']:.3f}</p>
                    <p class="stats-label">Avg Similarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col2:
                st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-value">{rec_stats['max_similarity']:.3f}</p>
                    <p class="stats-label">Max Similarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col3:
                st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-value">{rec_stats['min_similarity']:.3f}</p>
                    <p class="stats-label">Min Similarity</p>
                </div>
                """, unsafe_allow_html=True)
            
            with stat_col4:
                diversity_score = rec_stats.get('genre_diversity', 0)
                st.markdown(f"""
                <div class="stats-card">
                    <p class="stats-value">{diversity_score:.2f}</p>
                    <p class="stats-label">Genre Diversity</p>
                </div>
                """, unsafe_allow_html=True)
            
            # Similarity distribution visualization
            st.markdown("#### 📊 Similarity Distribution")
            fig, ax = plt.subplots(figsize=(10, 3))
            bars = ax.bar(range(len(similarity_scores)), similarity_scores, 
                         color=['#667eea', '#764ba2', '#f093fb', '#f5576c', '#4facfe', '#00f2fe'])
            ax.set_ylabel('Similarity Score')
            ax.set_xticks(range(len(similarity_scores)))
            ax.set_xticklabels([f'Rec {i+1}' for i in range(len(similarity_scores))])
            ax.set_ylim(0, 1)
            
            # Add value labels on bars
            for bar in bars:
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                       f'{height:.3f}', ha='center', va='bottom', fontweight='bold')
            
            st.pyplot(fig)
            
            # Display recommendations in responsive grid
            st.markdown("### 🎵 Your Recommendations")
            # Create appropriate number of columns based on screen size
            if len(rec_names) <= 3:
                cols = st.columns(len(rec_names))
            else:
                cols = st.columns(3)
                
            for idx, (name, img, link, preview, similarity_score) in enumerate(zip(
                rec_names, rec_images, rec_links, preview_urls, similarity_scores)):
                
                # For more than 3 items, we need to handle the column assignment
                if len(rec_names) > 3:
                    col_idx = idx % 3
                    if idx >= 3 and idx < 6:
                        # Create new row of columns for items 4-6
                        if idx == 3:
                            cols = st.columns(3)
                    elif idx >= 6:
                        # Create new row of columns for items 7+
                        if idx == 6:
                            cols = st.columns(3)
                else:
                    col_idx = idx
                    
                with cols[col_idx if len(rec_names) <= 3 else idx % 3]:
                    artist_name = music[music['song'] == name]['artist'].values
                    artist_text = artist_name[0] if len(artist_name) else "Unknown"
                    
                    # Card with no top white space
                    st.markdown('<div class="card">', unsafe_allow_html=True)
                    # Album cover at the top with no padding
                    st.image(img, use_container_width=True)
                    
                    # Content section
                    st.markdown('<div class="card-content">', unsafe_allow_html=True)
                    st.markdown(f'<p class="song-title">{name}</p>', unsafe_allow_html=True)
                    st.markdown(f'<p class="artist-name">{artist_text}</p>', unsafe_allow_html=True)
                    
                    # Similarity score
                    st.markdown(f'''
                    <div class="similarity-score">
                        <strong>Match Score:</strong> {similarity_score:.3f}
                        <div class="feature-bar">
                            <div class="feature-fill" style="width: {similarity_score * 100}%"></div>
                        </div>
                    </div>
                    ''', unsafe_allow_html=True)
                    
                    # Audio preview if available
                    if preview:
                        st.markdown('<div class="audio-preview">', unsafe_allow_html=True)
                        st.audio(preview, format="audio/mp3")
                        st.markdown('</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<p class="no-preview">No preview available</p>', unsafe_allow_html=True)
                    
                    # Spotify link
                    if link:
                        st.markdown(f'<a href="{link}" target="_blank" class="spotify-btn">🎵 Listen on Spotify</a>', 
                                  unsafe_allow_html=True)
                    else:
                        st.markdown('<p style="color: #999; font-size: 0.8rem; margin: 0;">Spotify link not available</p>', 
                                  unsafe_allow_html=True)
                    
                    st.markdown('</div>', unsafe_allow_html=True)  # Close card-content
                    st.markdown('</div>', unsafe_allow_html=True)  # Close card

# --- Model Performance Section ---
with st.expander("🔍 Model Performance & Analytics"):
    st.markdown("### 📈 System Performance Metrics")
    
    perf_col1, perf_col2, perf_col3 = st.columns(3)
    
    with perf_col1:
        st.metric("Database Coverage", f"{model_stats['total_songs']} songs")
        st.metric("Artist Diversity", f"{model_stats['total_artists']} artists")
    
    with perf_col2:
        st.metric("Average Similarity", f"{model_stats['avg_model_similarity']:.3f}")
        st.metric("Model Confidence", f"{model_stats['model_confidence']:.1f}%")
    
    with perf_col3:
        st.metric("Genre Coverage", f"{model_stats['total_genres']}")
        st.metric("Most Common Genre", model_stats['most_common_genre'])
    
    # Feature importance (if available in data)
    st.markdown("### 🎛️ Audio Features Analysis")
    if hasattr(music, 'columns'):
        audio_features = [col for col in music.columns if col not in ['song', 'artist', 'genre', 'link']]
        if audio_features:
            # Show feature distribution for a random song
            sample_song = random.choice(music['song'].values)
            sample_idx = music[music['song'] == sample_song].index[0]
            
            st.markdown(f"**Feature profile for: {sample_song}**")
            feature_data = {}
            for feature in audio_features[:8]:  # Show first 8 features
                if feature in music.columns:
                    try:
                        value = music.iloc[sample_idx][feature]
                        if isinstance(value, (int, float)):
                            feature_data[feature] = value
                    except:
                        continue
            
            if feature_data:
                features_df = pd.DataFrame({
                    'feature': list(feature_data.keys()),
                    'value': list(feature_data.values())
                })
                
                fig, ax = plt.subplots(figsize=(10, 4))
                bars = ax.barh(features_df['feature'], features_df['value'], 
                              color=plt.cm.viridis(np.linspace(0, 1, len(features_df))))
                ax.set_xlabel('Feature Value')
                ax.set_title('Audio Feature Profile')
                st.pyplot(fig)

# --- Fun interactive elements in expanders ---
with st.expander("🎲 Feeling Lucky?"):
    col1, col2, col3 = st.columns(3)
    with col1:
        if st.button("Random Song", use_container_width=True):
            random_song = random.choice(music_list)
            st.session_state.random_pick = random_song
            st.rerun()
    with col2:
        if st.button("Top Pick", use_container_width=True):
            # Simple heuristic for "top" song - you could replace with actual popularity metric
            st.info("🎯 Try popular songs like 'Shape of You' or 'Blinding Lights' for best results!")
    with col3:
        if st.button("Surprise Me!", use_container_width=True):
            st.balloons()
            st.success("🎉 Hope you discover something amazing!")

# Display random pick if selected
if 'random_pick' in st.session_state:
    st.info(f"🎲 Your random pick: **{st.session_state.random_pick}**")
    if st.button("Use this song for recommendations"):
        selected_song = st.session_state.random_pick
        st.rerun()

# --- Footer ---
st.markdown("---")
footer_col1, footer_col2, footer_col3 = st.columns(3)
with footer_col1:
    st.markdown("**How it works:**")
    st.markdown("""
    - ML analyzes song features
    - Finds similar audio patterns  
    - Fetches fresh data from Spotify
    - Provides similarity metrics
    """)
with footer_col2:
    st.markdown("**Pro tips:**")
    st.markdown("""
    - Check similarity scores
    - Use audio previews
    - Explore different genres
    - Save your favorites!
    """)
with footer_col3:
    st.markdown("**Model Metrics:**")
    st.markdown(f"""
    - {model_stats['total_songs']} songs
    - {model_stats['model_confidence']:.1f}% confidence
    - Cosine similarity algorithm
    """)

# Add some fun music-themed emojis
st.markdown("<br>", unsafe_allow_html=True)
st.markdown(
    "<div style='text-align: center; font-size: 1.5rem;'>🎸 🎹 🎤 🥁 🎧 🎼 🎵 🎶</div>", 
    unsafe_allow_html=True
)