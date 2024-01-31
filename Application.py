import pickle
import streamlit as st
from sklearn.metrics.pairwise import cosine_similarity
from PIL import Image

# Load data outside the function
df1 = pickle.load(open('movie_list.pkl', 'rb'))
tfidf_matrix = pickle.load(open('tfidf_matrix.pkl', 'rb'))

@st.cache
def get_recommendation(title, df, tfidf):
    try:
        idx = df.index[df['title'] == title].tolist()[0]
    except IndexError:
        return None, None, None, None

    poster = f'https://image.tmdb.org/t/p/w500/{df.loc[idx, "poster_path"]}'
    sim_scores = cosine_similarity(tfidf, tfidf[idx]).flatten()
    movie_indices = sim_scores.argsort()[::-1][1:13]  # Get indices of top matches

    recommended_movies = df.iloc[movie_indices]
    movie_details = [(movie.title, 
                      f'https://image.tmdb.org/t/p/w500/{movie.poster_path}', 
                      movie.overview) for movie in recommended_movies.itertuples()]

    return poster, movie_details

def display_movies(movie_details, cols):
    for i, (title, poster, overview) in enumerate(movie_details):
        with cols[i % 4]:
            st.image(poster)
            st.markdown(title)
            with st.expander("Overview"):
                st.write(overview)

# UI setup
image = Image.open('Movie recommender system.jpg')
st.image(image)
st.markdown('This Project was uses machine learning to...')

movies_list = df1['title'].values
selected_movie = st.selectbox('Type and Choose The Movie', movies_list)

if st.button('Show Recommendation'):
    poster, movie_details = get_recommendation(selected_movie, df1, tfidf_matrix)
    if poster and movie_details:
        st.image(poster, width=160)
        cols = st.columns(4)
        display_movies(movie_details, cols)
    else:
        st.error("Movie not found. Please try another title.")
