import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource 
def load_models():
    #pipeline = joblib.load('svd_pipeline.joblib')
    # sim_matrix = joblib.load('cosine_sim_lsa.joblib')
    # means = joblib.load('joke_means.joblib')
    # return pipeline, sim_matrix, means
    pass

@st.cache_data 
def load_data():
    jokes_df = pd.read_csv('../Data/processed_jokes.csv')  
    return jokes_df

def get_next_best_joke(jokes_df):
    """
    Тук интегрираш твоя SVD модел. 
    1. Вземаш текущите оценки от st.session_state.user_ratings
    2. Предсказваш оценките за всички неоценени шеги
    3. Връщаш тази с най-висок предвиден резултат
    """
    all_joke_ids = jokes_df['JokeID'].tolist()
    unrated_jokes = [j for j in all_joke_ids if j not in st.session_state.rated_jokes]
    
    # Временно: избираме случайна от неоценените (тук сложи SVD логиката)
    return np.random.choice(unrated_jokes)

def main():
    jokes_df = load_data()
    gauge_set = [7, 8, 13, 15, 16, 17, 18, 19]

    if 'rated_jokes' not in st.session_state:
        st.session_state.rated_jokes = [] 
    if 'user_ratings' not in st.session_state:
        st.session_state.user_ratings = {} 
    if 'current_joke_id' not in st.session_state:
        st.session_state.current_joke_id = gauge_set[0]
    if 'rated_jokes_cnt' not in st.session_state:
        st.session_state.rated_jokes_cnt = 0
    if 'slider_key' not in st.session_state:
        st.session_state.slider_key = 0

    st.title("Jester: The Infinite Joke Stream")


    total_rated = st.session_state.rated_jokes_cnt
    st.sidebar.metric("Jokes Rated", total_rated)

    current_id = st.session_state.current_joke_id
    joke_text = jokes_df.loc[jokes_df['JokeID'] == current_id, 'Text'].values[0]

    st.markdown("---")
    st.subheader(f"Joke #{current_id}")
    st.info(joke_text.replace("<br>", "\n"))

    rating = st.select_slider(
        "Your Rating:",
        options=np.arange(-10.0, 10.5, 0.5).tolist(),
        value=0.0,
        key=st.session_state.slider_key
    )

    
    if st.button("Rate & Next"):
        
        st.session_state.user_ratings[current_id] = rating
        st.session_state.rated_jokes.append(current_id)
        st.session_state.rated_jokes_cnt += 1
        st.session_state.slider_key += 1

        if  st.session_state.rated_jokes_cnt < len(gauge_set):
            st.session_state.current_joke_id = gauge_set[st.session_state.rated_jokes_cnt]
        else:
            with st.spinner('Calculating your taste...'):
                st.session_state.current_joke_id = get_next_best_joke(jokes_df)
                
        st.rerun()

if __name__ == '__main__':
    main()