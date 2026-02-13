import streamlit as st
import pandas as pd
import numpy as np
import joblib


@st.cache_resource 
def load_models():
    pipeline = joblib.load('../models/svd_final.pkl')
    sim_matrix = joblib.load('../models/item_similarity.pkl')
    return pipeline, sim_matrix

@st.cache_data 
def load_data():
    jokes_df = pd.read_csv('../Data/processed_jokes.csv')  
    return jokes_df

def center_user_ratings(X):
    user_means = np.nanmean(X, axis=1).reshape(-1, 1)
    return X - user_means

def get_next_best_joke(jokes_df, pipeline):
    
    model_input = st.session_state.user_vector[1:].reshape(1, -1)   
    user_mean = np.nanmean(model_input)
    if np.isnan(user_mean): user_mean = 0.0
    
    user_emb = pipeline.transform(model_input) 
    vt = pipeline.named_steps['svd'].components_
    preds_scaled = np.dot(user_emb, vt)
    
    scaler = pipeline.named_steps['scaler']
    preds_centered = scaler.inverse_transform(preds_scaled)
    raw_predictions = (preds_centered + user_mean).flatten()

    predictions = np.concatenate(([-100], raw_predictions))
    all_joke_ids = jokes_df['JokeID'].tolist()    
    unrated_jokes = [j for j in all_joke_ids if j not in st.session_state.rated_jokes]
    
    best_joke = None
    best_score = -999
    for joke_id in unrated_jokes:
        score = predictions[joke_id]
        if score > best_score:
            best_score = score
            best_joke = joke_id
    
    return best_joke

def get_recommendations(joke_id, jokes_df, sim_matrix, top_n=5):
    if joke_id < 1 or joke_id > 150:
        print("Invalid Joke Id")
        return

    idx = jokes_df.index.get_loc(joke_id)
    sim_scores = list(enumerate(sim_matrix[idx]))
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)[1:top_n + 1]

    indices = [i[0] for i in sim_scores]
    return jokes_df.iloc[indices][['Text']]

def main():
    jokes_df = load_data()
    pipeline, sim_matrix = load_models()
    gauge_set = [7, 8, 13, 15, 16, 18, 19]

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
    if 'user_vector' not in st.session_state:
        st.session_state.user_vector = np.full(151, np.nan)

    st.title("Jester: The Infinite Joke Stream")

    total_rated = st.session_state.rated_jokes_cnt
    st.sidebar.metric("Jokes Rated", total_rated)

    current_id = st.session_state.current_joke_id
    joke_text = jokes_df.loc[jokes_df['JokeID'] == current_id, 'Text'].values[0]

    st.markdown("---")
    st.subheader(f"Joke #{current_id}")
    st.info(joke_text)

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
                st.session_state.current_joke_id = get_next_best_joke(jokes_df, pipeline)
                
        st.rerun()

    st.markdown("---")
    st.subheader("You may also like")
    similar_jokes = get_recommendations(current_id, jokes_df, sim_matrix, top_n=3)
    if similar_jokes is not None and not similar_jokes.empty:
        for idx, row in similar_jokes.iterrows():
            st.info(row['Text'])
            st.markdown("")

if __name__ == '__main__':
    main()