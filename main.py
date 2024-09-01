import streamlit as st
import pickle
import pandas as pd
def recommend(movie):
    movie_index = movie1[movie1['title'] == movie].index[0]
    distance = similar_movie[movie_index]
    movie_list = sorted(list(enumerate(distance)), reverse=True, key=lambda x: x[1])[1:6]
    recommended_movies =[]
    for i in movie_list:
        recommended_movies.append(movie1.iloc[i[0]].title)
    return recommended_movies

movie_dict= pickle.load(open('movie_dictionary_list.pkl','rb'))
movie1 = pd.DataFrame(movie_dict)

similar_movie = pickle.load(open('simliar_movie_list.pkl','rb'))

st.title('Movie Recommender')

your_selected_movie = st.selectbox('Select the movie you like', movie1['title'].values)
st.write('Your selected movie :', your_selected_movie)

if st.button('similar movies to watch'):
    movie_title = recommend(your_selected_movie)

    col1, col2, col3, col4, col5 = st.columns(5, gap="large")

    with col1:
        st.subheader(movie_title[0])
        #st.image(movie_poster[0])
        st.markdown("&nbsp;" * 15)

    with col2:
        st.subheader(movie_title[1])
        #st.image(movie_poster[1])
        st.markdown("&nbsp;" * 15)

    with col3:
        st.subheader(movie_title[2])
        #st.image(movie_poster[3])
        st.markdown("&nbsp;" * 15)

    with col4:
        st.subheader(movie_title[3])
        #st.image(movie_poster[4])
        st.markdown("&nbsp;" * 15)

    with col5:
        st.subheader(movie_title[4])
        #st.image(movie_poster[5])
        st.markdown("&nbsp;" * 15)
