import pandas as pd 
import numpy as np
import sklearn as sk 

movies_data= pd.read_csv('/Users/tejusmacbookair/Downloads/movies.csv')

#movies_data.head() # provides the data present in movies_data file 


movies_data.head(1) #provides the first row of the data in movies_data file

movies_data_sliced = movies_data.iloc[:5000] #It gives frist 5000 rows values (from 0 to 4999) from the data
# I sliced the data to reduce dataload

#movies_data_sliced.head(1)['credits'].values
# to check format andtvvalues of specific column (here: credits)

#movies_data_sliced.shape
#To get the dimesions of the data structure (rows, col)

#movies_data['original_language'].value_counts()
# to get the value count of distinct values in an attribute 

movies_data_sliced = movies_data_sliced[['id','title','genres','overview','credits','keywords','poster_path']]
# conisdered only took reuired attributes 

movies_data_sliced.info() #to get detail count of each col values

movies_data_sliced.isnull().sum() # To get total no.of null values in each col

movies_data_sliced.dropna(inplace=True) # droping the null values

movies_data_sliced.duplicated().sum() #checking the duplicate values 

movies_data_sliced.drop_duplicates(inplace=True)#drop duplicates 

#movies_data_sliced.iloc[0].genres #checking the data format of geners

# string_to_list is the function converts string with (-) to list 
def string_to_list (obj):
    li = []
    li = obj.split('-')
    return li

#applying the string_to_list fucntion to genres,keywords,credits
movies_data_sliced['genres'] =movies_data_sliced['genres'].apply(string_to_list)
movies_data_sliced['keywords']=movies_data_sliced['keywords'].apply(string_to_list)
movies_data_sliced['credits']=movies_data_sliced['credits'].apply(string_to_list)

movies_data_sliced.iloc[0].keywords

#movies_data_sliced.iloc[0].credits

#movies_data_sliced.iloc[0].overview

movies_data_sliced['overview']

#below function is used to convert sentence to list where (' ')
def sentence_to_list (obj):
    li = obj.split()
    return li

#applied the sentence_to_list function 
movies_data_sliced['overview']=movies_data_sliced['overview'].apply(sentence_to_list)

#below function removes space in each element of the list
def replace_spaces_in_listelemnts(obj):
    li = [element.replace(' ', '') for element in obj]
    return li

#replace_space_in_listelemnts is applied all list types
movies_data_sliced['genres'] =movies_data_sliced['genres'].apply(replace_spaces_in_listelemnts)
movies_data_sliced['keywords'] =movies_data_sliced['keywords'].apply(replace_spaces_in_listelemnts)
movies_data_sliced['credits']=movies_data_sliced['credits'].apply(replace_spaces_in_listelemnts)
movies_data_sliced['overview']=movies_data_sliced['overview'].apply(replace_spaces_in_listelemnts)

movies_data_sliced.head(3)

#Combaining all the lists type attributes to concat_tags
movies_data_sliced['concat_tags'] = movies_data_sliced['genres'] + movies_data_sliced['keywords'] + movies_data_sliced['credits'] + movies_data_sliced['overview']

#movies_data_sliced.head(1)

#spliting the data for fetching the paster path
movies_poster_data = movies_data_sliced[['id','poster_path']]

#copying the data required into new_movie_data file 
new_movie_data = movies_data_sliced[['id', 'title', 'concat_tags']]

new_movie_data.head(1)

#below function is used to convert list to a string 
def list_to_string_sentence(obj):
    return ' '.join(obj)    

# alternative for above funtion
#new_movie_data['concat_tags']=new_movie_data['concat_tags'].apply(lambda x:" ".join(x))

new_movie_data['concat_tags'] = new_movie_data['concat_tags'].apply(list_to_string_sentence)

#new_movie_data['concat_tags'][0]

# convert everything to lower case 
def convert_to_lowercase(obj):
    return obj.lower()

new_movie_data['concat_tags'] = new_movie_data['concat_tags'].apply(convert_to_lowercase)

new_movie_data['concat_tags'][0]

import nltk
from nltk.stem.porter import PorterStemmer
p_s = PorterStemmer()

def stem(obj):
    li =[]
    for i in obj.split():
        li.append(p_s.stem(i))
    return " ".join(li)


from sklearn.feature_extraction.text import TfidfVectorizer
tfidf_v = TfidfVectorizer(max_features=15000, stop_words='english')

vectors = tfidf_v.fit_transform(new_movie_data['concat_tags']).toarray()

tfidf_v.fit_transform(new_movie_data['concat_tags']).toarray().shape

vectors #print the vector structure 

for element in tfidf_v.get_feature_names_out():
    print (element)


from sklearn.metrics.pairwise import cosine_similarity
#from sklearn.preprocessing import normalize

new_movie_data.isnull().any()# checks any null data present

has_nan = np.isnan(vectors).any() #checks for any null value in vector 
has_nan

has_inf = np.isinf(vectors).any() # checks for any infinte values in vector
has_inf

similarity_distance = cosine_similarity(vectors) # the distane between the each movie to 5000 other movies

similarity_distance.shape

[sum(v) for v in vectors if sum(v)==0.0] 

sorted(list(enumerate(similarity_distance[0])),reverse=True, key=lambda x:x[1]) [1:6]

# sorted desecding order and first five movies

def recommend(movie):
    movie_index = new_movie_data[new_movie_data['title'] == movie].index[0]
    distance = similarity_distance[movie_index]
    movie_list = sorted(list(enumerate(distance)),reverse=True, key=lambda x:x[1]) [1:6]
    
    for i in movie_list:
        print(new_movie_data.iloc[i[0]].title)

recommend("Barbie")

import pickle
#pickle.dump(new_movie_data.open('movietitles.pkl','wb') )
# Assuming `new_movie_data` is your Pandas DataFrame containing movie data
with open('movietitles.pkl', 'wb') as file:
    pickle.dump(new_movie_data, file)


with open('movie_dictionary_list.pkl', 'wb') as file:
    pickle.dump(new_movie_data.to_dict(orient='records'), file)


with open('simliar_movie_list.pkl', 'wb') as file:
    pickle.dump(similarity_distance, file)

with open('movies_poster_data.pkl', 'wb') as file:
    pickle.dump(movies_poster_data, file)

    