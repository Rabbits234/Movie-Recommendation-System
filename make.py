import ast
import pandas as pd
import numpy as np


movies=pd.read_csv('tmdb_5000_movies 2.csv')
credits=pd.read_csv('tmdb_5000_credits 2.csv')
movies=movies.merge(credits,on='title')
movies=movies[['genres','id','title','cast','crew','overview','keywords']]
movies.dropna(inplace=True)
movies.duplicated().sum()
movies.iloc[0].genres
def convert(obj):
    l=[]
    for i in ast.literal_eval(obj):
        l.append(i['name'])
    return l
movies['genres']=movies['genres'].apply(convert)
movies['keywords']=movies['keywords'].apply(convert)
def convert2(obj):
    l=[]
    counter=0
    for i in ast.literal_eval(obj):
        if counter!=3:
            l.append(i['name'])
            counter+=1
        else:
            break
    return l
def fetch_director(obj):
    l=[]
    for i in ast.literal_eval(obj):
        if i['job']=='Director':
            l.append(i['name'])
            break
    return l
movies['cast']=movies['cast'].apply(convert2)
movies['crew']=movies['crew'].apply(fetch_director)
movies['overview']=movies['overview'].apply(lambda x:x.split())
movies['genres']=movies['genres'].apply(lambda x:[i.replace(" ","") for i in x])
movies['cast']=movies['cast'].apply(lambda x:[i.replace(" ","") for i in x])
movies['crew']=movies['crew'].apply(lambda x:[i.replace(" ","") for i in x])
movies['keywords']=movies['keywords'].apply(lambda x:[i.replace(" ","") for i in x])
movies['tags']=movies['overview']+movies['genres']+movies['cast']+movies['crew']+movies['keywords']
new_df=movies[['id','title','tags']]
new_df['tags']=new_df['tags'].apply(lambda x:" ".join(x))
new_df['tags']=new_df['tags'].apply(lambda x:x.lower())

from sklearn.feature_extraction.text import CountVectorizer

cv= CountVectorizer(max_features=5000,stop_words='english')
vectors=cv.fit_transform(new_df['tags']).toarray()
import nltk
from nltk.stem.porter import PorterStemmer

ps=PorterStemmer()
def stem(text):
    y=[]
    for i in text.split():
        y.append(ps.stem(i))
    return " ".join(y)
new_df['tags']=new_df['tags'].apply(stem)

from sklearn.metrics.pairwise import cosine_similarity

siimilarity=cosine_similarity(vectors)
def recomand(movie):
    movie_index=new_df[new_df['title']==movie].index[0]
    distances=siimilarity[movie_index]
    movies_list=sorted(list(enumerate(distances)),reverse=True,key=lambda x:x[1])[1:6]
    for i in movies_list:
        print(new_df.iloc[i[0]].title)
import pickle

pickle.dump(movies, open('movies.pkl', 'wb'))
pickle.dump(siimilarity, open('similarity.pkl', 'wb'))

print(new_df.head())
