import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# Read CSV File

dataset = pd.read_csv('movie_dataset.csv')

# select features for finding correlations between movies

features = ['keywords', 'cast', 'genres', 'director']

# fill space string where there is na

for feature in features:
    dataset[feature] = dataset[feature].fillna(' ')

# creating a column which combines all selected features


def combine_features(row):
    return row['keywords'] + " " + row['cast'] + " " + row['genres'] + " " + row['director']


dataset["combined_features"] = dataset.apply(combine_features, axis=1)


# create the count matrix using combined features
cv = CountVectorizer()
count_matrix = cv.fit_transform(dataset["combined_features"])

# compute cosine similarity based on count matrix
cosine_similarity = cosine_similarity(count_matrix)

movie_user_likes = input("Enter movie name you like(e.g. 'Avatar'): ")

# get title from index


def get_index_from_title(title):
    try:
        return dataset[dataset.title == title]["index"].values[0]
    except:
        print("Movie not found try another name")


movie_index = get_index_from_title(movie_user_likes)
# get similar movies
similar_movies = list(enumerate(cosine_similarity[movie_index]))
sorted_similar_movies = sorted(similar_movies, key=lambda x: x[1], reverse=True)

# get title from index


def get_title_from_index(index):
    return dataset[dataset.index == index]["title"].values[0]


# print 10 similar movies
count = 0

print("Similar Movies:")
for movie in sorted_similar_movies:
    print(get_title_from_index(movie[0]))
    count = count+1
    if count > 10:
        break
