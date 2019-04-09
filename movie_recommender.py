"""Movie recommender

Takes a user's favourite movies from a web interface and returns recommendations"""


# from sklearn.decomposition import NMF
import pickle
import numpy as np
import pandas as pd
from fuzzywuzzy import process
from train import join_df, pivot, RATINGS, MOVIES
# import server

# lookup tables between movieId and title
LOOKUPTABLE = dict(MOVIES[['movieId', 'title']].groupby('movieId')['title'].first())
LOOKUPTABLE_2 = dict(MOVIES[['title', 'movieId']].groupby('title')['movieId'].first())

def get_movie_ids(entries):
    """Takes in user's movie choices and returns their ID's"""
    titles = list(MOVIES['title'])
    movie = LOOKUPTABLE_2[process.extractOne(entries, titles)[0]]
    return movie

def recommend(ratings_df, movies_df, query):
    """Main function - takes in user's entered movies and produces recommendations"""
    # load the trained model
    binary = open('nmf_model.bin', 'rb').read()
    nmf = pickle.loads(binary)

    # We join the movies dataframe onto the ratings one in order to get the
    # movie information for each rating:
    main_df = join_df(ratings_df, movies_df, 'movieId')
    # We start by running the 'pivot' function to take a dataframe and put it
    # into the desired shape - in this case users as rows, movies as columns,
    # ratings as values. We also impute the empty values with 0:
    sparse = pivot(main_df, 'userId', 'movieId', 'rating', 0.0)

    # We create a profile for a new user:
    # Firstly we generate a list of all movie ID's:
    movie_ids = list(sparse['rating'].columns)

    # Then we generate an "empty" set of selections for the new user:
    profile = {movie: np.nan for movie in movie_ids}

    # Then we assign some ratings:
    profile[get_movie_ids(query[0])] = 5.0
    profile[get_movie_ids(query[1])] = 5.0
    profile[get_movie_ids(query[2])] = 5.0

    new_user_input = list(profile.values())
    new_user_input = pd.DataFrame(new_user_input, index=movie_ids).T.fillna(0.0)

    # We produce the "hidden" profile based on our new user input:
    hidden_profile = nmf.transform(new_user_input)

    # And reconstruct to get our new user's predicted ratings for all movies:
    ypred = np.dot(hidden_profile, nmf.components_)

    # Now we wish to output the best-ranking movies that the user has not
    # rated yet - our recommendations!
    df_ratings = pd.DataFrame(ypred.T, columns=['rating'], index=movie_ids)
    df_ratings['profile'] = profile.values()

    # We need to get movieId into the dataframe:
    df_ratings_id = df_ratings[df_ratings['profile'].isna()].sort_values(by='rating', ascending=False)
    df_ratings_id['movieId'] = df_ratings_id.index
    df_ratings_id['title'] = df_ratings_id['movieId'].map(lambda x: LOOKUPTABLE[x])

    recos = list(df_ratings_id['title'].head(10).values)
    return recos


if __name__ == "__main__":
    MOVIE1 = input("Enter movie 1:")
    MOVIE2 = input("Enter movie 2:")
    MOVIE3 = input("Enter movie 3:")
    RESULT = recommend(RATINGS, MOVIES)
    print(RESULT)
