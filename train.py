"""Run this module only once to train the model and save as a pickle"""
import pandas as pd
import pickle
from sklearn.decomposition import NMF

RATINGS = pd.read_csv('ratings.dat', delimiter='::', names=['userId', 'movieId', 'rating', 'timestamp'], engine='python')
MOVIES = pd.read_csv('movies.dat', delimiter='::', names=['movieId', 'title', 'genres'], engine='python')

def join_df(df1, df2, index):
    """Joins one dataframe to the other on a given basis (df1 on the left), then removes any additional rows created"""
    df = df1.join(df2.set_index(index), how='right', on=index)
    df = df.dropna()
    return df

def pivot(df, rows, columns, values, empty_val):
    """Pivots a dataframe, allowing us to select which category is desired as rows, columns and values of a new dataframe.
    Also imputes missing values with a given value"""
    R = df.groupby([rows, columns])[[values]].first().unstack()
    R.fillna(empty_val, inplace=True)
    return R

def train(matrix):
    # We define the model, which produces our matrices whose dot product should approximate R. We give these matrices an *n-components* value of 3, meaning 3 hidden features:
    nmf = NMF(n_components=3, init='random', random_state=42)
    nmf.fit(matrix)
    # store the model in a pickle and then reload it so that we don't have to retrain
    binary = pickle.dumps(nmf)
    open('nmf_model.bin', 'wb').write(binary)
    print('Model trained and saved successfully!')

if __name__ == '__main__':

    # We join the movies dataframe onto the ratings one in order to get the movie information for each rating:
    df = join_df(RATINGS, MOVIES, 'movieId')
    # We start by running the 'pivot' function to take a dataframe and put it into the desired shape - in this case users as rows, movies as columns, ratings as values. We also impute the empty values with 3, the median possible rating:
    R = pivot(df, 'userId', 'movieId', 'rating', 0.0)
    # lookup tables between movieId and title
    train(R)
