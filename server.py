"""Flask server on which to run movie recommender"""
from flask import Flask, render_template, request
from movie_recommender import recommend
from train import RATINGS, MOVIES

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('main.html',
                           title="Welcome to the Dave Movie DataBase",
                           page_name='main page')


@app.route('/movies', methods=['POST', 'GET'])
def recommender():
    query = [request.args['movie1'],
             request.args['movie2'],
             request.args['movie3']]
    query_s = ", ".join(query)
    result = recommend(RATINGS, MOVIES, query)
    return render_template('recommendations.html',  # Jinja template
                           query=query_s,
                           movies=result)
