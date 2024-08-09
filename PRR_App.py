import numpy as np
import pandas as pd
import pickle

from flask import Flask, request, jsonify

from keras.preprocessing.sequence import pad_sequences
from keras.models import load_model

import mlflow

path_base_artifacts = '/teamspace/studios/this_studio/Personalized_Recipe_Recommender/artifacts/'
tokenizer_path = path_base_artifacts + 'tokenizer_info.pickle'

def load_tokenizer():
    # Load the pickle file
    with open(tokenizer_path, 'rb') as handle:
        tokenizer_info_data = pickle.load(handle)

    # Extract the tokenizer and max_sequence_length
    tokenizer = tokenizer_info_data['tokenizer']
    max_sequence_length = tokenizer_info_data['max_sequence_length']

    return max_sequence_length, tokenizer

def load_logged_model():

    # RUN_ID = beb560dccf2d49cdaa0efb87591b20f7

    # Inference after loading the logged model
    # model_uri = "runs:/{}/model".format(run.info.run_id)
    # loaded_model = mlflow.pytorch.load_model(model_uri)
    # Load the trained model
    logged_model = 'runs:/beb560dccf2d49cdaa0efb87591b20f7/model'

    # Load model as a PyFuncModel.
    loaded_model = mlflow.pyfunc.load_model(logged_model)

    return loaded_model


# Create the Flask app
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'hello world'

@app.route('/recommend/', methods=['POST'])
def recommend():
    # Get user input from request
    user_ingredients = request.json.get('ingredients', '')
    user_ingredients_list = [ingredient.strip() for ingredient in user_ingredients.split(',')]
    user_ingredients_str = ' '.join(user_ingredients_list)
    
    max_sequence_length, tokenizer = load_tokenizer()
    # Tokenize and pad the user ingredients
    user_sequences = tokenizer.texts_to_sequences([user_ingredients_str])
    user_padded = pad_sequences(user_sequences, maxlen=max_sequence_length)
    
    # Predict similarity scores using the model
    # user_predictions = model.predict(user_padded)
    loaded_model = load_logged_model()
    # Predict on a Pandas DataFrame.
    user_predictions = loaded_model.predict(user_padded)
    
    # Flatten the predictions if needed
    user_predictions_flattened = user_predictions.flatten()
    
    # Get the indices of the top 5 most similar recipes
    top_indices = user_predictions_flattened.argsort()[-5:][::-1]
    
    # Return the top 5 recommended recipes (indices or any other identifying information)
    return jsonify({
        'recommended_recipe_indices': top_indices.tolist()
    })

if __name__ == '__main__':
    app.run(debug=True)
