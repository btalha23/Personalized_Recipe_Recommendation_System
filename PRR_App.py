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




path_base_artifacts = '/teamspace/studios/this_studio/Personalized_Recipe_Recommender/artifacts/'
# path_base_artifacts = 'artifacts/'
tokenizer_path = path_base_artifacts + 'tokenizer_info.pickle'

def load_tokenizer():
    # Load the pickle file
    with open(tokenizer_path, 'rb') as handle:
        tokenizer_info_data = pickle.load(handle)

    # Extract the tokenizer and max_sequence_length
    tokenizer = tokenizer_info_data['tokenizer']
    max_sequence_length = tokenizer_info_data['max_sequence_length']

    print(f"max_sequence_length {max_sequence_length}")

    return max_sequence_length, tokenizer

def predict_from_model(RUN_ID, model, original_dataframe):
    # Get user input from request
    # user_ingredients = request.json.get('ingredients', '')
    # user_ingredients = "chicken garlic onion salt pepper"
    user_ingredients = "Beets vinegar cloves sugar salt water"
    user_ingredients_list = [ingredient.strip() for ingredient in user_ingredients.split(',')]
    user_ingredients_str = ' '.join(user_ingredients_list)
    print(user_ingredients_str)
    
    max_sequence_length, tokenizer = load_tokenizer()
    # Tokenize and pad the user ingredients
    user_sequences = tokenizer.texts_to_sequences([user_ingredients_str])
    user_padded = pad_sequences(user_sequences, maxlen=max_sequence_length)
    print(f"user_padded_sequence {user_padded}")
    
    model_original = model
    # Predict on a Pandas DataFrame.
    user_predictions_original_model = model_original.predict(user_padded)
    print(f"user_predictions_original_model {user_predictions_original_model}")
    
    # Flatten the predictions if needed
    user_predictions_original_model_flattened = user_predictions_original_model.flatten()
    
    # Get the indices of the top 5 most similar recipes
    top_indices_original_model = user_predictions_original_model_flattened.argsort()[-5:][::-1]
    recommended_recipes = original_dataframe.iloc[top_indices_original_model]

    print(top_indices_original_model)
    print(recommended_recipes)

    # Predict similarity scores using the model
    # user_predictions = model.predict(user_padded)
    # loaded_model = load_logged_model()
    loaded_model = mlflow.keras.load_model(f"runs:/{RUN_ID}/model")
    print(f"loaded_model {loaded_model}")
    # Predict on a Pandas DataFrame.
    user_predictions = loaded_model.predict(user_padded)
    print(f"user_predictions {user_predictions}")
    
    # Flatten the predictions if needed
    user_predictions_flattened = user_predictions.flatten()
    print(user_predictions_flattened)
    
    # Get the indices of the top 5 most similar recipes
    top_indices = user_predictions_flattened.argsort()[-5:][::-1]
    recommended_recipes = original_dataframe.iloc[top_indices]

    print(top_indices)
    print(recommended_recipes)

    # Flatten predictions if necessary
    if user_predictions.ndim > 1:
        user_predictions = user_predictions.flatten()

    # Get the indices of the top 5 most similar recipes
    top_indices_test = user_predictions.argsort()[-5:][::-1]
    print(top_indices_test)
    
    recommended_recipes = original_dataframe.iloc[top_indices_test]
    print(recommended_recipes)

    return top_indices

def compare_with_all_recipes(predicted_similarity_score, similarity_scores_csv):
    # Load the stored similarity scores from the CSV file
    df = pd.read_csv(similarity_scores_csv)
    
    # Assuming the CSV has at least two columns: 'Recipe_ID' and 'Stored_Similarity_Score'
    
    # Calculate the absolute difference between the predicted similarity score and each stored similarity score
    df['similarity_difference'] = abs(df['similarity_scores'] - predicted_similarity_score)
    
    # Sort the DataFrame by the similarity difference in ascending order
    df_sorted = df.sort_values(by='similarity_difference', ascending=True)
    
    # Get the top 5 recipes with the smallest differences
    top_5_recipes = df_sorted.head(5)
    print(top_5_recipes)
    
    # Print the ingredients for the top 5 recipes
    print("Top 5 Recipes' Ingredients:")
    for i, ingredients in enumerate(top_5_recipes['ingredients_str'], 1):
        print(f"{i}. {ingredients}")

    return top_5_recipes


