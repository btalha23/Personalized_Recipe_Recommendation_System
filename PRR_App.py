import numpy as np
import pandas as pd

import os
import glob
import pickle

import mlflow

from flask import Flask, request, jsonify

from keras_preprocessing.sequence import pad_sequences
from keras.models import load_model

# MLflow settings
S3_BUCKET_NAME = "prr-mlops-project-mlflow-artifacts"
EXPERIMENT_NAME = "PRR_for_Testing_Code"
mlflow.set_experiment(EXPERIMENT_NAME)

RUN_ID = '22178b67206c480fa6f91ae81bb158d3'

def load_logged_model(RUN_ID):

    required_model = f's3://{S3_BUCKET_NAME}/3/{RUN_ID}/artifacts/model'
    
    # Load the trained model
    loaded_model = mlflow.keras.load_model(required_model)

    return loaded_model

def load_tokenizer(RUN_ID):

    required_tokenizer = f's3://{S3_BUCKET_NAME}/3/{RUN_ID}/artifacts/tokenizer_info.pickle'
    
    tokenizer_path = mlflow.artifacts.download_artifacts(required_tokenizer)
    print(f"Downloading the tokenizer info data to {tokenizer_path}")

    # Load the pickle file
    with open(tokenizer_path, 'rb') as handle:
        tokenizer_info_data = pickle.load(handle)

    # Extract the tokenizer and max_sequence_length
    tokenizer = tokenizer_info_data['tokenizer']
    max_sequence_length = tokenizer_info_data['max_sequence_length']

    print(f"max_sequence_length {max_sequence_length}")

    return max_sequence_length, tokenizer

def predict_from_model(RUN_ID, user_ingredients):
    # user_ingredients = "chicken garlic onion salt pepper"
    # user_ingredients = "Beets vinegar cloves sugar salt water"
    # user_ingredients = "Beets vinegar cloves sugar salt water"
    user_ingredients_list = [ingredient.strip() for ingredient in user_ingredients.split(',')]
    user_ingredients_str = ' '.join(user_ingredients_list)
    print(user_ingredients_str)
    
    max_sequence_length, tokenizer = load_tokenizer(RUN_ID)

    # Tokenize and pad the user ingredients
    user_sequences = tokenizer.texts_to_sequences([user_ingredients_str])
    user_padded = pad_sequences(user_sequences, maxlen=max_sequence_length)
    print(f"user_padded_sequence {user_padded}")
    
    # Predict similarity scores using the model
    # user_predictions = model.predict(user_padded)
    loaded_model = load_logged_model(RUN_ID)
    # loaded_model = mlflow.keras.load_model(f"runs:/{RUN_ID}/model")
    print(f"loaded_model {loaded_model}")

    # Predict on a Pandas DataFrame.
    user_predictions = loaded_model.predict(user_padded)
    print(f"user_predictions {user_predictions}")
    
    # Flatten the predictions if needed
    if user_predictions.ndim > 1:
        user_predictions_flattened = user_predictions.flatten()
        print(user_predictions_flattened)
    
    # # Get the indices of the top 5 most similar recipes
    # top_indices = user_predictions_flattened.argsort()[-5:][::-1]
    # recommended_recipes = original_dataframe.iloc[top_indices]

    # print(top_indices)
    # print(recommended_recipes)

    return user_predictions_flattened

def compare_with_all_recipes(RUN_ID, predicted_similarity_score):

    required_artifacts = f's3://{S3_BUCKET_NAME}/3/{RUN_ID}/artifacts'
    # labelled_data_path = mlflow.artifacts.download_artifacts(f"runs:/{RUN_ID}/")
    labelled_data_path = mlflow.artifacts.download_artifacts(required_artifacts)    
    # # path to search all txt files 
    # labelled_data_path = "runs:/"+str(RUN_ID)+"/model/artifacts/"
    print(labelled_data_path)

    # Path to the directory (absolute or relative)
    dir = labelled_data_path #"C:\Users\Sauleyayan\Desktop\New folder"
    
    # os.listdir return a list of all files within 
    # the specified directory
    for file in os.listdir(dir):
        print(file)

            # The following condition checks whether 
        # the filename ends with .txt or not
        if file.startswith("labelled_data_"):
    
            # Appending the filename to the path to obtain 
            # the fullpath of the file
            labelled_data_path = os.path.join(dir, file)
            print(os.path.join(dir, file))
    
    # for file in glob.glob(labelled_data):
    #     print(labelled_data)

        # Load the stored similarity scores from the CSV file
    df = pd.read_csv(labelled_data_path)
    
    # Calculate the absolute difference between the predicted similarity score and each stored similarity score
    df['similarity_difference'] = abs(df['similarity_scores'] - predicted_similarity_score)
    df.head()

    # Sort the DataFrame by the similarity difference in ascending order
    df_sorted = df.sort_values(by='similarity_difference', ascending=True)
    
    # Get the top 5 recipes with the smallest differences
    top_5_recipes = df_sorted.head(5)
    print(top_5_recipes)

    return top_5_recipes


# Create the Flask app
app = Flask('Personalized Recipe Recommender')

# @app.route('/')
# def hello_world():
#     return 'hello world'

@app.route('/recommendations', methods=['POST'])
def predict_endpoint():
    # Get user input from request
    user_ingredients = request.json.get('ingredients', '')

    predictions_from_model = predict_from_model(RUN_ID, user_ingredients)
    top_5_recipes = compare_with_all_recipes(RUN_ID, predictions_from_model)

    #     # Print the ingredients for the top 5 recipes
    # print("Top 5 Recipes' Ingredients:")
    # for i, ingredients in enumerate(top_5_recipes['ingredients_str'], 1):
    #     print(f"{i}. {ingredients}")

    # # Create a list of dictionaries for the top 5 recipes
    # result = []
    # for i, ingredients in enumerate(top_5_recipes['ingredients_str'], 1):
    #     result.append({"rank": i, "ingredients": ingredients})
    # Create a list of dictionaries for the top 5 recipes
    result = []
    for i, row in top_5_recipes.iterrows():
        result.append({
            "Rank": i+1, 
            "Recipe Name": row['name'],
            "Ingredints": row['NER'],
            "Ingredient Quantities": row['ingredients'], 
            "Procedure": row['procedure']
        })

    return jsonify(result)

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=9696)

# recommend()

# def recommend():
#     # Get user input from request
#     # user_ingredients = request.json.get('ingredients', '')
#     user_ingredients = "chicken garlic onion salt pepper"
#     user_ingredients_list = [ingredient.strip() for ingredient in user_ingredients.split(',')]
#     user_ingredients_str = ' '.join(user_ingredients_list)
#     print(user_ingredients_str)
    
#     max_sequence_length, tokenizer = load_tokenizer()
#     # Tokenize and pad the user ingredients
#     user_sequences = tokenizer.texts_to_sequences([user_ingredients_str])
#     user_padded = pad_sequences(user_sequences, maxlen=max_sequence_length)
#     print(f"user_padded_sequence {user_padded}")
    
#     # Predict similarity scores using the model
#     # user_predictions = model.predict(user_padded)
#     loaded_model = load_logged_model()
#     print(f"loaded_model {load_logged_model}")
#     # Predict on a Pandas DataFrame.
#     user_predictions = loaded_model.predict(user_padded)
#     print(f"user_predictions {user_predictions}")
    
#     # Flatten the predictions if needed
#     user_predictions_flattened = user_predictions.flatten()
    
#     # Get the indices of the top 5 most similar recipes
#     top_indices = user_predictions_flattened.argsort()[-5:][::-1]
    
#     print(top_indices)

#     return top_indices
    
#     # Return the top 5 recommended recipes (indices or any other identifying information)
#     # return jsonify({
#     #     'recommended_recipe_indices': top_indices.tolist()
#     # })

# if __name__ == '__main__':
#     app.run(debug=True)
