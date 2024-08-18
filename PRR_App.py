import numpy as np
import pandas as pd

import os
import pickle

import mlflow

from flask import Flask, request, jsonify

from keras_preprocessing.sequence import pad_sequences
# from keras.saving import load_model

# MLflow settings
S3_BUCKET_NAME = "prr-mlops-project-mlflow-artifacts"
EXPERIMENT_NAME = "Personalized Recipe Recommender"
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id

RUN_ID = '22178b67206c480fa6f91ae81bb158d3'

def load_logged_model(RUN_ID):

    required_model = f's3://{S3_BUCKET_NAME}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/model'
    
    # Load the trained model
    loaded_model = mlflow.keras.load_model(required_model)

    return loaded_model

def load_tokenizer(RUN_ID):

    required_tokenizer = f's3://{S3_BUCKET_NAME}/{EXPERIMENT_ID}/{RUN_ID}/artifacts/tokenizer_info.pickle'
    
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
    
    loaded_model = load_logged_model(RUN_ID)
    print(f"loaded_model {loaded_model}")

    # Predict similarity scores using the model
    user_predictions = loaded_model.predict(user_padded)
    print(f"user_predictions {user_predictions}")
    
    # Flatten the predictions if needed
    if user_predictions.ndim > 1:
        user_predictions_flattened = user_predictions.flatten()
        print(user_predictions_flattened)
    
    return user_predictions_flattened

def compare_with_all_recipes(RUN_ID, predicted_similarity_score):

    required_artifacts = f's3://{S3_BUCKET_NAME}/{EXPERIMENT_ID}/{RUN_ID}/artifacts'
    labelled_data_path = mlflow.artifacts.download_artifacts(required_artifacts)    
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

@app.route('/recommendations', methods=['POST'])
def predict_endpoint():
    # Get user input from request
    user_ingredients = request.json.get('ingredients', '')

    predictions_from_model = predict_from_model(RUN_ID, user_ingredients)
    top_5_recipes = compare_with_all_recipes(RUN_ID, predictions_from_model)

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