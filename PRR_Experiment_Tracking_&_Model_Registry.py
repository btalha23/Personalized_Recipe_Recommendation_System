import os
import pandas as pd
import numpy as np
import random
from ast import literal_eval
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

# from tensorflow.keras.models import Sequential
# from tensorflow.keras.layers import Dense
from keras.models import Sequential
from keras.layers import Embedding, LSTM, Dense
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient

NUM_DATA_SAMPLES = 50000
NUM_INGREDIENTS_COMBINATIONS = 10000
MAX_INGREDIENTS_PER_COMBINATIONS = 5
EMBEDDING_OUT_DIM = 300
LSTM_UNITS = 128
# MAX_SEQUENCE_LENGTH = 100

path_base_dataset = '/teamspace/studios/this_studio/Personalized_Recipe_Recommender/dataset/'
path_base_artifacts = '/teamspace/studios/this_studio/Personalized_Recipe_Recommender/artifacts/'

file_name_sampled_data = 'sampled_dataset_' + str(NUM_DATA_SAMPLES) + '.csv'
sampled_data_file_path = path_base_dataset + file_name_sampled_data

file_name_simulated_data = 'simulated_user_input_' + str(NUM_INGREDIENTS_COMBINATIONS) + '.csv'
simulated_data_file_path = path_base_dataset + file_name_simulated_data

# mlflow.set_tracking_uri("sqlite:///mlflow.db")

# MLflow settings
# S3_BUCKET_NAME = "mlops-zoomcamp-prr"
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
# EXPERIMENT_NAME = "Personalized Recipe Recommender - Experiment Tracking & Model Registry"
EXPERIMENT_NAME = "PRR_for_Testing_Code"
# mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
# experiment_id = mlflow.create_experiment(EXPERIMENT_NAME)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(MLFLOW_TRACKING_URI)

def read_csv_data(filename: str) -> pd.DataFrame:
    """Read data into DataFrame"""
    
    df = pd.read_csv(filename)

    return df

def tokenize_data(X_train: pd.DataFrame):
    # Tokenizing only the ingredients
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train['NER'])  # Use the NER column only. NER column contains the ingredients
    ingredient_sequences = tokenizer.texts_to_sequences(X_train['NER'])
    print(f"ingredient_sequences -> {ingredient_sequences[0:5]}")

    # Dimension calculation for the input parameter of embeddings
    embeddings_input_dimension = len(tokenizer.word_index) + 1

    # Calculate the length of each ingredient list
    sequence_lengths = [len(seq) for seq in ingredient_sequences]

    # Set max_sequence_length to a suitable value, e.g., 90th percentile
    max_sequence_length = int(np.percentile(sequence_lengths, 90))

    ingredient_sequences_padded = pad_sequences(ingredient_sequences, maxlen=max_sequence_length)

    # Combine tokenizer and max_sequence_length into a dictionary
    tokenizer_info_data = {
        'tokenizer': tokenizer,
        'max_sequence_length': max_sequence_length
    }

    # Save the tokenizer
    if not os.path.exists(path_base_artifacts):
        os.makedirs(path_base_artifacts)
    
    tokenizer_path = path_base_artifacts + 'tokenizer_info.pickle'
    # tokenizer_path = 'tokenizer.pickle'
    if not os.path.exists(tokenizer_path):
        with open(tokenizer_path, 'wb') as handle:
            pickle.dump(tokenizer_info_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
        mlflow.log_artifact(tokenizer_path)
    else:
        print(f"The file '{tokenizer_path}' already exists.") 

    return max_sequence_length, ingredient_sequences_padded, embeddings_input_dimension

def compute_TFIDFVector(X_train: pd.DataFrame):
    print(X_train.head())
    print(X_train['ingredients_str'].head())
    # Fit a TF-IDF Vectorizer on the ingredients
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(X_train['ingredients_str'])
    print(tfidf_matrix[0:5])

    # Example: Print the shape of the TF-IDF matrix
    print(tfidf_matrix.shape)  # Output: (number of recipes, number of unique words)
    return tfidf, tfidf_matrix

# Function to compute similarity scores
def compute_similarity(user_ingredients, tfidf, tfidf_matrix):
    user_tfidf = tfidf.transform([user_ingredients])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)
    return cosine_sim.flatten()

def add_features(X_train: pd.DataFrame) -> pd.DataFrame:
    # Convert the string representation of lists to actual lists
    # string representation of list to list using ast.literal_eval()
    # df['ingredients_list'] = df['NER'].apply(ast.literal_eval)
    X_train['ingredients_list'] = X_train['NER'].apply(literal_eval)
    print(f"ingredients_list -> {X_train['ingredients_list'].head(5)}")

    # Convert the ingredients to a single string for each recipe
    X_train['ingredients_str'] = X_train['ingredients_list'].apply(lambda x: ' '.join(x))
    print(f"ingredients_str -> {X_train['ingredients_str'].head(5)}")

    return X_train

def preprocess_y_train(X_train: pd.DataFrame, for_y_train: pd.DataFrame):
    tfidf, tfidf_matrix = compute_TFIDFVector(X_train)

    # Iterate over the loaded data to preprocess and compute similarity scores
    for index, row in for_y_train.iterrows():
        user_ingredients = row['user_input_simulated']
        user_ingredients_list = [ingredient.strip() for ingredient in user_ingredients.split(',')]
        user_ingredients_str = ' '.join(user_ingredients_list)
        print(user_ingredients_str)
        
        # Compute similarity scores
        similarity_scores = compute_similarity(user_ingredients_str, tfidf, tfidf_matrix)

        # Print the similarity scores
        print(similarity_scores)
        y_train = similarity_scores
        return y_train

def preprocess_X_train(X_train: pd.DataFrame):
    X_new_features = add_features(X_train)
    print(X_new_features.head())
    X_train = X_new_features[:]
    print(X_train.head())

    max_sequence_length, ingredient_sequences_padded, embeddings_input_dimension = tokenize_data(X_train)

    return X_train, ingredient_sequences_padded, max_sequence_length, embeddings_input_dimension

# def prep_model_to_train():
#     model = Sequential()
#     model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, 
#                         output_dim=EMBEDDING_DIM, 
#                         input_length=max_sequence_length))
#     model.add(LSTM(LSTM_UNITS, return_sequences=False))
#     model.add(Dense(1, activation='sigmoid'))  # Output layer for binary classification

#     # model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])
#     model.compile(optimizer='adam', loss='mean_squared_error', metrics=['mean_absolute_error'])


def model_experiment_tracking(X_train, y_train, MAX_SEQUENCE_LENGTH, EMBEDDING_IN_DIM):
    
    mlflow.autolog()

    # # Define hyperparameters to try
    # batch_size_list = [32, 50] # default tested is batch_size=32
    # # activation_list = ['relu', 'tanh', 'sigmoid'] # default tested is activation_func=sigmoid
    # epochs_list = [10, 25] # default tested is epochs=10
    # optimizers_list = ['adam', 'sgd', 'rmsprop'] # default tested is optimizer=adam
    # lstm_units_list = [128, 256] # default tested is lstm_units=128
    
    # default hyperparameters
    batch_size_list = [32] # default tested is batch_size=32
    # activation_list = ['tanh'] # default tested is activation_func=sigmoid
    epochs_list = [10] # default tested is epochs=10
    optimizers_list = ['adam'] # default tested is optimizer=adam
    lstm_units_list = [128]


    for batch_size in batch_size_list:
        for lstm in lstm_units_list:
            for epochs in epochs_list:
                for optimizer in optimizers_list:
                    tags = {
                        "batch_size": batch_size,
                        "num_lstm_units": lstm,
                        "num_epochs": epochs,
                        "optimizer_type": optimizer
                    }
                    with mlflow.start_run():
                        mlflow.set_tags(tags=tags)
                        
                        mlflow.log_params(
                            {
                                "batch_size": batch_size,
                                "lstm": lstm,
                                "epochs": epochs,
                                "optimizer": optimizer,
                            }
                        )
                        
                        # Build the model
                        model = Sequential([
                                            Embedding(input_dim=EMBEDDING_IN_DIM, 
                                                      output_dim=EMBEDDING_OUT_DIM, 
                                                      input_length=MAX_SEQUENCE_LENGTH),
                                            LSTM(units=lstm, 
                                                 return_sequences=False),  # LSTM hidden layer
                                            Dense(1, activation='sigmoid')  # Output layer
                        ])

                        # Compile the model
                        # model.compile(optimizer=optimizer, loss='binary_crossentropy', metrics=['accuracy'])
                        model.compile(optimizer='adam', 
                                      loss='mean_squared_error', 
                                      metrics=['mean_absolute_error'])

                        # Train the model
                        # history = model.fit(X, y, epochs=epochs, verbose=1)
                        # Assuming ingredient_sequences is your padded input for recipes
                        history = model.fit(X_train, #ingredient_sequences_padded, 
                                            y_train, #similarity_scores, 
                                            epochs=epochs, 
                                            batch_size=batch_size, 
                                            validation_split=0.2)

                        
                        # Evaluate the model
                        loss, mean_absolute_error = model.evaluate(X_train, y_train, verbose=1)
                        # mlflow.log_metric("loss", loss)
                        # mlflow.log_metric("accuracy", accuracy)
                        mlflow.log_metrics(
                            {
                                "mean_absolute_error": mean_absolute_error,
                                "loss": loss,
                            }
                        )

                        # Log the model
                        mlflow.keras.log_model(model, "model")

                        # Make predictions and log them
                        predictions = model.predict(X_train)
                        print(f'Batch Size: {batch_size}, LSTM Units: {lstm}, Epochs: {epochs}, Optimizer: {optimizer}')
                        print(f'Loss: {loss}, Performace Metric: {mean_absolute_error}')
                        print('Predictions:')
                        print(predictions)

                        # convert array into dataframe 
                        pred_df = pd.DataFrame(predictions)
                        path_base_results = '/teamspace/studios/this_studio/Personalized_Recipe_Recommender/results/' 
                        artifacts_file_name = 'recommendations_bs' + str(batch_size) + '_lstm' + str(lstm) + '_epochs' + str(epochs) + '_optimizer_' + optimizer + '.csv' 
                        artifacts_file_name_and_path = path_base_results + artifacts_file_name
                        pred_df.to_csv(artifacts_file_name_and_path)
                        mlflow.log_artifact(artifacts_file_name_and_path)

def main_flow():
    # Load the data
    df_X_train = read_csv_data(sampled_data_file_path)
    df_y_train = read_csv_data(simulated_data_file_path)

    # Prepare the training data
    df_X_train, X_train, MAX_SEQUENCE_LENGTH, EMBEDDING_IN_DIM = preprocess_X_train(df_X_train)
    y_train = preprocess_y_train(df_X_train, df_y_train)

    # Train the model
    model_experiment_tracking(X_train, y_train, MAX_SEQUENCE_LENGTH, EMBEDDING_IN_DIM)
    # Get best model

    # Re-train best model on all data

    # Register best model

    return None

if __name__ == "__main__":
    main_flow()