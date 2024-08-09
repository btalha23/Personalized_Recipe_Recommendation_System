import os
import pandas as pd
import numpy as np
import random
from ast import literal_eval

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

from prefect import flow, task, get_run_logger


NUM_DATA_SAMPLES = 50000
NUM_INGREDIENTS_COMBINATIONS = 10000
MAX_INGREDIENTS_PER_COMBINATIONS = 5
EMBEDDING_OUT_DIM = 300
LSTM_UNITS = 128
# MAX_SEQUENCE_LENGTH = 100

path_base = '/teamspace/studios/this_studio/Personalized_Recipe_Recommender/dataset/'

file_name_sampled_data = 'sampled_dataset_' + str(NUM_DATA_SAMPLES) + '.csv'
sampled_data_file_path = path_base + file_name_sampled_data

file_name_simulated_data = 'simulated_user_input_' + str(NUM_INGREDIENTS_COMBINATIONS) + '.csv'
simulated_data_file_path = path_base + file_name_simulated_data

mlflow.set_tracking_uri("sqlite:///mlflow.db")

# MLflow settings
# S3_BUCKET_NAME = "mlops-zoomcamp-prr"
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
EXPERIMENT_NAME = "Personalized Recipe Recommender - Experiment Tracking & Model Registry"
# mlflow.set_tracking_uri(uri="http://127.0.0.1:5000")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
# mlflow.set_experiment("Personalized Recipe Recommender - Experiment Tracking & Model Registry")
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient()

@task(name="Load Data", log_prints=True, retries=3, retry_delay_seconds=2)
def read_csv_data(filename: str) -> pd.DataFrame:
    """   
    Read out the data from the specified CSV File
    This function is used to load the data sampled from a large dataset for training the model (X_train).
    The same function is used to load the simulated user responses that make the labels (y_train) for the model.

    Args:
        path (str): Path to the CSV file.

    Returns:
        pd.DataFrame: Loaded data as a DataFrame.
    """

    logger = get_run_logger()
    logger.info("Loading data from %s", filename)
    
    df = pd.read_csv(filename)

    return df

@task(log_prints=True)
def tokenize_data(X_train: pd.DataFrame):
    
    logger = get_run_logger()
    logger.info("Tokenizing the ingredients...")

    # Tokenizing only the ingredients
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(X_train['NER'])  # Use the NER column only. NER column contains the ingredients
    ingredient_sequences = tokenizer.texts_to_sequences(X_train['NER'])
    print(f"ingredient_sequences -> {ingredient_sequences[0:5]}")

    # Dimension calculation for the input parameter of embeddings
    logger.info("Dimension calculation for the input parameter of embeddings")
    embeddings_input_dimension = len(tokenizer.word_index) + 1

    # Calculate the length of each ingredient list
    logger.info("Calculate the length of each ingredient list")
    sequence_lengths = [len(seq) for seq in ingredient_sequences]

    # Set max_sequence_length to a suitable value, e.g., 90th percentile
    logger.info("Set max_sequence_length to a suitable value, e.g., 90th percentile")
    max_sequence_length = int(np.percentile(sequence_lengths, 90))

    logger.info("Padding the data as required to complete data preparation of X_train")
    ingredient_sequences_padded = pad_sequences(ingredient_sequences, maxlen=max_sequence_length)

    return max_sequence_length, ingredient_sequences_padded, embeddings_input_dimension

@task(log_prints=True)
def compute_TFIDFVector(X_train: pd.DataFrame):
    logger = get_run_logger()
    logger.info("TF-IDF vectorization of ingredients...")

    logger.info("Fit a TF-IDF Vectorizer on the ingredients")
    tfidf = TfidfVectorizer()
    tfidf_matrix = tfidf.fit_transform(X_train['ingredients_str'])
    print(tfidf_matrix[0:5])

    # Print the shape of the TF-IDF matrix
    logger.info("Print the shape of the TF-IDF matrix")
    print(tfidf_matrix.shape)  # Output: (number of recipes, number of unique words)
    
    return tfidf, tfidf_matrix

# Function to compute similarity scores
@task(log_prints=True)
def compute_similarity(user_ingredients, tfidf, tfidf_matrix):
    logger = get_run_logger()
    logger.info("Computation of similarity scores starts here...")

    user_tfidf = tfidf.transform([user_ingredients])
    cosine_sim = cosine_similarity(user_tfidf, tfidf_matrix)

    return cosine_sim.flatten()

@task(log_prints=True)
def add_features(X_train: pd.DataFrame) -> pd.DataFrame:

    logger = get_run_logger()
    logger.info("Feature engineering...")

    # Convert the string representation of lists to actual lists
    logger.info("Convert the string representation of lists to actual lists")
    # string representation of list to list using ast.literal_eval(); ast.literal_eval is imported as literal_eval
    X_train['ingredients_list'] = X_train['NER'].apply(literal_eval)
    print(f"ingredients_list -> {X_train['ingredients_list'].head(5)}")

    # Convert the ingredients to a single string for each recipe
    logger.info("Convert the ingredients to a single string for each recipe")
    X_train['ingredients_str'] = X_train['ingredients_list'].apply(lambda x: ' '.join(x))
    print(f"ingredients_str -> {X_train['ingredients_str'].head(5)}")

    return X_train

@task(name="Prepare y_train (Labels)", log_prints=True)
def preprocess_y_train(X_train: pd.DataFrame, for_y_train: pd.DataFrame):

    logger = get_run_logger()
    logger.info("Preparing the target labels (y_train)...")

    tfidf, tfidf_matrix = compute_TFIDFVector(X_train)

    # Iterate over the loaded data to preprocess and compute similarity scores
    logger.info("Iterate over the loaded data to preprocess and compute similarity scores")
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

@task(name="Prepare X_train (features)", log_prints=True)
def preprocess_X_train(X_train: pd.DataFrame):

    logger = get_run_logger()
    logger.info("Preparing the data for training (X_train)...")

    X_new_features = add_features(X_train)
    print(X_new_features.head())
    X_train = X_new_features[:]
    print(X_train.head())

    max_sequence_length, ingredient_sequences_padded, embeddings_input_dimension = tokenize_data(X_train)

    return X_train, ingredient_sequences_padded, max_sequence_length, embeddings_input_dimension

@task(name="Train Model", log_prints=True)
def model_experiment_tracking(X_train, y_train, MAX_SEQUENCE_LENGTH, EMBEDDING_IN_DIM):

    logger = get_run_logger()
    logger.info("Model training, experiment tracking, and model registry...")

    # Define hyperparameters to try
    batch_size_list = [32, 50] # default tested is batch_size=32
    # activation_list = ['relu', 'tanh', 'sigmoid'] # default tested is activation_func=sigmoid
    epochs_list = [5, 10, 25] # default tested is epochs=10
    optimizers_list = ['adam', 'sgd', 'rmsprop'] # default tested is optimizer=adam
    lstm_units_list = [64, 128, 256] # default tested is lstm_units=128
    

    # # default hyperparameters
    # batch_size_list = [32] # default tested is batch_size=32
    # activation_list = ['tanh'] # default tested is activation_func=sigmoid
    # epochs_list = [10] # default tested is epochs=10
    # optimizers_list = ['adam'] # default tested is optimizer=adam
    # lstm_units_list = [128]


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

@task(name="Get Best Model", log_prints=True)
def get_best_model(client, EXPERIMENT_NAME):
    logger = get_run_logger()
    logger.info("Get the model with the lowest mean absolute error...")

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.mean_absolute_error ASC"])[0]
    best_run_id = best_run.info.run_id

    best_run_tags = best_run.data.tags
    tag_key = 'model'
    tag_value = best_run_tags.get(tag_key)
    return best_run_id, tag_value

@task(name="Re-train best model on all training data", log_prints=True)
def train_all_data(S3_BUCKET_NAME, RUN_ID, X_train, y_train, tag_value):
    logger = get_run_logger()
    logger.info("Re-train best model on all data...")

    logged_model = f's3://{S3_BUCKET_NAME}/1/{RUN_ID}/artifacts/model'
    model = mlflow.keras.load_model(logged_model)

    with mlflow.start_run() as run:
        
        #Train the model
        model.fit(X_train, y_train)

        # Log the model
        logger.info("Logging the model...")

        mlflow.set_tag("model", tag_value)
        mlflow.keras.log_model(model, "model")

        logger.info("Completed training process...")

        register_run_id = run.info.run_id

        return register_run_id

@task(name="Register Best Model", log_prints=True)
def register_best_model(client, register_run_id, model_name, tag_value):
    logger = get_run_logger()
    logger.info(f"Register the best model which has run_id: {register_run_id}...")

    result = mlflow.register_model(
        model_uri=f"runs:/{register_run_id}/models",
        name=model_name)
    
    # Add a description to the model version
    description = f'{tag_value} model retrained with all training data.'
    client.update_model_version(
        name=result.name,
        version=result.version,
        description=description
    )
    logger.info(f"Model registered: {result.name}, version {result.version}...")

    return None

@flow(name="Train Model Pipeline", log_prints=True)
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
    best_run_id, tag_value = get_best_model(client, EXPERIMENT_NAME)

    # Re-train best model on all data
    register_run_id = train_all_data(S3_BUCKET_NAME, best_run_id, X_train, y_train, tag_value)

    # Register best model
    model_name = "Personalized-Recipe_Recommender"
    register_best_model(client, register_run_id, model_name, tag_value)

    return None

if __name__ == "__main__":
    main_flow()