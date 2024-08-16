import os
import pandas as pd
import numpy as np
import random
from ast import literal_eval
import pickle

from sklearn.metrics.pairwise import cosine_similarity
from sklearn.feature_extraction.text import TfidfVectorizer

from keras import models
from keras import layers
from keras_preprocessing.text import Tokenizer
from keras_preprocessing.sequence import pad_sequences

import mlflow
import mlflow.keras
from mlflow.tracking import MlflowClient
from mlflow.entities import ViewType

from prefect import flow, task, get_run_logger


NUM_DATA_SAMPLES = 50000
NUM_INGREDIENTS_COMBINATIONS = 10000
MAX_INGREDIENTS_PER_COMBINATIONS = 5
EMBEDDING_OUT_DIM = 300
LSTM_UNITS = 128
WRITE_TOKENIZER_AS_MLFLOW_ARTIFACT = True

os.chdir('./dataset/')
path_base_dataset = os.getcwd()
print(path_base_dataset)

os.chdir('../artifacts/')
path_base_artifacts = os.getcwd()
print(path_base_artifacts)

os.chdir('../results/')
path_base_results = os.getcwd()
print(path_base_results) 

os.chdir('../')
path_base_root_folder = os.getcwd()
print(path_base_root_folder) 

file_name_sampled_data = '/sampled_dataset_' + str(NUM_DATA_SAMPLES) + '.csv'
sampled_data_file_path = path_base_dataset + file_name_sampled_data

file_name_simulated_data = '/simulated_user_input_' + str(NUM_INGREDIENTS_COMBINATIONS) + '.csv'
simulated_data_file_path = path_base_dataset + file_name_simulated_data


# MLflow settings
S3_BUCKET_NAME = "prr-mlops-project-mlflow-artifacts"
MLFLOW_TRACKING_URI = 'http://127.0.0.1:5000'
EXPERIMENT_NAME = "Personalized Recipe Recommender"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment(EXPERIMENT_NAME)
client = MlflowClient(MLFLOW_TRACKING_URI)
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
EXPERIMENT_ID = experiment.experiment_id

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

@task(name="Tokenize X_train", log_prints=True)
def tokenize_data(X_train: pd.DataFrame):
    """
    Tokenizes the ingredient data from the input DataFrame, calculates the necessary padding, and prepares 
    the data for embedding layers in a neural network model.

    Parameters:
    ----------
    X_train : pd.DataFrame
        A DataFrame containing the training data. The DataFrame should have a column 'NER' that contains
        the ingredient lists for each recipe.

    Returns:
    -------
    max_sequence_length : int
        The maximum sequence length determined by the 90th percentile of ingredient list lengths.
    
    ingredient_sequences_padded : np.ndarray
        A NumPy array where each ingredient sequence is padded to the maximum sequence length.
        
    embeddings_input_dimension : int
        The dimension of the input for embedding layers, determined by the size of the tokenizer's word index.
    
    tokenizer_info_data : dict
        A dictionary containing:
            - 'tokenizer': The trained tokenizer object.
            - 'max_sequence_length': The calculated maximum sequence length.
    
    Notes:
    -----
    This function only tokenizes the 'NER' column of the input DataFrame, which contains the ingredient lists.
    The tokenized sequences are then padded to the same length based on the 90th percentile of sequence lengths.
    """
    
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

    # Combine tokenizer and max_sequence_length into a dictionary
    logger.info("Combine tokenizer and max_sequence_length into a dictionary")
    tokenizer_info_data = {
        'tokenizer': tokenizer,
        'max_sequence_length': max_sequence_length
    }

    return max_sequence_length, ingredient_sequences_padded, embeddings_input_dimension, tokenizer_info_data

@task(name="X_train TFIDF Vector", log_prints=True)
def compute_TFIDFVector(X_train: pd.DataFrame):
    """
    Computes the TF-IDF vectors for the ingredients in the input DataFrame.

    Parameters:
    ----------
    X_train : pd.DataFrame
        A DataFrame containing the training data. The DataFrame should have a column 'ingredients_str' 
        with the ingredients for each recipe as a single string.

    Returns:
    -------
    tfidf : TfidfVectorizer
        The fitted TF-IDF vectorizer object.
    
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        A sparse matrix representing the TF-IDF vectors of the ingredients in the input data.
    """
        
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
@task(name="Computing Similarity", log_prints=True)
def compute_similarity(user_input, tfidf, tfidf_matrix):
    """
    Computes the cosine similarity between the user's input and the TF-IDF vectors of the ingredients.

    Parameters:
    ----------
    user_input : str
        A string representing the user's input ingredients.
    
    tfidf : TfidfVectorizer
        The fitted TF-IDF vectorizer used to transform the ingredients into TF-IDF vectors.
    
    tfidf_matrix : scipy.sparse.csr.csr_matrix
        A sparse matrix representing the TF-IDF vectors of the ingredients in the dataset.

    Returns:
    -------
    numpy.ndarray
        A 1D array of cosine similarity scores between the user's input and each recipe in the dataset.
    """
        
    logger = get_run_logger()
    logger.info("Computation of similarity scores starts here...")

    user_input_tfidf = tfidf.transform([user_input])
    cosine_sim = cosine_similarity(user_input_tfidf, tfidf_matrix)

    return cosine_sim.flatten()

@task(name="Feature Engineering", log_prints=True)
def add_features(X_train: pd.DataFrame) -> pd.DataFrame:
    """
    Performs feature engineering on the ingredients data by converting string representations of lists into actual lists
    and then converting these lists into single strings for each recipe.

    Parameters:
    ----------
    X_train : pd.DataFrame
        A DataFrame containing the ingredients data. The 'NER' column is expected to contain string representations 
        of lists of ingredients.

    Returns:
    -------
    pd.DataFrame
        The input DataFrame with two additional columns: 
        'ingredients_list' containing the actual lists of ingredients and 
        'ingredients_str' containing a single string of ingredients for each recipe.
    """

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

@flow(name="Prepare y_train (Labels)", log_prints=True)
def preprocess_y_train(X_train: pd.DataFrame, for_y_train: pd.DataFrame) -> np.ndarray:
    """
    Preprocesses the target labels (`y_train`) by computing similarity scores between the simulated user input
    and the ingredient data.

    Parameters:
    ----------
    X_train : pd.DataFrame
        A DataFrame containing the training data with ingredients. The 'ingredients_str' column is expected 
        to contain the ingredients as a single string for each recipe.

    for_y_train : pd.DataFrame
        A DataFrame containing simulated user inputs. The 'user_input_simulated' column is expected to contain 
        strings representing the ingredients entered by a user.

    Returns:
    -------
    np.ndarray
        An array of similarity scores to be used as the target labels (`y_train`).
    """

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

@flow(name="Prepare X_train (Features)", log_prints=True)
def preprocess_X_train(X_train: pd.DataFrame) -> tuple:
    """
    Prepares the training data (`X_train`) for model training by adding features, tokenizing ingredients, 
    and generating necessary information for embeddings.

    Parameters:
    ----------
    X_train : pd.DataFrame
        A DataFrame containing the training data, including an 'NER' column with the ingredients.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - X_train: pd.DataFrame
            The updated DataFrame after adding new features.
        - ingredient_sequences_padded: np.ndarray
            The padded sequences of ingredient tokens.
        - max_sequence_length: int
            The maximum sequence length for the ingredient sequences.
        - embeddings_input_dimension: int
            The input dimension size for the embeddings layer.
        - tokenizer_info_data: dict
            A dictionary containing the tokenizer and the maximum sequence length.
    """
    
    logger = get_run_logger()
    logger.info("Preparing the data for training (X_train)...")

    X_new_features = add_features(X_train)
    print(X_new_features.head())
    X_train = X_new_features[:]
    print(X_train.head())

    max_sequence_length, ingredient_sequences_padded, embeddings_input_dimension, tokenizer_info_data = tokenize_data(X_train)

    return X_train, ingredient_sequences_padded, max_sequence_length, embeddings_input_dimension, tokenizer_info_data

@task(name="Train Model", log_prints=True)
def model_experiment_tracking(X_train, 
                              y_train, 
                              MAX_SEQUENCE_LENGTH, 
                              EMBEDDING_IN_DIM, 
                              df_X_train,
                              tokenizer_info_data) -> tuple:
    """
    Conducts model training, experiment tracking, and model registry using MLflow for hyperparameter tuning and 
    logging model artifacts.

    Parameters:
    ----------
    X_train : np.ndarray
        Padded tokenized sequences used as input features for training the model.
    y_train : np.ndarray
        Similarity scores used as target labels for training the model.
    MAX_SEQUENCE_LENGTH : int
        The maximum length of the tokenized sequences.
    EMBEDDING_IN_DIM : int
        The input dimension for the embedding layer.
    df_X_train : pd.DataFrame
        The DataFrame containing the training data with additional features.
    tokenizer_info_data : dict
        A dictionary containing the tokenizer and the maximum sequence length.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - run_id: str
            The Run ID generated by MLflow for the experiment.
        - model: keras.Sequential
            The trained Keras model.
    """

    logger = get_run_logger()
    logger.info("Model training, experiment tracking, and model registry...")

    global WRITE_TOKENIZER_AS_MLFLOW_ARTIFACT
    df = df_X_train

    mlflow.autolog()

    # # Selected set of hyperparameters
    # Define hyperparameters to try
    batch_size_list = [50] # default tested is batch_size=32 # for rmsprop add bs50 , epoc150, & lstm256
    activation_list = ['sigmoid'] # default tested is activation_func=sigmoid;
                                            # Activation function of the output layer
    epochs_list = [100] # default tested is epochs=10 # for rmsprop add bs50 , epoc150, & lstm256
    optimizers_list = ['rmsprop'] # default tested is optimizer=adam
    lstm_units_list = [256, 128] # default tested is lstm_units=128 # for rmsprop add bs50 , epoc150, & lstm256

    # # default hyperparameters
    # batch_size_list = [32] # default tested is batch_size=32
    # activation_list = ['sigmoid'] # default tested is activation_func=sigmoid
    # epochs_list = [10] # default tested is epochs=10
    # optimizers_list = ['adam'] # default tested is optimizer=adam
    # lstm_units_list = [128]

    for activation_func in activation_list:
        for batch_size in batch_size_list:
            for lstm in lstm_units_list:
                for epochs in epochs_list:
                    for optimizer in optimizers_list:
                        tags = {
                            "batch_size": batch_size,
                            "num_lstm_units": lstm,
                            "num_epochs": epochs,
                            "optimizer_type": optimizer,
                            "activation_func_output_layer": activation_func,
                            "model": 'LSTM_bs' + str(batch_size) + '_lstm' + str(lstm) + '_epochs' + str(epochs) + '_optimizer_' + optimizer + '_actFunc_' + activation_func
                        }
                        with mlflow.start_run() as run:
                            # Retrieve the Run ID
                            run_id = run.info.run_id
                            print(f'Run ID: {run_id}')
                            
                            mlflow.set_tags(tags=tags)
                            
                            mlflow.log_params(
                                {
                                    "batch_size": batch_size,
                                    "lstm": lstm,
                                    "epochs": epochs,
                                    "optimizer": optimizer,
                                    "activation_func_output_layer": activation_func
                                }
                            )
                            
                            # Build the model
                            model = models.Sequential([
                                                layers.Embedding(input_dim=EMBEDDING_IN_DIM, 
                                                                 output_dim=EMBEDDING_OUT_DIM, 
                                                                 input_length=MAX_SEQUENCE_LENGTH),
                                                layers.LSTM(units=lstm, 
                                                            return_sequences=False),  # LSTM hidden layer
                                                layers.Dense(1, activation=activation_func)  # Output layer
                            ])

                            # Compile the model
                            model.compile(optimizer=optimizer, 
                                        loss='mean_squared_error', 
                                        metrics=['mean_absolute_error'])

                            # Train the model
                            # Assuming ingredient_sequences is your padded input for recipes
                            history = model.fit(X_train, #ingredient_sequences_padded, 
                                                y_train, #similarity_scores, 
                                                epochs=epochs, 
                                                batch_size=batch_size, 
                                                validation_split=0.2)
                            
                            # Evaluate the model
                            loss, mean_absolute_error = model.evaluate(X_train, y_train, verbose=1)
                            mlflow.log_metrics(
                                {
                                    "mean_absolute_error": mean_absolute_error,
                                    "loss": loss,
                                }
                            )

                            # Log the model
                            mlflow.keras.log_model(model, "model")

                            print(X_train)
                            # Make predictions and log them
                            predictions = model.predict(X_train)
                            print(f'Batch Size: {batch_size}, LSTM Units: {lstm}, Epochs: {epochs}, Optimizer: {optimizer}')
                            print(f'Loss: {loss}, Performace Metric: {mean_absolute_error}')
                            print('Predictions:')
                            print(predictions)
                            df['similarity_scores'] = predictions
                            print(df.head())

                            # convert array into dataframe 
                            pred_df = pd.DataFrame(predictions)
                            # path_base_results = './results/' 
                            artifacts_file_name = '/recommendations_bs' + str(batch_size) + '_lstm' + str(lstm) + '_epochs' + str(epochs) + '_optimizer_' + optimizer + '_' + str(run_id) + '.csv' 
                            artifacts_file_name_and_path = path_base_results + artifacts_file_name
                            pred_df.to_csv(artifacts_file_name_and_path)
                            mlflow.log_artifact(artifacts_file_name_and_path)

                            labelled_source_data_file_name = '/labelled_data_bs' + str(batch_size) + '_lstm' + str(lstm) + '_epochs' + str(epochs) + '_optimizer_' + optimizer + '_' + str(run_id) + '.csv' 
                            artifacts_file_name_and_path_labelled_data = path_base_results + labelled_source_data_file_name
                            df.to_csv(artifacts_file_name_and_path_labelled_data)
                            mlflow.log_artifact(artifacts_file_name_and_path_labelled_data)

                            # Save the tokenizer
                            if not os.path.exists(path_base_artifacts):
                                os.makedirs(path_base_artifacts)
                            
                            tokenizer_path = path_base_artifacts + '/tokenizer_info.pickle'
                            # tokenizer_path = 'tokenizer.pickle'
                            
                            if not os.path.exists(tokenizer_path):
                                with open(tokenizer_path, 'wb') as handle:
                                    pickle.dump(tokenizer_info_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                mlflow.log_artifact(tokenizer_path)
                            else:
                                print(f"The file '{tokenizer_path}' already exists.")
                            
                            if WRITE_TOKENIZER_AS_MLFLOW_ARTIFACT:
                                with open(tokenizer_path, 'wb') as handle:
                                    pickle.dump(tokenizer_info_data, handle, protocol=pickle.HIGHEST_PROTOCOL)
                                mlflow.log_artifact(tokenizer_path)
                                WRITE_TOKENIZER_AS_MLFLOW_ARTIFACT = False
            
    return run_id, model

@task(name="Get Best Model", log_prints=True)
def get_best_model(client, EXPERIMENT_NAME):
    """
    Retrieves the best model run with the lowest mean absolute error from an MLflow experiment.

    Parameters:
    ----------
    client : mlflow.tracking.MlflowClient
        An instance of MLflowClient used to interact with the MLflow tracking server.
    EXPERIMENT_NAME : str
        The name of the experiment from which to retrieve the best model run.

    Returns:
    -------
    tuple
        A tuple containing the following elements:
        - best_run_id: str
            The Run ID of the best model run.
        - tag_value: str
            The value of the 'model' tag associated with the best run.
    """
        
    logger = get_run_logger()
    logger.info("Get the model with the lowest mean absolute error...")

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    print(f"experiment ID: {experiment.experiment_id}")
    logger.info(f"experiment ID: {experiment.experiment_id}")

    best_run = client.search_runs(
        experiment_ids=experiment.experiment_id,
        run_view_type=ViewType.ACTIVE_ONLY,
        max_results=1,
        order_by=["metrics.mean_absolute_error ASC"])[0]
    best_run_id = best_run.info.run_id
    print(f"ID of the best run: {best_run_id}")
    logger.info(f"ID of the best run: {best_run_id}")

    best_run_tags = best_run.data.tags
    tag_key = 'model'
    tag_value = best_run_tags.get(tag_key)
    print(f"Value of the model tag: {tag_value}")
    logger.info(f"Value of the model tag: {tag_value}")

    return best_run_id, tag_value

@task(name="Re-train best model on all training data", log_prints=True)
def train_all_data(S3_BUCKET_NAME, 
                   RUN_ID, 
                   tag_value, 
                   X_train, 
                   y_train, 
                   MAX_SEQUENCE_LENGTH, 
                   EMBEDDING_IN_DIM, 
                   df_X_train):

    logger = get_run_logger()
    logger.info("Re-train best model on all data...")

    logged_model = f's3://{S3_BUCKET_NAME}/2/{RUN_ID}/artifacts/model'
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
    df_X_train, X_train, MAX_SEQUENCE_LENGTH, EMBEDDING_IN_DIM, tokenizer_info_data = preprocess_X_train(df_X_train)
    y_train = preprocess_y_train(df_X_train, df_y_train)

    # Train the model
    RUN_ID, model_original = model_experiment_tracking(X_train, 
                                                       y_train, 
                                                       MAX_SEQUENCE_LENGTH, 
                                                       EMBEDDING_IN_DIM, 
                                                       df_X_train,
                                                       tokenizer_info_data)

    # Get best model
    best_run_id, tag_value = get_best_model(client, EXPERIMENT_NAME)
    print(f"best_run_id {best_run_id}")
    print(f"tag_value {tag_value}")
    
    # Re-train best model on all data
    register_run_id = train_all_data(S3_BUCKET_NAME, RUN_ID, 
                                     tag_value, X_train, y_train, 
                                     MAX_SEQUENCE_LENGTH, 
                                     EMBEDDING_IN_DIM, 
                                     df_X_train)

    # Register best model
    model_name = "Personalized_Recipe_Recommender"
    register_best_model(client, register_run_id, model_name, tag_value)

    return None

if __name__ == "__main__":
    main_flow()