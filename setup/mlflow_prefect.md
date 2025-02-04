## MLflow & Prefect

***Workflow of Model Training, Experiment Tracking, & Model Registry:***

  - Load the sampled dataset that has features (X_train) from the github repo.
  - Load the simulated user input dataset that will be used to compute labels (y_train) from the github repo.
  - Tokenize and pad X_train to prepare it for inputting in the ML model. 
  - Calculate the similarity scores for the simulated user input data and store them as y_train.
  - Train the LSTM model with a variety of hyperparameters' settings to find a model that has the lowest mean absolute error.
  - Train the model with lowest mean absolute error on all data to be ready to use in production.
  - Register the best model and utilized tokenizer in MLflow registry. All trained model artifacts are stored in AWS S3, identified by their `run_id`. 
  - The registered best model is the final model which is ready to be used in the production setting.


***Procedure for Execution of the Workflow:***

Change directory to `training_orchestration` folder

```
cd ~/personalized_recipe_recommender/training_orchestration
```

Open a new terminal and run the following to start the MLflow tracking server

```
mlflow server --backend-store-uri sqlite:///mlflow.db --default-artifact-root s3://prr-mlops-project-mlflow-artifacts
```
Note that `prr-mlops-project-mlflow-artifacts` is the name of your S3 bucket. Furthermore, in order to allow access to this bucket from MLflow, IAM permissions with this bucket need to be set. For more information, please refer to to the document [here](setup/iam_s3_bucket_permissions.md).

Open a new terminal to start `prefect` server by executing

```
prefect server start
```

Open a new terminal. There are 2 options to run the training pipeline.

***Option 1:*** Run the prefect flow by directly invoking it via `python`

- Make sure that you are in `personalized_recipe_recommender/training_orchestration`. If unsure, run the command below.

	```
	cd ~/personalized_recipe_recommender/training_orchestration
	```

- From inside `personalized_recipe_recommender/training_orchestration`, run

	```
	python training_orchestartion.py
	```


***Option 2:*** Trigger Prefect flow

- For triggering the Prefect flow, make sure that you are in the project root folder. If unsure, run the command

    ```
    cd ~/personalized_recipe_recommender
    ```

- Start the Prefect worker in a separate terminal that pulls work from the `prr_pool` work pool. This can be done by executing

    ```
	prefect worker start --pool 'prr_pool'
    ```

- In a separate terminal, execute the following command to deploy the pipeline
    
	```
    prefect deploy training_orchestration/training_orchestration.py:main_flow -n prr_flow -p prr_pool
	```

- Run the ML model training and model registration pipeline
    
	```
    prefect deployment run 'Train Model Pipeline/prr_flow'
	```

A screenshot to show the Prefect deployment when it is running:

![prefect](../images/prefect_1.png)

This next screenshot displays a completed Prefect deployment run:

![prefect](../images/prefect_2.png)

Model's experiment training runs are observable the MLflow UI:

![mlflow](../images/mlflow_1.png)

The outlook of MLflow's model registry is:

![mlflow](../images/mlflow_3.png)

The training model's artifacts can be browsed through via the MLflow UI:

![mlflow](../images/mlflow_2.png)

The training pipeline/ Prefect deployment stores the artifacts in S3 bucket:

![s3](../images/S3_1.png)

![s3](../images/S3_2.png)

![s3](../images/S3_3.png)
