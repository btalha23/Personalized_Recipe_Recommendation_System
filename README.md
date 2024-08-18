# Personalized Recipe Recommender

## Description

### Problem
In today's digital age, the sheer volume of available recipes can be overwhelming for individuals seeking to find dishes that align with their dietary preferences, tastes, and cooking skills. Traditional recipe search engines often lack personalization and fail to account for users' unique preferences, resulting in a time-consuming and frustrating search experience. A personalized recipe recommender system that understands individual user preferences and dietary restrictions can significantly enhance the cooking experience by providing tailored recipe suggestions.

### Project Objectives:
1. **Develop a Personalized Recipe Recommender System** - making use of the concepts of neural networks build a model that is capable of recommending recipes based on user preference.
	* ***Selected Model for the Project:*** Long short-term memory (LSTM)
	* ***Justification for Choosing LSTM:***
		1.	*Sequential Data Handling:* Recipes and procedures are sequences of words. LSTMs are designed to handle and learn from such sequential data, capturing dependencies between words effectively.
		2.	*Memory Capabilities:* LSTMs can remember information over long sequences, which is useful for understanding the context in a recipe's ingredients and procedures.
		3.	*Better for NLP Tasks:* LSTMs have been shown to perform well on various NLP tasks, including text generation and sequence prediction, making them ideal for understanding and generating recipe instructions.

2. **Identification of a Comprehensive Dataset & Data Handling:**
	* Selected Dataset:* The dataset exploited in this project is [RecipeNLG](https://www.kaggle.com/datasets/saldenisov/recipenlg) that is available on Kaggle. The dataset is rich in recipes, including ingredients, procedures, and named entity recognition (NER) for key elements.
	* Pre-processing: The selected dataset is extremely comprehensive with 2.2GB in size. This extremely huge dataset is not feasible to train on local computers. For this reason, a randomly selected subset of 50000 recipes has been extracted from the 1M+ recipes and the subset is used in this project.
3. **Implement Experiment Tracking & Model Registry:**
	* LSTM is trained on the selected dataset.
	* Model's hyperparameters have been tuned, where all experiments have been saved & tracked.
	* The best model is stored in the model registry for inference and predictions.
	* The performance metric dictating the selection of the best model is *Mean Absolute Error*.
4. **Machine Learning Workflow Orchestration:**
	* A workflow in ML is a sequence of tasks that runs subsequently in the machine learning process. It is ensured that the pipeline is robust and can handle data preprocessing, model training, evaluation, and deployment.
5. Deployment
6. Monitoring


### Tools & Technologies

- Cloud - [**Amazon Web Services**](https://aws.amazon.com/)
- Containerization - [**Docker**](https://www.docker.com) and [**Docker Compose**](https://docs.docker.com/compose/)
- Workflow Orchestration - [**Prefect**](https://www.prefect.io/)
- Experiment Tracking & Model Management - [**MLflow**](https://mlflow.org/)
- Model Artifacts Storage - [**Amazon S3**](https://aws.amazon.com/s3/)
- Model monitoring - [**Evidently AI**](https://www.evidentlyai.com/) and [**Grafana**](https://grafana.com/)
- Language - [**Python**](https://www.python.org)


### Architecture

![architecture](images/architecture.png) TBD

### Exploratory Data Analysis and Modeling
TBD

### Training Pipeline

![prefect](images/prefect_1.png)

![prefect](images/prefect_2.png)

### Experiment Tracking & Model Registry

![mlflow](images/mlflow_1.png)

![mlflow](images/mlflow_2.png)

![mlflow](images/mlflow_3.png)

### Model Storage

![S3](images/S3_1.png)

![S3](images/S3_2.png)

![S3](images/S3_3.png)
