## Model Deployment


Change directory to `deployment` folder

```
cd ~/personalized_recipe_recommender/deployment
```

You have a few options to utilize for model inference testing,

1. Flask
2. Guicorn
3. Docker 

Open a new terminal and make sure that you are in `personalized_recipe_recommender/deployment` 

***Flask:*** Run the Flask application by directly invoking it via `python`

- In the terminal, run the command below.

	```
	pipenv shell
	```

It will activate a new environment and will install the required dependencies mentioned in the `Pipfile` file.

- Once the environment is ready, the run

	```
	python personalized_recipe_recommender.py
	```

This will start the Flask application that would then be waiting for the requests from the user to perform prediction.

- One a new terminal and make sure that you are in `personalized_recipe_recommender/deployment`. From inside `personalized_recipe_recommender/deployment`, run 

	```
	python test.py
	```
`test.py` has input values that are passed on to the Flask application and received results are printed on the console window.


***Gunicorn:*** Trigger the Flask application using Gunicorn

- One a new terminal and make sure that you are in `personalized_recipe_recommender/deployment`. From inside `personalized_recipe_recommender/deployment`, run the command below.

	```
	pipenv shell
	```

It will activate a new environment and will install the required dependencies mentioned in the `Pipfile` file.

- Once the environment is ready, the run

    ```
    gunicorn --bind=0.0.0.0:9696 personalized_recipe_recommender:app
    ```
Same as mentioned in Option #1, this will start the Flask application that would then be waiting for the requests from the user to perform prediction.

- One a new terminal and make sure that you are in `personalized_recipe_recommender/deployment`. From inside `personalized_recipe_recommender/deployment`, run 

	```
	python test.py
	```
`test.py` has input values that are passed on to the Flask application and received results are printed on the console window.

***Docker:*** Trigger the Flask application from Docker

- One a new terminal and make sure that you are in `personalized_recipe_recommender/deployment`. From inside `personalized_recipe_recommender/deployment`, run the command below to build the docker image

	```
	docker build -t personalized-recipe-recommeder-service:v1 .
	```

- When Docker image is built successfully, run it using the command

    ```
    docker run -e AWS_ACCESS_KEY_ID=<aws_access_key> -e AWS_SECRET_ACCESS_KEY=<aws_secret_key> -it --rm -p 9696:9696  personalized-recipe-recommeder-service:v1
    ```
Same as mentioned in the last two options, this will start the Flask application that would then be waiting for the requests from the user to perform prediction.

- One a new terminal and make sure that you are in `personalized_recipe_recommender/deployment`. From inside `personalized_recipe_recommender/deployment`, run 

	```
	python test.py
	```

	`test.py` has input values that are passed on to the Flask application and received results are printed on the console window.
