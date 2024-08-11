import requests
import json

user_ingredients = "chicken garlic onion salt pepper"
data = {"ingredients": user_ingredients}

url = 'http://localhost:9696/recommendations'
response = requests.post(url, json=data)

# # Print the response status code and content
# print(response.status_code)
# print(response.json())  # This should print the result from the Flask app
# Print the JSON response in a pretty format
print("Recommendations:")
print(json.dumps(response.json(), indent=4))