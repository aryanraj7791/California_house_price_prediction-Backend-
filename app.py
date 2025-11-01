from flask import Flask, request, jsonify
import joblib
import requests, os
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allows requests from react frontend

MODEL_URL = "https://drive.google.com/uc?export=download&id=10WQt1AEH3yvyJbmq5bqfwCnbtxcTL4u_&confirm=t"
PIPELINE_URL = "https://drive.google.com/uc?export=download&id=1ia4MSIjEPx0D4S5eKCvO1ZnIn8WogHubk"

def download_file(url, filename):
    response = requests.get(url)
    response.raise_for_status()
    with open(filename, "wb") as f:
        f.write(response.content)

# Download model and pipline if not exist
if not os.path.exists("model.pkl"):
    download_file(MODEL_URL, "model.pkl")

if not os.path.exists("pipeline.pkl"):
    download_file(PIPELINE_URL, "pipeline.pkl")

# Load the model and pipeline
model = joblib.load("./model.pkl")
pipeline = joblib.load("./pipeline.pkl")

@app.route("/", methods=["GET"])
def home():
    return "California House Price Prediction API is running!"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()
        
        # Convert json to pandas dataframe
        input_data = pd.DataFrame([data])

        # Transform the input data
        transformed_input = pipeline.transform(input_data)

        # Ensure it's 2D if it is 1D convert it to 2D
        if transformed_input.ndim == 1:
            transformed_input = transformed_input.reshape(1,-1)

        # Predicting price
        prediction = model.predict(transformed_input)

        return jsonify({
            "predicted_price": float(prediction[0])
        })
        
    
    except Exception as e:
        return jsonify({"Error": str(e)}), 500


if __name__ == "__main__":
    app.run(debug=True)
    
