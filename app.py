from flask import Flask, request, jsonify
import joblib
import gdown, os
import pandas as pd
from flask_cors import CORS

app = Flask(__name__)
CORS(app) # Allows requests from react frontend

MODEL_URL = "https://drive.google.com/uc?export=download&id=1PcucFOxs6zNvqoWsiCIeAFTXaygCEyVF&confirm=t"
PIPELINE_URL = "https://drive.google.com/uc?export=download&id=13V3RmCWgn1XRTc6edgN1XtLQchr3dQHP&confirm=t"

# Download model and pipline if not exist
if not os.path.exists("model.pkl"):
    print("Downloading model.pkl ...")
    gdown.download(MODEL_URL, "model.pkl", quiet=False)

if not os.path.exists("pipeline.pkl"):
    print("Downloading pipeline.pkl ...")
    gdown.download(PIPELINE_URL, "pipeline.pkl", quiet=False)

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
    
