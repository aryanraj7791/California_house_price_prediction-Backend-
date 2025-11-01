# California House Price Prediction – Backend

This repository contains the backend service for the California House Price Prediction project — a full-stack Machine Learning web application that predicts housing prices in California based on input features like location, income, and population metrics.
The backend is built using Flask and serves a REST API endpoint that interacts with the trained regression model and data preprocessing pipeline built with scikit-learn.

## Features

1. REST API built using Flask
2. Serves ML predictions via /predict endpoint
3. Handles preprocessing through serialized pipeline.pkl
4. Deployed on Render for production-ready hosting
5. CORS enabled to allow integration with React frontend on Netlify

## Machine Learning Overview

Algorithm Used: Random Forest Regressor

Model Selection: The Random Forest Regressor was chosen after evaluating multiple regression models, including Decision Tree Regressor and Linear Regressor, based on the mean of RMSE scores obtained through cross-validation.

Dataset: California Housing Dataset (from OpenML)

Objective: Predict median house value based on numerical and categorical features.

### Model Artifacts:

model.pkl → Trained ML model

pipeline.pkl → Preprocessing pipeline (handles scaling, encoding, transformations)

## Future Enhancements

Add user authentication and query history.

Integrate database logging (MongoDB / MySQL).

Implement MLOps for retraining and continuous model improvement.

Containerize with Docker for scalable deployment.

### Author

#### Aryan Raj
Data Science | Machine Learning | Full Stack Developer
