# Financial Transaction Multiclass Classifier

This project is a machine learning solution for classifying financial transactions into different categories using XGBoost. The model processes three main inputs: transaction ID, price, and a textual description. The classification pipeline involves several preprocessing steps, including text embeddings from Hugging Face and price normalization using Min-Max Scaler. The model has been deployed using FastAPI for real-time inference.

## Table of Contents
- [Project Overview](#project-overview)
- [Preprocessing](#preprocessing)
- [Model Training](#model-training)
- [Deployment](#deployment)
- [Running the Application](#running-the-application)
- [API Documentation](#api-documentation)

## Project Overview
This project aims to classify financial transactions based on their price, ID, and description into predefined categories. It uses an XGBoost model for multiclass classification.

## Preprocessing
1. **Text Embeddings**: Transaction descriptions are transformed into embeddings using a pre-trained Hugging Face model.
2. **Price Normalization**: The transaction price is scaled between 0 and 1 using the Min-Max Scaler.

## Model Training
The model was trained using the XGBoost classifier after preprocessing the input data. It was retrained with optimized hyperparameters to improve accuracy.

## Deployment
The model is deployed using FastAPI, allowing for real-time transaction classification via REST API.

## Running the Application
To run the application locally, follow these steps:

1. Clone the repository and navigate to the project directory.
2. Install the required dependencies:

    ```bash
    pip install -r requirements.txt
    ```

3. Run the FastAPI server:

    ```bash
    python main.py
    ```

## API Documentation
Once the server is running, you can access the API documentation using FastAPI's Swagger UI by navigating to:
http://127.0.0.1:8000/docs

This interface allows you to test the API endpoints interactively.

