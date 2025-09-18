from fastapi import FastAPI 
import uvicorn 
import xgboost as xgb
import numpy as np
from sklearn.preprocessing import OneHotEncoder
from sentence_transformers import SentenceTransformer
import gdown
import pickle

# Replace this with your Google Drive file ID or shareable link
file_id = '1Cq8psDJwo7CLjIPQwAhkhrmSL1jNRPxz'
output = 'xgboost_model.pkl'

# Use gdown to download the file
gdown.download(f"https://drive.google.com/uc?id={file_id}", output, quiet=False)

# Load the model
with open('xgboost_model.pkl', 'rb') as file:
    updated_model = pickle.load(file)

modeld = SentenceTransformer('sentence-transformers/all-mpnet-base-v2')


classes= ["Bank Transfers - ATM fees",
"Bank Transfers - ATM withdrawals",
"Bank Transfers - Banking rewards",
"Bank Transfers - Foreign transaction fees",
"Bank Transfers - Other bank fees",
"Bank Transfers - Overdraft fees",
"Credit card payment",
"Entertainment - Amusement Parks",
"Entertainment - Casinos and gambling",
"Entertainment - Other",
"Entertainment - TV and movies",
"Entertainment - Video games",
"Food & Drink - Alcohol & Bars",
"Food & Drink - Coffee and tea",
"Food & Drink - Fast food",
"Food & Drink - Food delivery",
"Food & Drink - Groceries",
"Food & Drink - Restaurants",
"Food & Drink - Vending machines",
"General Merchandise - Books",
"General Merchandise - Clothing",
"General Merchandise - Electronics",
"General Merchandise - Houseware",
"General Merchandise - Online Marketplaces",
"General Merchandise - Pet supplies",
"General Merchandise - Product refunds",
"General Merchandise - Sporting goods",
"General Services - Auto insurance",
"General Services - Cloud storage",
"General Services - Education",
"General Services - Health + Fitness",
"General Services - Health insurance",
"General Services - Home Repair + Maintenance",
"General Services - Other insurance",
"General Services - Other non-entertainment",
"General Services - Other non-entertainment online subscriptions",
"General Services - Service refunds",
"General Services - Shipping + Postage",
"Government + Non-Profit - Other",
"Government + Non-Profit - Other donations",
"Government + Non-Profit - Political donations",
"Government + Non-Profit - Taxes",
"Income - Other",
"Income - Salary",
"Income - Tax refund",
"Loans - Car loan payments",
"Loans - Credit card payment",
"Loans - Mortgage Payments",
"Loans - Mortgage payments",
"Loans - Personal loans",
"Medical - Pharmacies and supplements",
"Medical - Veterinary services",
"Medical - Veterinary services",
"Rent & Utilities - Internet",
"Rent & Utilities - Mobile Phone",
"Rent & Utilities - Rent",
"Rent & Utilities - Water",
"Subscription - Others",
"Supermarkets - Groceries",
"Transportation - Automobile maintenance and fees",
"Transportation - Gas",
"Transportation - Parking",
"Transportation - Public Transport",
"Transportation - Taxis and rideshares",
"Transportation - Tolls",
"Travel - Flights",
"Travel - Lodging",
"Travel - Rental cars"]
def encode_Ids(column_vector, categories):
# Create an instance of the OneHotEncoder
    encoder = OneHotEncoder(categories=[categories], sparse=False)

# Perform one-hot encoding on the column vector
    encoded_vector = encoder.fit_transform(column_vector)
    return encoded_vector
def new_classification(ID, amount, description):
    IDs = ['CPA.1', 'CPA_MP.1', 'HP.1', 'IPA.1', 'SEW0C_HI.1', 'SEW0C_LI.1', 'SEWC_CL.1', 'SEWC_P.1', 'SEWC_PA.1',
           'SEWC_P_PL.1', 'TA.1', 'US.1']

    # Perform one-hot encoding on the first column
    encoded_column1 = encode_Ids(np.atleast_2d(ID), IDs)  # Reshape ID as 2D array
    column2 = (np.atleast_2d(float((amount))) - 4) / (6400 - 4)  # Convert amount to float and reshape as 1D array
    # Encode the description using modeld
    description_embedding = modeld.encode([description])

    # Combine the columns to form the updated matrix
    updated_matrix = np.concatenate((encoded_column1, column2.T, description_embedding), axis=1)  # Reshape column2
    # Convert the new data into DMatrix format
    dnew = xgb.DMatrix(updated_matrix)
    # Make predictions on the new data
    predictions = updated_model.predict(dnew)
    # Convert predictions from float32 to integers
    predictions = predictions.astype(int)

    # Inverse transform the encoded predictions to get the original category names
    predicted_category = classes[predictions[0]]  # Get the category with the highest probability
    return predicted_category

app = FastAPI()

@app.get("/")
def home():
    return {"health_check": "OK"}

@app.get("/predict")
def predict(ID: str,amount: str,description:str):
    
    category = new_classification(ID,amount,description)
    return {"category": category}

if __name__=='__main__':
    uvicorn.run(app)
