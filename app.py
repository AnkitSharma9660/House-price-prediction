import streamlit as st
import pickle
import pandas as pd
import os

# load model
model = pickle.load(open('ridge_model.pkl','rb'))

# load dataset
df = pd.read_csv("house_prediction.csv")

# take unique locations
locations = sorted(df['location'].dropna().unique())

st.title("Bangalore House Price Predictor")

location = st.selectbox("Location", locations)
sqft = st.number_input("Total Sqft")
bath = st.number_input("Bathrooms")
bhk = st.number_input("BHK")

if st.button("Predict Price"):
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location','total_sqft','bath','bhk'])
    
    price = model.predict(input_df)[0]
    st.success(f"Estimated Price : {price:.2f} Lakhs")

st.write("Current Working Directory:", os.getcwd())