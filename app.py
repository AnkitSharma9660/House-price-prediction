import streamlit as st
import pickle
import pandas as pd

model = pickle.load(open('ridge_model.pkl','rb'))

st.title("Bangalore House Price Predictor")

location = st.selectbox("Location", ["Indira Nagar","Whitefield","HSR Layout"])
sqft = st.number_input("Total Sqft")
bath = st.number_input("Bathrooms")
bhk = st.number_input("BHK")

if st.button("Predict Price"):
    input_df = pd.DataFrame([[location, sqft, bath, bhk]],
                            columns=['location','total_sqft','bath','bhk'])
    
    price = model.predict(input_df)[0]
    st.success(f"Estimated Price : {price:.2f} Lakhs")
