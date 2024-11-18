import streamlit as st
import numpy as np
import pickle

st.header("Flower Predictor!")
st.image ("https://encrypted-tbn1.gstatic.com/shopping?q=tbn:ANd9GcS7VH0Lguft1r9beEy-apsvna54L3_T3w5TM_GzMeYLXyyQNDUAeIvtM1IxSf5HFAs9Fnrj0PnNdd4t17M-oz4pbA6QpX4bT3m-QyfdZFTF0zXSRx38E0ST&usqp=CAc")
st.write ("Enter your info on your  flower below")
flower_form = st.form("FlowerForm")

sepal_length = st.number_input('Sepal Length')
sepal_width = st.number_input('Sepal Width')
petal_length = st.number_input('Petal Length')
petal_width = st.number_input('Petal Width')

new_flower = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

with open ('iris_class.pkl', 'rb') as f:
    modle = pickle.load(f)

prediction = modle.predict(new_flower)
st.write(prediction)
