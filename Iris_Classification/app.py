import streamlit as st
import pickle
import numpy as np

# Load the trained model
model = pickle.load(open('model.pkl', 'rb'))

# Set page configuration
st.set_page_config(page_title='Iris Flower Classification', page_icon='🌸', layout='centered')

# App title and description
st.title('🌸 Iris Flower Classification')
st.markdown("""
    Welcome to the **Iris Flower Classification** app!  
    This app uses machine learning to predict the species of an iris flower based on the following input measurements:
""")

# Organize inputs in two columns
col1, col2 = st.columns(2)

with col1:
    sepal_length = st.number_input("Sepal Length (cm)", min_value=0.0, format="%.2f")
    petal_length = st.number_input("Petal Length (cm)", min_value=0.0, format="%.2f")

with col2:
    sepal_width = st.number_input("Sepal Width (cm)", min_value=0.0, format="%.2f")
    petal_width = st.number_input("Petal Width (cm)", min_value=0.0, format="%.2f")

# Center the button
st.markdown("<br>", unsafe_allow_html=True)

# Classify button
if st.button('Classify Flower'):
    if sepal_length and sepal_width and petal_length and petal_width:
        # Create an array with input values
        features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])
        
        # Get the model's prediction
        result = model.predict(features)

        # Output the result with some flair
        st.markdown("<br>", unsafe_allow_html=True)
        st.success('🌺 The predicted species of the flower is:')
        if result == 0:
            st.write('**Iris-setosa**')
        elif result == 1:
            st.write('**Iris-versicolor**')
        else:
            st.write('**Iris-virginica**')
    else:
        st.warning("Please enter all the values to classify the flower.")