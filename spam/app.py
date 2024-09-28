import streamlit as st
import pickle
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('punkt_tab')
import string
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
from sklearn.preprocessing import MinMaxScaler

ps = PorterStemmer()

# Preprocessing function for transforming the email text
def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    
    y = [i for i in text if i.isalnum()]
    
    y = [i for i in y if i not in stopwords.words('english') and i not in string.punctuation]
    
    y = [ps.stem(i) for i in y]
    
    return " ".join(y)

# Load the pre-trained model, tf-idf vectorizer, and the MinMaxScaler
tfidf = pickle.load(open('vectorizer.pkl', 'rb'))
model = pickle.load(open('model.pkl', 'rb'))
scaler = pickle.load(open('scaler.pkl', 'rb'))  

# Streamlit App UI
st.set_page_config(page_title="Email Spam Classifier", page_icon="📧")

# Adding title and description
st.title("📧 Email Spam Classifier")
st.write("This application helps classify emails as **Spam** or **Not Spam** using a machine learning model.")

# Input field for the email
st.markdown("### Please enter the email content below:")
input_email = st.text_area("Enter the email text", height=200, placeholder="Type your email here...")

# Button for predicting the result
if st.button('Classify Email'):
    if input_email:
        # Preprocess the input
        transformed_email = transform_text(input_email)
        
        # Vectorize the input text
        vector_input = tfidf.transform([transformed_email])
        
        # Apply Min-Max Scaling
        scaled_input = scaler.transform(vector_input.toarray())  # Ensure the input is scaled correctly

        # Predict using the model
        result = model.predict(scaled_input)[0]
        
        # Display the result
        if result == 1:
            st.markdown("<h2 style='color:red;'>🚨 This email is classified as: Spam</h2>", unsafe_allow_html=True)
        else:
            st.markdown("<h2 style='color:green;'>✅ This email is classified as: Not Spam</h2>", unsafe_allow_html=True)
    else:
        st.warning("Please enter some email content to classify.")
        