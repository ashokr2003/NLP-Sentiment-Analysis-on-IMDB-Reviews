import streamlit as st
import joblib

# Load trained model & vectorizer
model = joblib.load("sentiment_model.pkl")
vectorizer = joblib.load("vectorizer.pkl")

# Set Streamlit Page Config
st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="🔍",
    layout="centered",
    initial_sidebar_state="collapsed",
)

# Custom Styling
st.markdown(
    """
    <style>
    .main {
        background-color: #f5f5f5;
    }
    .stTextArea textarea {
        font-size: 18px;
        padding: 10px;
    }
    .stButton button {
        background-color: #008CBA;
        color: white;
        font-size: 18px;
        padding: 10px 24px;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# Streamlit UI
st.title("🔍 Sentiment Analysis App")
st.markdown("### 🎬 Analyze the sentiment of movie reviews")
st.write("Enter a sentence below, and I will predict whether it's **Positive** or **Negative**!")

# User Input
user_input = st.text_area("📝 Enter your text here:", height=150)

# Button for Prediction
if st.button("🔍 Predict Sentiment"):
    if user_input:
        # Convert input text into numbers
        transformed_text = vectorizer.transform([user_input])
        prediction = model.predict(transformed_text)[0]

        # Show Result
        if prediction == 1:
            st.success("😊 **Positive Sentiment!**")
        else:
            st.error("😡 **Negative Sentiment!**")
    else:
        st.warning("⚠️ Please enter some text to analyze!")

# Footer
st.markdown("---")
st.markdown("💡 **Built with Python, Machine Learning, and Streamlit**")
