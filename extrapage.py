import streamlit as st
from googletrans import Translator
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests

# Load the model once using caching
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"
    if not os.path.exists(model_path):
        return None
    return tf.keras.models.load_model(model_path)

model = load_model()

# TensorFlow Model Prediction
def model_prediction(test_image):
    image = Image.open(test_image).convert('RGB')
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0  # Normalize the image
    input_arr = np.expand_dims(input_arr, axis=0)  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Function to fetch weather data
def get_weather_data(location, api_key="9f36996dbcf80db71dcbec83d14459f0"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    else:
        return None

# Initialize the translator
translator = Translator()

# Available languages
languages = {
    'en': 'English',
    'hi': 'Hindi',
    'ta': 'Tamil',
    'te': 'Telugu',
    'mr': 'Marathi',
    'gu': 'Gujarati',
    'bn': 'Bengali',
    'kn': 'Kannada',
    'ml': 'Malayalam',
    'pa': 'Punjabi'
}

# Sidebar
st.sidebar.title("Dashboard")
st.sidebar.markdown("Navigate through the pages to explore the app!")
selected_language = st.sidebar.selectbox("Select Language", list(languages.values()))
selected_lang_code = list(languages.keys())[list(languages.values()).index(selected_language)]

# Function to translate text
def translate_text(text, target_lang):
    try:
        translation = translator.translate(text, dest=target_lang)
        return translation.text
    except Exception as e:
        st.error(f"Translation error: {e}")
        return text

# Translate navigation options
nav_home = translate_text("Home", selected_lang_code)
nav_about = translate_text("About", selected_lang_code)
nav_disease_recognition = translate_text("Disease Recognition", selected_lang_code)
nav_field = translate_text("Detection", selected_lang_code)
nav_nearby_shops = translate_text("Nearby Shops", selected_lang_code)
nav_weather_monitor = translate_text("Weather Monitoring", selected_lang_code)
nav_diseases = translate_text("Diseases", selected_lang_code)

# Translate the main navigation options
app_mode = st.sidebar.selectbox(
    translate_text("Select Page", selected_lang_code),
    [nav_home, nav_about, nav_disease_recognition, nav_field, nav_nearby_shops, nav_weather_monitor, nav_diseases]
)

# Custom CSS for theme adjustments
st.markdown("""
    <style>
        .main {
          background-color: #f5f9f7;
          font-family: 'Open Sans', sans-serif;
          padding: 20px;
        }
        .main-header {
          font-size: 40px;
          font-weight: bold;
          color: #388E3C;
          text-align: center;
        }
        .main-subheader {
          font-size: 30px;
          font-weight: 500;
          color: #2E7D32;
          text-align: center;
        }
        .stButton > button {
          background-color: #66BB6A;
          color: white;
          border: none;
          border-radius: 5px;
          padding: 10px 20px;
          font-size: 16px;
          cursor: pointer;
          transition: background-color 0.2s ease-in-out;
        }
        .stButton > button:hover {
          background-color: #388E3C;
        }
        .container {
          display: flex;
          justify-content: center;
          align-items: center;
          flex-direction: column;
          height: 100vh;
          background-color: #f0f4f7;
          padding: 20px;
        }
        .field_title {
          font-family: 'Arial', sans-serif;
          font-size: 32px;
          font-weight: bold;
          color: #2c3e50;
          margin-bottom: 20px;
          display: flex;
          align-items: center;
          justify-content: center;
          text-align: center;
          padding: 10px 20px;
          background: linear-gradient(90deg, #ff6e7f, #bfe9ff);
          border-radius: 10px;
          box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);
          transition: transform 0.3s ease;
        }
        .map {
          background-color: rgb(230, 255, 240);
          border: 2px solid #2c3e50;
          border-radius: 15px;
          padding-left: 30px;
          margin-left: 50px;
          background-size: cover;
          background-position: center;
          width: 50vw;
          height: 50vw;
          max-width: 600px;
          max-height: 500px;
          min-width: 150px;
          min-height: 150px;
          position: relative;
          box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.15);
          transition: box-shadow 0.3s ease-in-out, transform 0.3s ease-in-out;
        }
        .map:hover {
          box-shadow: 0px 12px 30px rgba(0, 0, 0, 0.2);
          transform: scale(1.02);
        }
        .blinking-alert {
          width: 30px;
          height: 30px;
          background-color: red;
          position: absolute;
          top: 10px;
          right: 10px;
          border-radius: 50%;
          animation: blink 1s infinite;
          box-shadow: 0px 4px 10px rgba(255, 0, 0, 0.5);
        }
        @keyframes blink {
          0% { opacity: 1; }
          50% { opacity: 0; }
          100% { opacity: 1; }
        }
    </style>
""", unsafe_allow_html=True)

# Main Page
if app_mode == nav_home:
    st.markdown(f'<h2 class="main-header">{translate_text("🌿 PLANT DISEASE RECOGNITION SYSTEM", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(f'<p class="main-subheader">{translate_text("Welcome to the Plant Disease Recognition System!", selected_lang_code)}</p>', unsafe_allow_html=True)

elif app_mode == nav_diseases:
    st.markdown(f'<h2 class="main-header">{translate_text("Select Plant Disease", selected_lang_code)}</h2>', unsafe_allow_html=True)

elif app_mode == nav_nearby_shops:
    st.markdown(f'<h2 class="main-header">🛒 {translate_text("Nearby Shops for Plant Care", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.markdown("""
        - *Shop 1:* Green Garden Supplies, 123 Plant Lane
        - *Shop 2:* Healthy Harvest, 456 Agriculture Ave
        - *Shop 3:* Nature's Best, 789 Flora Street
    """)
    
elif app_mode == nav_about:
    st.markdown(f'<h2 class="main-header">{translate_text("About", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.markdown(translate_text("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. 
                #### 📊 Content
                1. **Train:** 70,295 images
                2. **Test:** 33 images
                3. **Validation:** 17,572 images
                """, selected_lang_code))

elif app_mode == nav_weather_monitor:
    st.markdown(f'<h2 class="main-header">🌍 {translate_text("Weather Monitoring", selected_lang_code)}</h2>', unsafe_allow_html=True)
    location = st.text_input("Enter a location to monitor:")
    if location and st.button("Get Weather"):
        weather_data = get_weather_data(location)
        if weather_data:
            st.write(f"*Location:* {weather_data['name']}")
            st.write(f"*Temperature:* {weather_data['main']['temp']} °C")
            st.write(f"*Humidity:* {weather_data['main']['humidity']} %")
            st.write(f"*Weather Description:* {weather_data['weather'][0]['description'].capitalize()}")
            temp_threshold = 30
            humidity_threshold = 70
            if weather_data['main']['temp'] > temp_threshold:
                st.warning(f"⚠ {translate_text('Alert: The temperature is above the threshold!', selected_lang_code)}")
            if weather_data['main']['humidity'] > humidity_threshold:
                st.warning(f
