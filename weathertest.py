import streamlit as st
from deep_translator import GoogleTranslator

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
#translator = Translator()

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
# def translate_text(text, target_lang):
#     try:
#         translation = translator.translate(text, dest=target_lang)
#         return translation.text
#     except Exception as e:
#         st.error(f"Translation error: {e}")
#         return text




def translate_text(text, target_lang):
    try:
        return GoogleTranslator(source='auto', target=target_lang).translate(text)
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

# Translate the main navigation options
app_mode = st.sidebar.selectbox(
    translate_text("Select Page", selected_lang_code),
    [nav_home, nav_about, nav_disease_recognition, nav_field, nav_nearby_shops, nav_weather_monitor]
)

# Custom CSS for theme adjustments
st.markdown("""
    <style>
        /* Background and font styling for a clean and modern theme */
.main {
  background-color: #f5f9f7; /* Light gray background */
  font-family: 'Open Sans', sans-serif; /* Use Open Sans for body text */
  padding: 20px; /* Add some padding for better breathing room */
}

/* Styling for headers */
.main-header {
  font-size: 40px;
  font-weight: bold;
  color: #388E3C; /* Earthy green color */
  text-align: center;
}

.main-subheader {
  font-size: 30px;
  font-weight: 500;
  color: #2E7D32; /* Darker green for subheadings */
  text-align: center;
}

/* Sidebar styling */
.sidebar .sidebar-content {
  background-color: #E0F2EF; /* Lighter green for sidebar */
}

/* Buttons */
.stButton > button {
  background-color: #66BB6A; /* Green button color */
  color: white;
  border: none;
  border-radius: 5px;
  padding: 10px 20px;
  font-size: 16px;
  cursor: pointer; /* Indicate clickable button */
  transition: background-color 0.2s ease-in-out; /* Add hover effect */
}

.stButton > button:hover {
  background-color: #388E3C; /* Darker green on hover */
}

/* Image styling */
.stImage {
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1); /* Add subtle drop shadow */
}

/* Custom class for description text (optional) */
.du {
  font-size: 18px;
  color: #555; /* Grayish color for descriptions */
  margin-bottom: 15px; /* Add some margin below */
}

/* Media queries for responsiveness (example) */
@media (max-width: 768px) {
  .main-header {
    font-size: 32px;
  }
  .main-subheader {
    font-size: 24px;
  }
}
.div{
    font-size:18px;
}
.pc{
    font-size:25px;
}

.field_title{
  
}

.alert_icon{
   border: 2px solid green;
}
    </style>
    """, unsafe_allow_html=True)

# Add JavaScript for automatic location detection
st.markdown("""
    <script>
    function getLocation() {
        if (navigator.geolocation) {
            navigator.geolocation.getCurrentPosition(function(position) {
                var lat = position.coords.latitude;
                var lon = position.coords.longitude;
                document.getElementById("location-input").value = lat + "," + lon;
                document.getElementById("location-form").submit();
            }, function(error) {
                document.getElementById("location-input").value = "Location detection failed.";
                document.getElementById("location-form").submit();
            });
        } else {
            document.getElementById("location-input").value = "Geolocation is not supported by this browser.";
            document.getElementById("location-form").submit();
        }
    }
    window.onload = getLocation;
    </script>
    <form id="location-form" action="#" method="post">
        <input type="hidden" id="location-input" name="location" value="">
    </form>
    """, unsafe_allow_html=True)

# Main Page
if app_mode == nav_home:
    st.markdown(f'<h2 class="main-header">{translate_text("🌿 PLANT DISEASE RECOGNITION SYSTEM", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(f'<p class="main-subheader">{translate_text("Welcome to the Plant Disease Recognition System!", selected_lang_code)}</p>', unsafe_allow_html=True)

    st.markdown(f'<div class= "div">{translate_text("""
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### 🚀 How It Works
    1. Upload Image: Go to the Disease Recognition page and upload an image of a plant with suspected diseases.
    2. Analysis: Our system will process the image using advanced algorithms to identify potential diseases.
    3. Results: View the results and recommendations for further action.

    ### 🛠 Why Choose Us?
    - Accuracy: Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - User-Friendly: Simple and intuitive interface for seamless user experience.
    - Fast and Efficient: Receive results in seconds, allowing for quick decision-making.

    ### 🔍 Get Started
    Click on the Disease Recognition page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### 👥 About Us
    Learn more about the project, our team, and our goals on the About page.
    """, selected_lang_code)}</div>', unsafe_allow_html=True)

# Nearby Shops
elif app_mode == nav_nearby_shops:
    st.markdown(f'<h2 class="main-header">🛒 Nearby Shops for Plant Care</h2>', unsafe_allow_html=True)
    st.markdown("""
        Find the nearest shops that sell plant care products like pesticides, fertilizers, and seeds.
        - Shop 1: Green Garden Supplies, 123 Plant Lane
        - Shop 2: Healthy Harvest, 456 Agriculture Ave
        - Shop 3: Nature's Best, 789 Flora Street
    """)
    st.write("For more details, search for plant care shops near your location!")

# About Project
elif app_mode == nav_about:
    st.markdown(f'<h2 class="main-header">{translate_text("About the Project", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.markdown("""
        This project is designed to assist farmers and plant enthusiasts in identifying plant diseases through image recognition. The model uses advanced machine learning techniques to provide accurate results quickly. The goal is to enhance agricultural productivity and ensure healthier crops.
    """)

# Disease Recognition
elif app_mode == nav_disease_recognition:
    st.markdown(f'<h2 class="main-header">{translate_text("🌱 Disease Recognition", selected_lang_code)}</h2>', unsafe_allow_html=True)
    uploaded_file = st.file_uploader("Upload an image of the plant", type=["jpg", "jpeg", "png"])
    if uploaded_file:
        result = model_prediction(uploaded_file)
        st.image(uploaded_file, use_column_width=True)
        st.write(f"Prediction Result: {result}")

# # Detection Page
elif app_mode == nav_field:
    st.markdown(f'<h2 class="main-header">{translate_text("🔍 Detection", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.write("This page is for detecting various plant-related parameters.")

# Weather Monitoring
elif app_mode == nav_weather_monitor:
    location = st.text_input("Enter your city")
    if location:
        weather_data = get_weather_data(location)
        if weather_data:
            st.write(f"Weather in {location}:")
            st.write(f"Temperature: {weather_data['main']['temp']}°C")
            st.write(f"Weather: {weather_data['weather'][0]['description']}")
            st.write(f"Humidity: {weather_data['main']['humidity']}%")
        else:
            st.write("Could not retrieve weather data. Please check the city name.")