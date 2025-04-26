import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import json
from groq import Groq  # Make sure Groq is installed: pip install groq
from PIL import Image
import os
import requests
from deep_translator import GoogleTranslator
from googletrans import Translator
import google.generativeai as genai
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Initialize Gemini API
try:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
    if not GEMINI_API_KEY:
        st.error("Gemini API key not found in environment variables")
        genai.configure(api_key="YOUR_GEMINI_API_KEY")  # Replace with your actual API key
    else:
        genai.configure(api_key=GEMINI_API_KEY)
    st.success("Gemini API initialized successfully")
except Exception as e:
    st.error(f"Error initializing Gemini API: {str(e)}")

# Add language selection at the top
languages = {
    "English": "en",
    "Hindi": "hi",
    "Telugu": "te",
    "Tamil": "ta",
    "Kannada": "kn",
    "Malayalam": "ml",
    "Marathi": "mr",
    "Gujarati": "gu",
    "Bengali": "bn",
    "Punjabi": "pa"
}

# Function to translate text
def translate_text(text, target_lang):
    if target_lang == "en":
        return text
    
    try:
        translator = GoogleTranslator(source='auto', target=target_lang)
        translated = translator.translate(text)
        return translated
    except Exception as e:
        st.error(f"Translation error: {str(e)}")
        return text

# Language selection in sidebar
selected_language = st.sidebar.selectbox("Select Language", list(languages.keys()), key="language_selector")

# Function to get translated text
def get_text(key, default_text):
    if selected_language == "English":
        return default_text
    else:
        return translate_text(default_text, languages[selected_language])

# Update the sidebar title with translation
st.sidebar.title(get_text("dashboard", "Dashboard"))

# Update the navigation options with translations
nav_options = [
    get_text("home", "Home"),
    get_text("about", "About"),
    get_text("disease_recognition", "Disease Recognition"),
    get_text("nearby_shops", "Nearby Shops"),
    get_text("weather_monitoring", "Weather Monitoring")
]

app_mode = st.sidebar.selectbox(get_text("select_page", "Select Page"), nav_options, key="page_selector")

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
    predictions = model.predict(input_arr)
    return np.argmax(predictions)  # Return index of max element

# Function to get weather data using OpenWeather API
def get_weather_data(location):
    try:
        # Use the API key from secrets.toml
        api_key = st.secrets["OPENWEATHER_API_KEY"]
        
        # First try with the exact location
        url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
        response = requests.get(url)
        
        # If location not found, try with state name
        if response.status_code != 200:
            # Add common Indian states for better matching
            states = ["Andhra Pradesh", "Telangana", "Maharashtra", "Karnataka", "Tamil Nadu", 
                     "Kerala", "Delhi", "West Bengal", "Gujarat", "Rajasthan", "Uttar Pradesh"]
            
            for state in states:
                location_with_state = f"{location}, {state}"
                url = f"http://api.openweathermap.org/data/2.5/weather?q={location_with_state}&appid={api_key}&units=metric"
                response = requests.get(url)
                if response.status_code == 200:
                    break
        
        if response.status_code == 200:
            data = response.json()
            return {
                "name": data["name"],
                "main": {
                    "temp": data["main"]["temp"],
                    "humidity": data["main"]["humidity"]
                },
                "weather": [{"description": data["weather"][0]["description"]}]
            }
        else:
            st.error(f"Could not find weather data for '{location}'. Please try a nearby city or check the spelling.")
            return None
            
    except Exception as e:
        st.error(f"Error fetching weather data: {str(e)}")
        return None

# Function to get treatment suggestion from Gemini API
def get_treatment_suggestion(disease_name):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"recomend 5 retail Agro Farms nearby vasavi college of engineering ,ibrahimbagh ,hydereabad .add the distant how far is it. only give shop names and distant in km")
        return response.text
    except Exception as e:
        st.error(f"Error getting treatment suggestion: {str(e)}")
        return """1. Green Valley Agro Farms - 2.5 km
2. Nature's Harvest - 3.2 km
3. Organic Solutions - 4.1 km
4. Farm Fresh Supplies - 5.3 km
5. Agro Tech Center - 6.7 km"""

# Function to get treatment suggestion from Gemini API
def get_treatment_suggestion1(disease_name):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"recomend 3 medicines to cure this {disease_name} disease and give cost of it. give me in boldpoint only only point")
        return response.text
    except Exception as e:
        st.error(f"Error getting medicine suggestion: {str(e)}")
        return """• Mancozeb 75% WP - ₹250 per 500g
• Chlorothalonil 75% WP - ₹300 per 500g
• Copper Oxychloride 50% WP - ₹200 per 500g"""

# Function to get nearby shops using Gemini API
def get_nearby_shops(disease, location):
    try:
        model = genai.GenerativeModel('gemini-pro')
        response = model.generate_content(f"Find nearby shops that fertilizers for the disease: {disease}, Location: is {location}.")
        return response.text
    except Exception as e:
        st.error(f"Error getting nearby shops: {str(e)}")
        return """1. Green Valley Agro Farms - 2.5 km
2. Nature's Harvest - 3.2 km
3. Organic Solutions - 4.1 km
4. Farm Fresh Supplies - 5.3 km
5. Agro Tech Center - 6.7 km"""

# Streamlit App Structure
# Remove the duplicate sidebar title and app_mode selection
# st.sidebar.title("Dashboard")
# app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Nearby Shops","Weather Monitoring"])

# Remove the duplicate language selection
# lang_options = ["English", "Hindi", "Telugu"]
# default_lang = "English"
# selected_lang = st.sidebar.selectbox("Choose Language", lang_options, index=lang_options.index(default_lang))

# Add custom CSS
st.markdown("""
    <style>
        /* Import retro font */
        @import url('https://fonts.googleapis.com/css2?family=VT323&display=swap');
        @import url('https://fonts.googleapis.com/css2?family=Press+Start+2P&display=swap');
        
        /* Base theme - Retro Green */
        .main {
            background-color: #0a1f0a !important;
            color: #00ff00 !important;
            font-family: 'VT323', monospace !important;
        }
        
        /* Force dark background on all Streamlit elements */
        .stApp, .main .block-container, div[data-testid="stAppViewContainer"], div[data-testid="stHeader"] {
            background-color: #0a1f0a !important;
            color: #00ff00 !important;
        }
        
        /* Override any white backgrounds */
        .element-container, .stMarkdown, .stButton, .stSelectbox {
            background-color: transparent !important;
        }
        
        /* Ensure text is visible and larger, but without shadow */
        p, h1, h2, h3, h4, span, div {
            color: #00ff00 !important;
            font-family: 'VT323', monospace !important;
            text-shadow: none !important;
            font-size: 1.5rem !important; /* Increased base font size */
        }
        
        /* Headers with retro styling and larger size, but without shadow */
        h1 {
            font-family: 'Press Start 2P', cursive !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            border-bottom: 2px solid #00ff00 !important;
            padding-bottom: 10px !important;
            margin-bottom: 20px !important;
            font-size: 2.5rem !important; /* Larger header size */
            text-shadow: none !important;
            animation: glitch 1s linear infinite, fadeIn 1s ease-in-out !important;
            position: relative !important;
            white-space: normal !important; /* Allow text to wrap */
            overflow: visible !important; /* Ensure text is visible */
            width: 100% !important; /* Full width */
        }
        
        h2 {
            font-family: 'Press Start 2P', cursive !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            border-bottom: 2px solid #00ff00 !important;
            padding-bottom: 10px !important;
            margin-bottom: 20px !important;
            font-size: 2rem !important; /* Larger subheader size */
            text-shadow: none !important;
            animation: pulse 2s infinite, slideIn 1s ease-out !important;
            position: relative !important;
            white-space: normal !important; /* Allow text to wrap */
            overflow: visible !important; /* Ensure text is visible */
            width: 100% !important; /* Full width */
        }
        
        h3 {
            font-family: 'Press Start 2P', cursive !important;
            text-transform: uppercase !important;
            letter-spacing: 2px !important;
            border-bottom: 2px solid #00ff00 !important;
            padding-bottom: 10px !important;
            margin-bottom: 20px !important;
            font-size: 1.8rem !important; /* Larger h3 size */
            text-shadow: none !important;
            animation: shake 0.5s ease-in-out, fadeIn 1.5s ease-in-out !important;
            position: relative !important;
            white-space: normal !important; /* Allow text to wrap */
            overflow: visible !important; /* Ensure text is visible */
            width: 100% !important; /* Full width */
        }
        
        /* Sidebar styling with larger text */
        .sidebar .sidebar-content {
            background-color: #0d2b0d !important;
            color: #00ff00 !important;
            border-right: 3px solid #00ff00 !important;
            padding-right: 10px !important; /* Add padding to ensure text is visible */
        }
        
        /* Sidebar title with reduced size and no shadow */
        .sidebar .sidebar-content h1 {
            font-family: 'Press Start 2P', cursive !important;
            font-size: 1.1rem !important; /* Further reduced sidebar title size */
            text-align: center !important;
            margin-bottom: 30px !important;
            text-shadow: none !important;
            white-space: normal !important; /* Allow text to wrap */
            overflow: visible !important; /* Ensure text is visible */
            width: 100% !important; /* Full width */
            padding-right: 5px !important; /* Add padding to ensure text is visible */
        }
        
        /* Sidebar selectbox with larger text but no shadow */
        .sidebar .sidebar-content .stSelectbox > div > div > select {
            background-color: #0d2b0d !important;
            color: #00ff00 !important;
            border: 2px solid #00ff00 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.1rem !important; /* Further reduced sidebar select text size */
            padding: 10px !important;
            text-shadow: none !important;
            width: 100% !important; /* Full width */
            padding-right: 5px !important; /* Add padding to ensure text is visible */
        }
        
        /* Button styling with larger text but no shadow */
        .stButton > button {
            background-color: #0d2b0d !important;
            color: #00ff00 !important;
            border: 2px solid #00ff00 !important;
            font-family: 'Press Start 2P', cursive !important;
            font-size: 1.2rem !important; /* Larger button text */
            padding: 15px 25px !important; /* Larger button padding */
            text-transform: uppercase !important;
            letter-spacing: 1px !important;
            transition: all 0.3s ease !important;
            box-shadow: 0 0 10px #00ff00 !important;
            text-shadow: none !important;
        }
        
        .stButton > button:hover {
            background-color: #00ff00 !important;
            color: #0a1f0a !important;
            box-shadow: 0 0 20px #00ff00 !important;
        }
        
        /* Input fields with larger text but no shadow */
        .stTextInput > div > div > input {
            background-color: #0d2b0d !important;
            color: #00ff00 !important;
            border: 2px solid #00ff00 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.8rem !important; /* Larger input text */
            padding: 15px !important; /* Larger input padding */
            text-shadow: none !important;
        }
        
        /* Select boxes with larger text but no shadow */
        .stSelectbox > div > div > select {
            background-color: #0d2b0d !important;
            color: #00ff00 !important;
            border: 2px solid #00ff00 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.8rem !important; /* Larger select text */
            padding: 15px !important; /* Larger select padding */
            text-shadow: none !important;
        }
        
        /* File uploader with larger text but no shadow */
        .stFileUploader > div {
            background-color: #0d2b0d !important;
            color: #00ff00 !important;
            border: 2px solid #00ff00 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.8rem !important; /* Larger file uploader text */
            padding: 15px !important; /* Larger file uploader padding */
            text-shadow: none !important;
        }
        
        /* Success/Error messages with larger text but no shadow */
        .stSuccess {
            background-color: #0d2b0d !important;
            color: #00ff00 !important;
            border: 2px solid #00ff00 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.8rem !important; /* Larger success text */
            padding: 20px !important; /* Larger success padding */
            border-radius: 5px !important;
            box-shadow: 0 0 10px #00ff00 !important;
            text-shadow: none !important;
        }
        
        .stError {
            background-color: #0d2b0d !important;
            color: #ff0000 !important;
            border: 2px solid #ff0000 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.8rem !important; /* Larger error text */
            padding: 20px !important; /* Larger error padding */
            border-radius: 5px !important;
            box-shadow: 0 0 10px #ff0000 !important;
            text-shadow: none !important;
        }
        
        .stWarning {
            background-color: #0d2b0d !important;
            color: #ffff00 !important;
            border: 2px solid #ffff00 !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.8rem !important; /* Larger warning text */
            padding: 20px !important; /* Larger warning padding */
            border-radius: 5px !important;
            box-shadow: 0 0 10px #ffff00 !important;
            text-shadow: none !important;
        }
        
        .stInfo {
            background-color: #0d2b0d !important;
            color: #00ffff !important;
            border: 2px solid #00ffff !important;
            font-family: 'VT323', monospace !important;
            font-size: 1.8rem !important; /* Larger info text */
            padding: 20px !important; /* Larger info padding */
            border-radius: 5px !important;
            box-shadow: 0 0 10px #00ffff !important;
            text-shadow: none !important;
        }
        
        /* Remove card styling to eliminate unnecessary blocks */
        .stMarkdown > div {
            background-color: transparent !important;
            border: none !important;
            padding: 0 !important;
            margin-bottom: 20px !important;
            box-shadow: none !important;
        }
        
        /* Image container with larger border */
        .stImage > div {
            border: 3px solid #00ff00 !important;
            border-radius: 5px !important;
            padding: 15px !important;
            box-shadow: 0 0 15px #00ff00 !important;
        }
        
        /* Add CRT screen effect */
        .main::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.15),
                rgba(0, 0, 0, 0.15) 1px,
                transparent 1px,
                transparent 2px
            );
            pointer-events: none;
            z-index: 999;
        }
        
        /* Add scanline effect */
        .main::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                rgba(0, 255, 0, 0.03) 50%,
                rgba(0, 0, 0, 0.05) 50%
            );
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 999;
        }
        
        /* Remove glow animation */
        h1, h2, h3 {
            animation: none !important;
        }
        
        /* Add hover effect for navigation */
        .sidebar .sidebar-content .stSelectbox > div > div > select:hover {
            background-color: #00ff00 !important;
            color: #0a1f0a !important;
            box-shadow: 0 0 15px #00ff00 !important;
        }
        
        /* Add retro grid background */
        .main {
            background-image: 
                linear-gradient(rgba(0, 255, 0, 0.1) 1px, transparent 1px),
                linear-gradient(90deg, rgba(0, 255, 0, 0.1) 1px, transparent 1px);
            background-size: 20px 20px;
        }
        
        /* Make navbar text larger and more prominent but without shadow */
        .sidebar .sidebar-content .stSelectbox > div > div > select option {
            font-size: 1.1rem !important;
            padding: 10px !important;
            text-shadow: none !important;
        }
        
        /* Fix for sidebar width to ensure text is visible */
        .sidebar .sidebar-content {
            width: 100% !important;
            max-width: 100% !important;
            overflow: visible !important;
        }
        
        /* Fix for sidebar selectbox to ensure text is visible */
        .sidebar .sidebar-content .stSelectbox > div {
            width: 100% !important;
            max-width: 100% !important;
            overflow: visible !important;
        }
        
        /* Fix for sidebar selectbox options to ensure text is visible */
        .sidebar .sidebar-content .stSelectbox > div > div > select option {
            width: 100% !important;
            max-width: 100% !important;
            overflow: visible !important;
            padding-right: 10px !important;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            h1 { font-size: 2rem !important; }
            h2 { font-size: 1.8rem !important; }
            h3 { font-size: 1.6rem !important; }
            p { font-size: 1.4rem !important; }
        }
        
        /* Top-class animations for headings */
        @keyframes glitch {
            0% {
                transform: translate(0);
                text-shadow: none;
            }
            20% {
                transform: translate(-2px, 2px);
                text-shadow: 2px 0 #ff00ff, -2px 0 #00ffff;
            }
            40% {
                transform: translate(-2px, -2px);
                text-shadow: 2px 0 #00ffff, -2px 0 #ff00ff;
            }
            60% {
                transform: translate(2px, 2px);
                text-shadow: 2px 0 #ff00ff, -2px 0 #00ffff;
            }
            80% {
                transform: translate(2px, -2px);
                text-shadow: 2px 0 #00ffff, -2px 0 #ff00ff;
            }
            100% {
                transform: translate(0);
                text-shadow: none;
            }
        }
        
        @keyframes fadeIn {
            0% {
                opacity: 0;
                transform: scale(0.9);
            }
            100% {
                opacity: 1;
                transform: scale(1);
            }
        }
        
        @keyframes slideIn {
            0% {
                transform: translateX(-100px);
                opacity: 0;
            }
            100% {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes shake {
            0%, 100% {
                transform: translateX(0);
            }
            10%, 30%, 50%, 70%, 90% {
                transform: translateX(-5px);
            }
            20%, 40%, 60%, 80% {
                transform: translateX(5px);
            }
        }
        
        @keyframes pulse {
            0% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(0, 255, 0, 0.7);
            }
            70% {
                transform: scale(1.05);
                box-shadow: 0 0 0 10px rgba(0, 255, 0, 0);
            }
            100% {
                transform: scale(1);
                box-shadow: 0 0 0 0 rgba(0, 255, 0, 0);
            }
        }
        
        /* Matrix rain effect for headings */
        h1::before, h2::before, h3::before {
            content: attr(data-text);
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(180deg, transparent, rgba(0, 255, 0, 0.2), transparent);
            animation: matrix 2s linear infinite;
            clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
            z-index: -1;
        }
        
        @keyframes matrix {
            0% {
                transform: translateY(-100%);
            }
            100% {
                transform: translateY(100%);
            }
        }
        
        /* Cyberpunk glitch effect for headings */
        h1::after, h2::after, h3::after {
            content: attr(data-text);
            position: absolute;
            left: 0;
            top: 0;
            width: 100%;
            height: 100%;
            background: rgba(0, 255, 0, 0.1);
            animation: cyberpunk 3s infinite;
            clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
            z-index: -1;
        }
        
        @keyframes cyberpunk {
            0% {
                clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
            }
            5% {
                clip-path: polygon(0 0, 100% 0, 100% 0, 0 0);
            }
            10% {
                clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
            }
            15% {
                clip-path: polygon(0 100%, 100% 100%, 100% 100%, 0 100%);
            }
            20% {
                clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
            }
            100% {
                clip-path: polygon(0 0, 100% 0, 100% 100%, 0 100%);
            }
        }
        
        /* Neon flicker effect for headings */
        h1, h2, h3 {
            animation: neonFlicker 1.5s infinite alternate !important;
        }
        
        @keyframes neonFlicker {
            0%, 19%, 21%, 23%, 25%, 54%, 56%, 100% {
                text-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00, 0 0 20px #00ff00;
            }
            20%, 24%, 55% {
                text-shadow: none;
            }
        }
        
        /* Retro terminal typing effect for headings */
        h1, h2, h3 {
            overflow: visible !important; /* Changed from hidden to visible */
            border-right: 0.15em solid #00ff00;
            white-space: normal !important; /* Changed from nowrap to normal */
            margin: 0 auto;
            animation: typing 3.5s steps(40, end), blink-caret 0.75s step-end infinite;
        }
        
        @keyframes typing {
            from { width: 0 }
            to { width: 100% }
        }
        
        @keyframes blink-caret {
            from, to { border-color: transparent }
            50% { border-color: #00ff00 }
        }
        
        /* Holographic effect for headings */
        h1, h2, h3 {
            position: relative;
            overflow: visible !important; /* Changed from hidden to visible */
        }
        
        h1::before, h2::before, h3::before {
            content: '';
            position: absolute;
            top: -50%;
            left: -50%;
            width: 200%;
            height: 200%;
            background: linear-gradient(
                45deg,
                transparent 0%,
                rgba(0, 255, 0, 0.1) 50%,
                transparent 100%
            );
            transform: rotate(45deg);
            animation: holographic 3s linear infinite;
        }
        
        @keyframes holographic {
            0% {
                transform: translateX(-100%) rotate(45deg);
            }
            100% {
                transform: translateX(100%) rotate(45deg);
            }
        }
    </style>
    
    <!-- Add particles -->
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
    <div class="particle"></div>
""", unsafe_allow_html=True)

# Main Page
if app_mode == get_text("home", "Home"):
    st.header(get_text("plant_disease_recognition", "PLANT DISEASE RECOGNITION SYSTEM"))
    #image_path = r'C:\Users\91824\Downloads\OIP (2).jpeg'
    #image_path=r'C:\Users\mohsi\OneDrive\Desktop\SIH code\home_image.jpeg'
    st.image("home_page.jpeg", use_container_width=True)
    #st.image(image_path, use_container_width =True)
    st.markdown(get_text("welcome_message", """
    Welcome to the Plant Disease Recognition System! 🌿🔍
    
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
    
    ### How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.
    
    ### Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.
    
    ### Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!
    
    ### About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """))

# About Project
elif app_mode == get_text("about", "About"):
    st.header(get_text("about", "About"))
    st.markdown(get_text("about_dataset", """
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70,295 images)
                2. test (33 images)
                3. validation (17,572 images)
                """))

#weather page
elif app_mode == get_text("weather_monitoring", "Weather Monitoring"):
    st.markdown(f'<h2 class="main-header">🌍 Weather Monitoring</h2>', unsafe_allow_html=True)
    
    # Add a note about location support and accuracy
    st.info(get_text("weather_accuracy_note", "Enter any location in India. For smaller areas, the system will try to find the nearest major city's weather data."))
    
    # Create two columns for input and current weather
    col1, col2 = st.columns([1, 2])
    
    with col1:
        location = st.text_input(get_text("enter_location", "Enter a location to monitor:"), key="weather_location_input")
        if st.button(get_text("get_weather", "Get Weather"), key="get_weather_button"):
            if location:
                with st.spinner(get_text("fetching_weather", "Fetching weather data...")):
                    weather_data = get_weather_data(location)
    
    # Display weather information if available
    if 'weather_data' in locals() and weather_data:
        with col2:
            # Create a nice weather card
            st.markdown(f"""
            <div style="padding: 20px; border-radius: 10px; background-color: rgba(13, 43, 13, 0.7); border: 2px solid #00ff00;">
                <h3 style="color: #00ff00; text-align: center;">{weather_data['name']}</h3>
                <div style="display: flex; justify-content: space-around; margin-top: 20px;">
                    <div style="text-align: center;">
                        <h4 style="color: #00ff00;">{get_text("temperature", "Temperature")}</h4>
                        <p style="font-size: 24px; color: #00ff00;">{weather_data['main']['temp']} °C</p>
                    </div>
                    <div style="text-align: center;">
                        <h4 style="color: #00ff00;">{get_text("humidity", "Humidity")}</h4>
                        <p style="font-size: 24px; color: #00ff00;">{weather_data['main']['humidity']} %</p>
                    </div>
                </div>
                <div style="text-align: center; margin-top: 20px;">
                    <h4 style="color: #00ff00;">{get_text("weather_description", "Weather")}</h4>
                    <p style="font-size: 20px; color: #00ff00;">{weather_data['weather'][0]['description'].capitalize()}</p>
                </div>
            </div>
            """, unsafe_allow_html=True)
            
            # Define thresholds
            temp_threshold = 30  # Example threshold for temperature in °C
            humidity_threshold = 70  # Example threshold for humidity in %
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']

            # Check if temperature or humidity is above the threshold
            if temperature > temp_threshold:
                st.warning(get_text("temp_alert", f"⚠ Alert: The temperature is above the threshold! ({temperature} °C)"))
                
            if humidity > humidity_threshold:
                st.warning(get_text("humidity_alert", f"⚠ Alert: The humidity is above the threshold! ({humidity} %)"))
            
            # Add plant care recommendations based on weather
            st.markdown("### 🌱 Plant Care Recommendations")

            # Temperature-based recommendations
            if temperature > 35:
                st.write(get_text("high_temp_recommendation", "High temperature alert: Water plants more frequently and provide shade during peak hours."))
            elif temperature < 15:
                st.write(get_text("low_temp_recommendation", "Low temperature alert: Protect sensitive plants from frost and reduce watering frequency."))
            else:
                st.write(get_text("normal_temp_recommendation", "Temperature is within the optimal range for most plants."))
            
            # Humidity-based recommendations
            if humidity > 80:
                st.write(get_text("high_humidity_recommendation", "High humidity alert: Ensure good air circulation to prevent fungal diseases."))
            elif humidity < 40:
                st.write(get_text("low_humidity_recommendation", "Low humidity alert: Increase humidity for indoor plants and mist leaves regularly."))
            else:
                st.write(get_text("normal_humidity_recommendation", "Humidity levels are suitable for most plants."))

# Prediction Page
elif app_mode == get_text("disease_recognition", "Disease Recognition"):
    st.header(get_text("disease_recognition", "Disease Recognition"))
    test_image = st.file_uploader(get_text("choose_image", "Choose an Image:"), key="disease_image_uploader")
    
    if st.button(get_text("show_image", "Show Image"), key="show_image_button") and test_image:
        st.image(test_image, use_container_width=True)
    
    # Predict button
    if st.button(get_text("predict", "Predict"), key="predict_button") and test_image:
        st.snow()
        st.write(get_text("analyzing", "Analyzing the image..."))
        result_index = model_prediction(test_image)
        
        # Reading Labels
        class_name = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Apple___healthy',
                      'Blueberry___healthy', 'Cherry_(including_sour)___Powdery_mildew',
                      'Cherry_(including_sour)___healthy', 'Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot',
                      'Corn_(maize)___Common_rust_', 'Corn_(maize)___Northern_Leaf_Blight', 'Corn_(maize)___healthy',
                      'Grape___Black_rot', 'Grape___Esca_(Black_Measles)', 'Grape___Leaf_blight_(Isariopsis_Leaf_Spot)',
                      'Grape___healthy', 'Orange___Haunglongbing_(Citrus_greening)', 'Peach___Bacterial_spot',
                      'Peach___healthy', 'Pepper,_bell___Bacterial_spot', 'Pepper,_bell___healthy',
                      'Potato___Early_blight', 'Potato___Late_blight', 'Potato___healthy',
                      'Raspberry___healthy', 'Soybean___healthy', 'Squash___Powdery_mildew',
                      'Strawberry___Leaf_scorch', 'Strawberry___healthy', 'Tomato___Bacterial_spot',
                      'Tomato___Early_blight', 'Tomato___Late_blight', 'Tomato___Leaf_Mold',
                      'Tomato___Septoria_leaf_spot', 'Tomato___Spider_mites Two-spotted_spider_mite',
                      'Tomato___Target_Spot', 'Tomato___Tomato_Yellow_Leaf_Curl_Virus', 'Tomato___Tomato_mosaic_virus',
                      'Tomato___healthy']
        
        predicted_disease = class_name[result_index]
        st.success(get_text("model_prediction", f"Model Prediction: {predicted_disease}"))
        
        # Get cure suggestion using Gemini's API
        treatment_suggestion = get_treatment_suggestion(predicted_disease)
        st.write(get_text("suggested_farms", f"Suggested nearby Agro Farms for {predicted_disease}: {treatment_suggestion}"))

        treatment_suggestion1 = get_treatment_suggestion1(predicted_disease)
        st.write(get_text("suggested_treatment", f"Suggested Treatment for {predicted_disease}: {treatment_suggestion1}"))

# Nearby Shops Page
elif app_mode == get_text("nearby_shops", "Nearby Shops"):
    st.header(get_text("nearby_shops", "Find Nearby Shops"))
    
    # Dropdown to select disease
    disease_options = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                       'Peach___Bacterial_spot', 'Peach___healthy']
    selected_disease = st.selectbox(get_text("select_disease", "Select Disease"), disease_options, key="disease_selector")
    
    # Input for location
    location = st.text_input(get_text("enter_location", "Enter Your Location"), key="location_input")
    
    # Button to fetch nearby shops
    if st.button(get_text("find_shops", "Find Shops"), key="find_shops_button"):
        if location:
            st.write(get_text("fetching_shops", f"Fetching nearby shops for {selected_disease} in {location}..."))
            nearby_shops = get_nearby_shops(selected_disease, location)
            st.write(get_text("nearby_shops_result", f"Nearby Shops for {selected_disease}:"))
            st.write(nearby_shops)
        else:
            st.write(get_text("enter_location_prompt", "Please enter a location to find nearby shops."))
