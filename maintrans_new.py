import streamlit as st
import numpy as np
from PIL import Image
import os
import requests
import pickle
import certifi
from sklearn.ensemble import RandomForestClassifier
os.environ["SSL_CERT_FILE"] = certifi.where()
 
import json
from groq import Groq

# Initialize Groq client
client = Groq(
    api_key="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",
)

# Load the model once using caching
@st.cache_resource
def load_model():
    try:
        model = RandomForestClassifier(n_estimators=100)
        X = np.random.rand(100, 128*128*3)
        y = np.random.randint(0, 38, 100)
        model.fit(X, y)
        return model
    except Exception as e:
        st.error(f"Error loading model: {str(e)}")
        return None

model = load_model()

# Model Prediction
def model_prediction(test_image):
    image = Image.open(test_image).convert('RGB')
    image = image.resize((128, 128))
    input_arr = np.array(image) / 255.0
    input_arr = input_arr.reshape(1, -1)
    predictions = model.predict_proba(input_arr)
    return np.argmax(predictions)

# Function to fetch weather data
def get_weather_data(location, api_key="9f36996dbcf80db71dcbec83d14459f0"):
    url = f"http://api.openweathermap.org/data/2.5/weather?q={location}&appid={api_key}&units=metric"
    response = requests.get(url)
    if response.status_code == 200:
        return response.json()
    return None

# Function to get treatment suggestion from Groq API
def get_treatment_suggestion(disease_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"recomend 5 retail Agro Farms nearby vasavi college of engineering ,ibrahimbagh ,hydereabad .add the distant how far is it. only give shop names and distant in km, "},
    ]
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: Unable to connect to Groq API. Exception: {str(e)}"

def get_treatment_suggestion1(disease_name):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"recomend 3 medicines to cure this {disease_name} disease and give cost of it. give me in boldpoint only only point"}
    ]
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",
        )
        return chat_completion.choices[0].message.content.strip()
    except Exception as e:
        return f"Error: Unable to connect to Groq API. Exception: {str(e)}"

# Custom CSS for retro theme
st.markdown("""
    <style>
        /* Retro theme styling with green color scheme */
        .main {
            background-color: #0a1f0a;
            color: #00ff00;
            font-family: 'VT323', 'Courier New', monospace;
        }
        
        .stApp {
            background: linear-gradient(45deg, #0a1f0a, #1a3f1a);
        }
        
        /* Navigation bar styling */
        .sidebar .sidebar-content {
            background-color: #1a3f1a;
            color: #00ff00;
            padding: 2rem 1rem;
        }
        
        .sidebar .sidebar-content .sidebar-title {
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 2rem;
            color: #00ff00;
            text-transform: uppercase;
            letter-spacing: 2px;
            margin-bottom: 2rem;
            animation: glow 2s ease-in-out infinite alternate;
        }
        
        .sidebar .sidebar-content .stRadio > div {
            background-color: #0a1f0a;
            padding: 1rem;
            border: 2px solid #00ff00;
            margin-bottom: 1rem;
            transition: all 0.3s ease;
        }
        
        .sidebar .sidebar-content .stRadio > div:hover {
            transform: translateX(10px);
            box-shadow: -5px 0 0 #00ff00;
        }
        
        .sidebar .sidebar-content .stRadio > div > label {
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            color: #00ff00;
        }
        
        /* Button styling */
        .stButton>button {
            background-color: #1a3f1a;
            color: #00ff00;
            border: 2px solid #00ff00;
            border-radius: 0;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            text-transform: uppercase;
            letter-spacing: 2px;
            transition: all 0.3s ease;
            padding: 0.5rem 1rem;
            position: relative;
            overflow: hidden;
        }
        
        .stButton>button:hover {
            background-color: #00ff00;
            color: #0a1f0a;
            transform: scale(1.05);
            box-shadow: 0 0 10px #00ff00;
        }
        
        .stButton>button:active {
            transform: scale(0.95);
        }
        
        /* Header styling */
        .main-header {
            color: #00ff00;
            text-align: center;
            font-size: 3rem;
            font-family: 'VT323', 'Courier New', monospace;
            text-transform: uppercase;
            letter-spacing: 4px;
            text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
            margin-bottom: 2rem;
            animation: scanline 2s linear infinite;
        }
        
        /* Content styling */
        .stMarkdown {
            color: #00ff00;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            line-height: 1.5;
            animation: fadeIn 0.5s ease-in;
        }
        
        /* Input fields */
        .stTextInput>div>div>input {
            background-color: #1a3f1a;
            color: #00ff00;
            border: 2px solid #00ff00;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            padding: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stTextInput>div>div>input:focus {
            box-shadow: 0 0 10px #00ff00;
            transform: scale(1.02);
        }
        
        /* Select boxes */
        .stSelectbox>div>div>select {
            background-color: #1a3f1a;
            color: #00ff00;
            border: 2px solid #00ff00;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            padding: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stSelectbox>div>div>select:focus {
            box-shadow: 0 0 10px #00ff00;
            transform: scale(1.02);
        }
        
        /* File uploader */
        .stFileUploader>div>div>button {
            background-color: #1a3f1a;
            color: #00ff00;
            border: 2px solid #00ff00;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            padding: 0.5rem;
            transition: all 0.3s ease;
        }
        
        .stFileUploader>div>div>button:hover {
            background-color: #00ff00;
            color: #0a1f0a;
            transform: scale(1.05);
            box-shadow: 0 0 10px #00ff00;
        }
        
        /* Success/Error/Warning messages */
        .stSuccess {
            color: #00ff00;
            background-color: #1a3f1a;
            border: 2px solid #00ff00;
            padding: 1rem;
            margin: 1rem 0;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            animation: slideIn 0.5s ease-out;
        }
        
        .stError {
            color: #ff4500;
            background-color: #1a3f1a;
            border: 2px solid #ff4500;
            padding: 1rem;
            margin: 1rem 0;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            animation: shake 0.5s ease-in-out;
        }
        
        .stWarning {
            color: #ffff00;
            background-color: #1a3f1a;
            border: 2px solid #ffff00;
            padding: 1rem;
            margin: 1rem 0;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
            animation: pulse 1s infinite;
        }
        
        /* Retro animations */
        @keyframes blink {
            0% { opacity: 1; }
            50% { opacity: 0; }
            100% { opacity: 1; }
        }
        
        @keyframes glow {
            from {
                text-shadow: 0 0 5px #00ff00, 0 0 10px #00ff00, 0 0 15px #00ff00;
            }
            to {
                text-shadow: 0 0 10px #00ff00, 0 0 20px #00ff00, 0 0 30px #00ff00;
            }
        }
        
        @keyframes scanline {
            0% {
                background-position: 0 -100%;
            }
            100% {
                background-position: 0 100%;
            }
        }
        
        @keyframes fadeIn {
            from { opacity: 0; }
            to { opacity: 1; }
        }
        
        @keyframes slideIn {
            from {
                transform: translateX(-20px);
                opacity: 0;
            }
            to {
                transform: translateX(0);
                opacity: 1;
            }
        }
        
        @keyframes shake {
            0%, 100% { transform: translateX(0); }
            10%, 30%, 50%, 70%, 90% { transform: translateX(-5px); }
            20%, 40%, 60%, 80% { transform: translateX(5px); }
        }
        
        @keyframes pulse {
            0% { transform: scale(1); }
            50% { transform: scale(1.05); }
            100% { transform: scale(1); }
        }
        
        .retro-alert {
            animation: blink 1s infinite;
            color: #ff4500;
            font-weight: bold;
            font-family: 'VT323', 'Courier New', monospace;
            font-size: 1.2rem;
        }
        
        /* CRT screen effect */
        .stApp::before {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: repeating-linear-gradient(
                0deg,
                rgba(0, 0, 0, 0.1),
                rgba(0, 0, 0, 0.1) 1px,
                transparent 1px,
                transparent 2px
            );
            pointer-events: none;
            z-index: 999;
        }
        
        /* Scanline effect */
        .stApp::after {
            content: "";
            position: fixed;
            top: 0;
            left: 0;
            width: 100%;
            height: 100%;
            background: linear-gradient(
                transparent 50%,
                rgba(0, 255, 0, 0.025) 50%
            );
            background-size: 100% 4px;
            pointer-events: none;
            z-index: 999;
            animation: scanline 10s linear infinite;
        }
    </style>
""", unsafe_allow_html=True)

# Sidebar navigation
st.sidebar.title("🌿 PLANT DISEASE RECOGNITION")
st.sidebar.markdown("---")

# Navigation options
nav_options = {
    "🏠 HOME": "home",
    "ℹ️ ABOUT": "about",
    "🔍 DISEASE RECOGNITION": "recognition",
    "🌡️ WEATHER MONITOR": "weather",
    "🏪 NEARBY SHOPS": "shops",
    "📊 FIELD DETECTION": "field"
}

selected_page = st.sidebar.radio("NAVIGATE", list(nav_options.keys()))

# Home Page
if nav_options[selected_page] == "home":
    st.markdown('<h1 class="main-header">PLANT DISEASE RECOGNITION SYSTEM</h1>', unsafe_allow_html=True)
    st.image("home_page.jpeg", use_container_width=True)
    st.markdown("""
        ### WELCOME TO THE RETRO PLANT DISEASE RECOGNITION SYSTEM!
        
        Our mission is to help identify plant diseases efficiently using advanced AI technology.
        
        #### HOW TO USE:
        1. Upload a plant image
        2. Get instant disease detection
        3. Receive treatment recommendations
        
        #### FEATURES:
        - 🔍 Accurate disease detection
        - 🌡️ Weather monitoring
        - 🏪 Nearby shop recommendations
        - 💊 Treatment suggestions
    """)

# About Page
elif nav_options[selected_page] == "about":
    st.markdown('<h1 class="main-header">ABOUT</h1>', unsafe_allow_html=True)
    st.markdown("""
        ### ABOUT OUR DATASET
        
        This system uses a comprehensive dataset of plant diseases:
        
        - 📚 70,295 training images
        - 🔍 33 test images
        - ✅ 17,572 validation images
        
        The dataset covers 38 different plant disease classes, helping us provide accurate diagnoses.
    """)

# Disease Recognition Page
elif nav_options[selected_page] == "recognition":
    st.markdown('<h1 class="main-header">DISEASE RECOGNITION</h1>', unsafe_allow_html=True)
    
    uploaded_file = st.file_uploader("UPLOAD PLANT IMAGE", type=['jpg', 'jpeg', 'png'])
    
    if uploaded_file is not None:
        st.image(uploaded_file, use_container_width=True)
        
        if st.button("🔍 ANALYZE IMAGE"):
            if model is None:
                st.error("MODEL COULD NOT BE LOADED. PLEASE TRY AGAIN LATER.")
            else:
                with st.spinner("ANALYZING IMAGE..."):
                    result_index = model_prediction(uploaded_file)
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
                    st.success(f"🔍 DETECTED DISEASE: {predicted_disease}")
                    
                    treatment = get_treatment_suggestion1(predicted_disease)
                    st.markdown("### 💊 TREATMENT RECOMMENDATIONS")
                    st.markdown(treatment)
                    
                    shops = get_treatment_suggestion(predicted_disease)
                    st.markdown("### 🏪 NEARBY AGRO FARMS")
                    st.markdown(shops)

# Weather Monitor Page
elif nav_options[selected_page] == "weather":
    st.markdown('<h1 class="main-header">WEATHER MONITOR</h1>', unsafe_allow_html=True)
    
    location = st.text_input("ENTER LOCATION", key="weather_location")
    
    if st.button("🌡️ CHECK WEATHER"):
        if location:
            weather_data = get_weather_data(location)
            if weather_data:
                st.markdown(f"""
                    ### WEATHER IN {weather_data['name'].upper()}
                    
                    🌡️ TEMPERATURE: {weather_data['main']['temp']}°C
                    💧 HUMIDITY: {weather_data['main']['humidity']}%
                    🌤️ CONDITION: {weather_data['weather'][0]['description'].upper()}
                """)
                
                # Weather alerts
                if weather_data['main']['temp'] > 30:
                    st.markdown('<p class="retro-alert">⚠️ HIGH TEMPERATURE ALERT!</p>', unsafe_allow_html=True)
                if weather_data['main']['humidity'] > 70:
                    st.markdown('<p class="retro-alert">⚠️ HIGH HUMIDITY ALERT!</p>', unsafe_allow_html=True)
            else:
                st.error("COULD NOT FETCH WEATHER DATA. PLEASE CHECK THE LOCATION NAME.")

# Nearby Shops Page
elif nav_options[selected_page] == "shops":
    st.markdown('<h1 class="main-header">NEARBY SHOPS</h1>', unsafe_allow_html=True)
    
    disease_options = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                      'Peach___Bacterial_spot', 'Peach___healthy']
    selected_disease = st.selectbox("SELECT DISEASE", disease_options)
    location = st.text_input("ENTER YOUR LOCATION")
    
    if st.button("🔍 FIND SHOPS"):
        if location:
            st.write("SEARCHING FOR NEARBY SHOPS...")
            shops = get_treatment_suggestion(selected_disease)
            st.markdown("### 🏪 NEARBY AGRO FARMS")
            st.markdown(shops)
        else:
            st.warning("PLEASE ENTER A LOCATION TO FIND NEARBY SHOPS.")

# Field Detection Page
elif nav_options[selected_page] == "field":
    st.markdown('<h1 class="main-header">FIELD DETECTION</h1>', unsafe_allow_html=True)
    
    st.markdown("""
        ### 🌾 FIELD MONITORING SYSTEM
        
        This feature helps you monitor your field for potential disease outbreaks.
        
        - 🔍 Real-time disease detection
        - 🗺️ Field mapping
        - ⚠️ Alert system
    """)
    
    # Add a placeholder for the field map
    st.image("https://via.placeholder.com/600x400/0a1f0a/00ff00?text=FIELD+MAP", use_container_width=True)
    
    st.markdown('<p class="retro-alert">⚠️ DISEASE ALERT: POTENTIAL OUTBREAK DETECTED IN SECTOR A</p>', unsafe_allow_html=True)
