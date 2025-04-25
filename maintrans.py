import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os
import requests
from deep_translator import GoogleTranslator

import certifi
os.environ["SSL_CERT_FILE"] = certifi.where()


import json
from groq import Groq  # Make sure Groq is installed: pip install groq
from PIL import Image

# Initialize Groq client with your API key
client = Groq(
    api_key="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",  # Your provided API key
)

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
      


# Function to get treatment suggestion from Groq API
def get_treatment_suggestion(disease_name):
    # Messages to send to Groq API (LLaMA model or Mixtral model)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        
        {"role": "user", "content": f"recomend 5 retail Agro Farms nearby vasavi college of engineering ,ibrahimbagh ,hydereabad .add the distant how far is it. only give shop names and distant in km, "},
                                       
        ]
    
    try:
        # Chat completion using Groq's Mixtral model
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",  # Adjust as per Groq's available models
        )
        return chat_completion.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error: Unable to connect to Groq API. Exception: {str(e)}"


# Function to get treatment suggestion from Groq API
def get_treatment_suggestion1(disease_name):
    # Messages to send to Groq API (LLaMA model or Mixtral model)
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        #{"role": "user", "content": f"recomend 5 medicines to cure this {disease_name} disease. give me in point"},
        {"role": "user", "content": f"recomend 3 medicines to cure this {disease_name} disease and give cost of it. give me in boldpoint only only point"}
    ]
    
    try:
        # Chat completion using Groq's Mixtral model
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",  # Adjust as per Groq's available models
        )
        return chat_completion.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error: Unable to connect to Groq API. Exception: {str(e)}"


# Function to get nearby shops using LLaMA AI
def get_nearby_shops(disease, location):
    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": f"Find nearby shops that fertilizers for the disease: {disease}, Location: is {location}."}
    ]
    
    try:
        chat_completion = client.chat.completions.create(
            messages=messages,
            model="llama-3.3-70b-versatile",  # Adjust as per Groq's available models
        )
        return chat_completion.choices[0].message.content.strip()
    
    except Exception as e:
        return f"Error: Unable to connect to Groq API. Exception: {str(e)}"




# Sidebar
st.sidebar.title("Dashboard")
st.sidebar.markdown("Navigate through the pages to explore the app!")
#selected_language = st.sidebar.selectbox("Select Language", list(languages.values())) = list(languages.keys())[list(languages.values()).index(selected_language)]



# Translate navigation options
nav_home = ("Home")
nav_about = ("About")
nav_disease_recognition = ("Disease Recognition")
nav_field= ("Detection")
nav_nearby_shops = ("Nearby Shops")
nav_weather_monitor =("Weather Monitoring")
nav_diseases=("Diseases")


# Translate the main navigation options
app_mode = st.sidebar.selectbox(
    ("Select Page"),
    [nav_home, nav_about, nav_disease_recognition,nav_field,nav_nearby_shops,nav_weather_monitor,nav_diseases]
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
   
}

/* Container for centering and padding */
.container {
  display: flex;
  justify-content: center;
  align-items: center;
  flex-direction: column;
  height: 100vh;
  background-color: #f0f4f7;
  padding: 20px;
}

/* Field title with improved styling */
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
  background: linear-gradient(90deg, #ff6e7f, #bfe9ff);  /* Gradient background */
  border-radius: 10px;
  box-shadow: 0px 4px 15px rgba(0, 0, 0, 0.1);  /* Soft shadow */
  transition: transform 0.3s ease; /* Smooth hover effect */
}

/* Field title hover effect */
.field_title:hover {
  transform: scale(1.05);
}

/* Map container with grid lines and enhanced styling */
.map {
  background-color: rgb(230, 255, 240);  /* Light, soft green background */
  border: 2px solid #2c3e50;  /* Darker border for contrast */
  border-radius: 15px;  /* Rounded corners */
  padding-left: 30px;
  margin-left: 50px;
    background-image: url("/mnt/data/grass_background.jpeg");  /* Path to the uploaded image */
  background-size: cover;
  background-position: center;

  
  /* Responsive width and height */
  width: 50vw;
  height: 50vw;
  max-width: 600px;
  max-height: 500px;
  min-width: 150px;
  min-height: 150px;

  position: relative;
  
  /* Grid lines */
  background-image: 
    repeating-linear-gradient(#2c3e50, #2c3e50 1px, transparent 1px, transparent 20px),
    repeating-linear-gradient(90deg, #2c3e50, #2c3e50 1px, transparent 1px, transparent 20px);
  
  /* Box shadow for depth */
  box-shadow: 0px 8px 20px rgba(0, 0, 0, 0.15);
  
  /* Smooth transition */
  transition: box-shadow 0.3s ease-in-out, transform 0.3s ease-in-out;
}

/* Hover effect for the map */
.map:hover {
  box-shadow: 0px 12px 30px rgba(0, 0, 0, 0.2);
  transform: scale(1.02);
}

/* Blinking alert div */
.blinking-alert {
  width: 30px;
  height: 30px;
  background-color: red;
  position: absolute;
  top: 10px;
  right: 10px;
  border-radius: 50%;
  
  /* Blinking animation */
  animation: blink 1s infinite;
  
  /* Box shadow for emphasis */
  box-shadow: 0px 4px 10px rgba(255, 0, 0, 0.5);
}

/* Keyframes for blinking alert */
@keyframes blink {
  0% { opacity: 1; }
  50% { opacity: 0; }
  100% { opacity: 1; }
}

/* Alert icon styling */
.alert_icon {
  margin-right: 10px;
  font-size: 36px;
}


.mainp{
    display: flex;
    height: 100vh;
    padding: 0.5rem;
}
.sidebarp{
    background-color: grey;
    width: 340px;
    border-radius: 1rem;  /* 1rem=16px */
    margin-right: 0.5rem;
}
.main-content{
    background-color:#121212;
    flex: 1;
    border-radius: 1rem;  
    overflow: auto;
    padding: 0 1.5rem 0 1.5rem;
}

/*.navp{
    background-color: #121212;
    border-radius: 1rem;
    display: flex;
    flex-direction: column;
    justify-content:center;
    height: 100px;
    padding: 0.5rem 0.75rem;
}*/

/* Keyframes for blinking effect */
@keyframes blink {
  0% { opacity: 1; }
  50% { opacity: 0; }
  100% { opacity: 1; }
}



    </style>
    """, unsafe_allow_html=True)

# Main Page
if app_mode == nav_home:
    st.markdown(f'<h2 class="main-header">{("🌿 PLANT DISEASE RECOGNITION SYSTEM")}</h2>', unsafe_allow_html=True)
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown(f'<p class="main-subheader">{("Welcome to the Plant Disease Recognition System!")}</p>', unsafe_allow_html=True)

    st.markdown(f'<div class= "div">{("""
    Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!

    ### 🚀 How It Works
    1. **Upload Image:** Go to the **Disease Recognition** page and upload an image of a plant with suspected diseases.
    2. **Analysis:** Our system will process the image using advanced algorithms to identify potential diseases.
    3. **Results:** View the results and recommendations for further action.

    ### 🛠️ Why Choose Us?
    - **Accuracy:** Our system utilizes state-of-the-art machine learning techniques for accurate disease detection.
    - **User-Friendly:** Simple and intuitive interface for seamless user experience.
    - **Fast and Efficient:** Receive results in seconds, allowing for quick decision-making.

    ### 🔍 Get Started
    Click on the **Disease Recognition** page in the sidebar to upload an image and experience the power of our Plant Disease Recognition System!

    ### 👥 About Us
    Learn more about the project, our team, and our goals on the **About** page.
    """)}</div>', unsafe_allow_html=True)

#plant diseases page
#elif(app_mode == nav_diseases):
elif(app_mode == nav_diseases):
    st.markdown(f'<div class="mainp">' + 
                f'<div class="sidebarp">' + 
                f'<div class="navp"></div>' + 
                f'</div>' + 
                f'</div>', unsafe_allow_html=True)
    
    

#near by shops
elif app_mode == nav_nearby_shops:
    st.markdown(f'<h2 class="main-header">🛒 Nearby Shops for Plant Care</h2>', unsafe_allow_html=True)
    
        # Dropdown to select disease
    disease_options = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 'Peach___Bacterial_spot', 'Peach___healthy']
    selected_disease = st.selectbox("Select Disease", disease_options)
    
    # Input for location
    location = st.text_input("Enter Your Location (text)")
    
    # Button to fetch nearby shops
    if st.button("Find Shops"):
        if location:
            st.write(f"Fetching nearby shops for {selected_disease} in {location}...")
            nearby_shops = get_nearby_shops(selected_disease, location)
            st.write(f"Nearby Shops for {selected_disease}:")
            st.write(nearby_shops)
        else:
            st.write("Please enter a location to find nearby shops.")
    

# About Project
elif app_mode == nav_about:
    st.markdown(f'<h2 class="main-header">{("About")}</h2>', unsafe_allow_html=True)
    st.markdown(("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### 📊 Content
                1. **Train:** 70,295 images
                2. **Test:** 33 images
                3. **Validation:** 17,572 images
                """))
#weather page
elif app_mode == nav_weather_monitor:
    st.markdown(f'<h2 class="main-header">🌍 Weather Monitoring</h2>', unsafe_allow_html=True)
    location = st.text_input("Enter a location to monitor:")
    
    if location and st.button("Get Weather"):
        weather_data = get_weather_data(location)
        
        if weather_data:
            st.write(f"*Location:* {weather_data['name']}")
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            
            st.write(f"*Temperature:* {temperature} °C")
            st.write(f"*Humidity:* {humidity} %")
            st.write(f"*Weather Description:* {weather_data['weather'][0]['description'].capitalize()}")
            
            # Define thresholds
            temp_threshold = 30  # Example threshold for temperature in °C
            humidity_threshold = 70  # Example threshold for humidity in %

            # Check if temperature or humidity is above the threshold
            if temperature > temp_threshold:
                st.warning(f"⚠ Alert: The temperature is above the threshold! ({temperature} °C)")
                
            if humidity > humidity_threshold:
                st.warning(f"⚠ Alert: The humidity is above the threshold! ({humidity} %)")

        else:
            st.error("Could not fetch weather data. Please check the location name.")
            
#area alert detection page

elif app_mode==nav_field:
  st.markdown(f'''
    <div class="container">
        <h2 class="field_title">
            <span class="alert_icon" style="color: red">&#10071;</span>
            {("Alert - Disease detected")}
        </h2>
        <div class="map">
            <div class="blinking-alert"></div>
        </div>
    </div>
''', unsafe_allow_html=True)

    


# Prediction Page
elif app_mode == nav_disease_recognition:
    st.markdown(f'<h2 class="main-header">{("🌱 Disease Recognition")}</h2>', unsafe_allow_html=True)
    st.write(f'<p class= "pc">{("Upload an image of a plant, and the model will predict the disease.")}</p>', unsafe_allow_html=True)
    
    test_image = st.file_uploader(("Choose an Image:"), type=['png', 'jpg', 'jpeg'])
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)
    
        if st.button(("Predict")):
            if model is None:
                st.error(("Model could not be loaded. Please check the model path."))
            else:
                with st.spinner(('Analyzing the image...')):
                    result_index = model_prediction(test_image)
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
                    st.success(f"Model Prediction: {predicted_disease}")
        
                    # Get cure suggestion using Groq's API
                    treatment_suggestion = get_treatment_suggestion(predicted_disease)
                    st.write(f"Suggested nearby Agro Farms  {predicted_disease}: {treatment_suggestion}")

                    treatment_suggestion1 = get_treatment_suggestion1(predicted_disease)
                    st.write(f"Suggested Treatment for {predicted_disease}: {treatment_suggestion1}")        
                    st.success((f"🌟 Model Prediction: It's a {class_name[result_index]}"))
