import streamlit as st
import tensorflow as tf
import numpy as np
import requests
import json
from groq import Groq  # Make sure Groq is installed: pip install groq
from PIL import Image
import os
import requests
# Initialize Groq client with your API key
client = Groq(
    api_key="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",  # Your provided API key
)

# Tensorflow Model Prediction
def model_prediction(test_image):
    model = tf.keras.models.load_model("trained_plant_disease_model.keras")
    image = tf.keras.preprocessing.image.load_img(test_image, target_size=(128, 128))
    input_arr = tf.keras.preprocessing.image.img_to_array(image)
    input_arr = np.array([input_arr])  # Convert single image to batch
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
            model="llama-3.3-70b-versatile",  # Adjust as per Groq's available models   mixtral-8x7b-32768
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

# Streamlit App Structure
# Sidebar
st.sidebar.title("Dashboard")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition", "Nearby Shops","Weather Monitoring"])

# Language selection
lang_options = ["English", "Hindi", "Telugu"]
default_lang = "English"
selected_lang = st.sidebar.selectbox("Choose Language", lang_options, index=lang_options.index(default_lang))


# Main Page
if app_mode == "Home":
    st.header("PLANT DISEASE RECOGNITION SYSTEM")
    #image_path = r'C:\Users\91824\Downloads\OIP (2).jpeg'
    #image_path=r'C:\Users\mohsi\OneDrive\Desktop\SIH code\home_image.jpeg'
    st.image("home_page.jpeg", use_container_width =True)
    #st.image(image_path, use_container_width =True)
    st.markdown("""
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
    """)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this github repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves which is categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation set preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### Content
                1. train (70,295 images)
                2. test (33 images)
                3. validation (17,572 images)
                """)


#weather page
elif app_mode == "Weather Monitoring":
    st.markdown(f'<h2 class="main-header">🌍 Weather Monitoring</h2>', unsafe_allow_html=True)
    location = st.text_input("Enter a location to monitor:")
    
    if location and st.button("Get Weather"):
        weather_data = get_weather_data(location)
        
        if weather_data:
            st.write(f"Location: {weather_data['name']}")
            temperature = weather_data['main']['temp']
            humidity = weather_data['main']['humidity']
            
            st.write(f"Temperature: {temperature} °C")
            st.write(f"Humidity: {humidity} %")
            st.write(f"Weather Description: {weather_data['weather'][0]['description'].capitalize()}")
            
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

            
# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("Disease Recognition")
    test_image = st.file_uploader("Choose an Image:")
    
    if st.button("Show Image") and test_image:
        st.image(test_image, use_column_width=True)
    
    # Predict button
    if st.button("Predict") and test_image:
        st.snow()
        st.write("Analyzing the image...")
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
        st.success(f"Model Prediction: {predicted_disease}")
        
        # Get cure suggestion using Groq's API
        treatment_suggestion = get_treatment_suggestion(predicted_disease)
        st.write(f"Suggested nearby Agro Farms  {predicted_disease}: {treatment_suggestion}")

        treatment_suggestion1 = get_treatment_suggestion1(predicted_disease)
        st.write(f"Suggested Treatment for {predicted_disease}: {treatment_suggestion1}")

# Nearby Shops Page
elif app_mode == "Nearby Shops":
    st.header("Find Nearby Shops for {disease_options} treatment in {location}")
    
    # Dropdown to select disease
    disease_options = ['Apple___Apple_scab', 'Apple___Black_rot', 'Apple___Cedar_apple_rust', 
                       'Peach___Bacterial_spot', 'Peach___healthy']
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
