import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image
import os

# Load the model once using caching
@st.cache_resource
def load_model():
    model_path = "trained_plant_disease_model.keras"
    # Check if file exists
    if not os.path.exists(model_path):
        #st.error(f"Model file not found at {model_path}. Please check the path.")
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

# Sidebar
st.sidebar.title("Dashboard")
st.sidebar.markdown("Navigate through the pages to explore the app!")
app_mode = st.sidebar.selectbox("Select Page", ["Home", "About", "Disease Recognition"])

# Main Page
if app_mode == "Home":
    st.header("🌿 PLANT DISEASE RECOGNITION SYSTEM")
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown("""
    <style>
            /* Main styling for the app */
        .main {
            background-color: #f8f9f0;
            font-family: 'Arial', sans-serif;
        }
        /* Header styling */
        .main-header {
            font-weight: bold;
            color: #2E7D32;  /* Earthy green color */
            text-align: center;
        }
        /* Sidebar styling */
        .sidebar .sidebar-content {
            background-color: #A5D6A7;  /* Light green */
        }
        /* Button styling */
        .stButton > button {
            background-color: #66BB6A;
            color: white;
            border: none;
            border-radius: 5px;
            padding: 8px 16px;
        }
        .stButton > button:hover {
            background-color: #388E3C;
        }
    </style>
    """, unsafe_allow_html=True)
    st.markdown('<p class="main-header">Welcome to the Plant Disease Recognition System!</p>', unsafe_allow_html=True)
    st.markdown("""
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
    """, unsafe_allow_html=True)

# About Project
elif app_mode == "About":
    st.header("About")
    st.markdown("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### 📊 Content
                1. **Train:** 70,295 images
                2. **Test:** 33 images
                3. **Validation:** 17,572 images
                """)

# Prediction Page
elif app_mode == "Disease Recognition":
    st.header("🌱 Disease Recognition")
    st.write("Upload an image of a plant, and the model will predict the disease.")
    
    test_image = st.file_uploader("Choose an Image:", type=['png', 'jpg', 'jpeg'])
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)
    
        if st.button("Predict"):
            #if model is None:
             #   st.error("Model could not be loaded. Please check the model path.")
            #else:
                with st.spinner('Analyzing the image...'):
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
                    st.success(f"🌟 Model Prediction: It's a {class_name[result_index]}")
