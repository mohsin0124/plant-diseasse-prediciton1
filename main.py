import streamlit as st
from googletrans import Translator
import tensorflow as tf
import numpy as np
from PIL import Image
import os

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

# Translate the main navigation options
app_mode = st.sidebar.selectbox(
    translate_text("Select Page", selected_lang_code),
    [nav_home, nav_about, nav_disease_recognition]
)

# Custom CSS for theme adjustments
st.markdown("""
<style>
        /* Global styles */
body {
  font-family: 'Open Sans', sans-serif;
  margin: 0;
  padding: 0;
}

/* Main container */
.main {
  max-width: 1200px;
  margin: 0 auto;
  padding: 20px;
}

/* Header */
header {
  text-align: center;
  padding: 20px 0;
}

.main-header {
  font-size: 40px;
  font-weight: bold;
  color: #388E3C; /* Earthy green color */
}

.main-subheader {
  font-size: 28px;
  font-weight: 500;
  color: #2E7D32; /* Darker green for subheadings */
}

/* Sidebar */
aside {
  background-color: #E0F2EF; /* Lighter green for sidebar */
  padding: 20px;
  min-height: 100vh;
}

.sidebar-title {
  font-size: 24px;
  font-weight: bold;
  margin-bottom: 20px;
}

.sidebar-nav {
  list-style: none;
  padding: 0;
}

.sidebar-nav li {
  margin-bottom: 10px;
}

.sidebar-nav a {
  text-decoration: none;
  color: #333;
  padding: 10px;
  display: block;
  border-radius: 5px;
}

.sidebar-nav a:hover {
  background-color: #388E3C;
  color: white;
}

/* Content area */
main {
  padding: 20px;
}

/* Image */
.image-container {
  text-align: center;
  margin-bottom: 20px;
}

.image-container img {
  max-width: 100%;
  height: auto;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

/* Prediction result */
.prediction-result {
  background-color: #F5F9F7;
  padding: 20px;
  border-radius: 5px;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
}

.prediction-result h3 {
  margin-bottom: 10px;
}

.prediction-result p {
  margin-bottom: 0;
}

/* Responsive layout */
@media (max-width: 768px) {
  aside {
    display: none;
  }

  main {
    margin: 0;
  }
}
 </style>
    """, unsafe_allow_html=True)

# Main Page
if app_mode == nav_home:
    st.markdown(f'<h2 class="main-header">{translate_text("🌿 PLANT DISEASE RECOGNITION SYSTEM", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.image("home_page.jpeg", use_column_width=True)
    st.markdown('<p class="main-header">Welcome to the Plant Disease Recognition System!</p>', unsafe_allow_html=True)
    st.markdown(translate_text("""
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
    """, selected_lang_code), unsafe_allow_html=True)

# About Project
elif app_mode == nav_about:
    st.markdown(f'<h2 class="main-header">{translate_text("About", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.markdown(translate_text("""
                #### About Dataset
                This dataset is recreated using offline augmentation from the original dataset. The original dataset can be found on this GitHub repo.
                This dataset consists of about 87K RGB images of healthy and diseased crop leaves, categorized into 38 different classes. The total dataset is divided into an 80/20 ratio of training and validation sets, preserving the directory structure.
                A new directory containing 33 test images is created later for prediction purposes.
                #### 📊 Content
                1. **Train:** 70,295 images
                2. **Test:** 33 images
                3. **Validation:** 17,572 images
                """, selected_lang_code))

# Prediction Page
elif app_mode == nav_disease_recognition:
    st.markdown(f'<h2 class="main-header">{translate_text("🌱 Disease Recognition", selected_lang_code)}</h2>', unsafe_allow_html=True)
    st.write(translate_text("Upload an image of a plant, and the model will predict the disease.", selected_lang_code))
    
    test_image = st.file_uploader(translate_text("Choose an Image:", selected_lang_code), type=['png', 'jpg', 'jpeg'])
    
    if test_image is not None:
        st.image(test_image, use_column_width=True)
    
        if st.button(translate_text("Predict", selected_lang_code)):
            if model is None:
                st.error(translate_text("Model could not be loaded. Please check the model path.", selected_lang_code))
            else:
                with st.spinner(translate_text('Analyzing the image...', selected_lang_code)):
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
                    st.success(translate_text(f"🌟 Model Prediction: It's a {class_name[result_index]}", selected_lang_code))
