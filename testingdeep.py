import streamlit as st
from deep_translator import GoogleTranslator

# Function to translate text using deep_translator with enhanced error handling
def translate_text(text, target_language):
    try:
        # Initialize the translator
        translated = GoogleTranslator(source='auto', target=target_language).translate(text)
        return translated
    except Exception as e:
        # Return an error message to display in Streamlit
        return f"Error: Unable to translate text. Exception: {str(e)}"

# Sidebar Language Selection
languages = {
    "English": "en",
    "Hindi": "hi",
    "Tamil": "ta",
    "Telugu": "te",
    "Kannada": "kn",
    "Bengali": "bn",
    "Gujarati": "gu",
    "Marathi": "mr",
    "Malayalam": "ml",
}


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

        .div {
            font-size: 20px; /* Set a font size */
            line-height: 1.5; /* Line height for better readability */
            color: #333; /* Text color */
        }

        .how-it-works {
            font-size: 28px;
            font-weight: bold;
            color: #388E3C;
            margin-top: 20px;
            margin-bottom: 10px;
        }

        .how-it-works-content {
            font-size: 18px;
            color: #555;
        }

    </style>
    """, unsafe_allow_html=True)


# Language selection in sidebar
selected_language = st.sidebar.selectbox("Select Language", list(languages.keys()))  # User can select language here
target_language = languages[selected_language]  # Map selected language to its language code

# Main title with translation
original_text = "🌿 PLANT DISEASE RECOGNITION SYSTEM"
translated_text = translate_text(original_text, target_language)  # Translate the text

# Display the translated text
st.markdown(f'<h2 class="main-header">{translated_text}</h2>', unsafe_allow_html=True)

st.image("home_page.jpeg", use_column_width=True)

# Sample paragraph to be translated
original_paragraph1 = """
Our mission is to help in identifying plant diseases efficiently. Upload an image of a plant, and our system will analyze it to detect any signs of diseases. Together, let's protect our crops and ensure a healthier harvest!
"""

translated_paragraph = translate_text(original_paragraph1, target_language)

# Display the mission text
st.markdown(f'<div class="div">{translated_paragraph}</div>', unsafe_allow_html=True)

# "How It Works" Section
how_it_works_title = "🚀 How It Works"
how_it_works_title_trans= translate_text(how_it_works_title,target_language)

#st.markdown(f'<div class="div">{how_it_works_title_trans}</div>', unsafe_allow_html=True)

how_it_works_content = translate_text("""

   1. <b>Upload Image:</b> Go to the <b>Disease Recognition</b> page and upload an image of a plant with suspected diseases.
   2. <b>Analysis:</b> Our system will process the image using advanced algorithms to identify potential diseases.
   3. <b>Results:</b> View the results and recommendations for further action.
""", target_language)

# Display the "How It Works" section with proper CSS classes
st.markdown(f'<div class="how-it-works">{how_it_works_title_trans}</div>', unsafe_allow_html=True)
st.markdown(f'<div class="how-it-works-content">{how_it_works_content}</div>', unsafe_allow_html=True)
