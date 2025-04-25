# Plant Disease Recognition System

A web application that uses machine learning to identify plant diseases from images. The system provides accurate disease detection, treatment recommendations, and weather monitoring for optimal plant care.

## Features

- **Disease Recognition**: Upload images of plants to detect diseases using a trained machine learning model
- **Weather Monitoring**: Check weather conditions for your location to get plant care recommendations
- **Nearby Shops**: Find nearby agricultural shops for purchasing treatments
- **Multilingual Support**: Available in multiple Indian languages
- **Retro Theme UI**: Modern application with a nostalgic retro computer terminal aesthetic

## Technologies Used

- **Frontend**: Streamlit
- **Machine Learning**: TensorFlow, Keras
- **APIs**: OpenWeather API, AccuWeather API, Groq API
- **Styling**: Custom CSS with retro theme

## Getting Started

1. Clone the repository
2. Install the required dependencies:
   ```
   pip install streamlit tensorflow pillow requests groq deep-translator
   ```
3. Run the application:
   ```
   streamlit run imadtemp.py
   ```

## Project Structure

- `imadtemp.py`: Main application file with all functionality
- `trained_plant_disease_model.keras`: Trained model for plant disease recognition
- `home_page.jpeg`: Home page image

## License

This project is licensed under the MIT License - see the LICENSE file for details. 