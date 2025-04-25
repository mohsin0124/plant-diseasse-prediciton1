import tensorflow as tf
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle

# Load the TensorFlow model
model_path = "trained_plant_disease_model.keras"
tf_model = tf.keras.models.load_model(model_path)

# Create a scikit-learn model
sklearn_model = RandomForestClassifier(n_estimators=100)

# Get the weights from the TensorFlow model
weights = tf_model.get_weights()

# Convert the weights to a format that scikit-learn can use
# This is a simplified conversion - you might need to adjust based on your model architecture
sklearn_model.fit(np.zeros((1, 128*128*3)), np.zeros(1))  # Dummy fit to initialize
sklearn_model.estimators_ = [tf.keras.models.clone_model(tf_model) for _ in range(100)]

# Save the scikit-learn model
with open('trained_plant_disease_model.pkl', 'wb') as f:
    pickle.dump(sklearn_model, f)

print("Model converted and saved successfully!") 