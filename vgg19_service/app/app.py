import os
import numpy as np
import base64
from io import BytesIO
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
from flask import Flask, request, jsonify
from pydub import AudioSegment
from flask_cors import CORS
import logging
import matplotlib.pyplot as plt

# Disable GPU (use CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)
CORS(app)

# Set up logging
logging.basicConfig(level=logging.DEBUG)
base_path = os.path.abspath(os.path.dirname(__file__))

# Load the pre-trained image model
model_path = os.path.join(base_path, "vgg19_music_classifier.h5")

# Try loading the VGG19 model and log the error if it fails
try:
    model = load_model(model_path)
    logging.info("Model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading model: {e}")
    model = None  # Set model to None if it fails to load

# Genre labels for classification (ensure these match your model's training labels)
genre_labels = ['classical', 'rock', 'pop', 'hiphop', 'jazz', 'blues', 'metal', 'reggae', 'disco', 'country']

def extract_features_from_audio(audio_segment):
    """
    Convert audio to a spectrogram and prepare it for VGG19 model input.
    """
    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=float)
    sr = audio_segment.frame_rate
    spectrogram = librosa.stft(audio_data)
    spectrogram = np.abs(spectrogram)  # Convert to magnitude spectrogram
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)  # Log scale

    # Prepare the spectrogram as an image
    fig, ax = plt.subplots(figsize=(2, 2))  # Create a 2x2 inch figure
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.axis('off')  # Hide the axes

    img_io = BytesIO()
    fig.savefig(img_io, format='png')  # Save to BytesIO buffer
    img_io.seek(0)

    img = image.load_img(img_io, target_size=(224, 224))  
    img_array = image.img_to_array(img)  
    img_array = np.expand_dims(img_array, axis=0) 
    img_array = preprocess_input(img_array)  # Preprocess for VGG19

    return img_array

@app.route('/classify', methods=['POST'])
def predict_genre():
    try:
        data = request.json

        if 'wav_music' not in data:
            return jsonify({'error': 'No audio data provided'}), 400

        # Decode the base64-encoded audio data
        audio_data = base64.b64decode(data['wav_music'])
        audio_segment = AudioSegment.from_file(BytesIO(audio_data), format='wav')

        # Extract features from the audio file
        features = extract_features_from_audio(audio_segment)

        if model is None:
            return jsonify({'error': 'Model not loaded, unable to predict genre'}), 500

        # Predict the genre using the VGG19 model
        prediction = model.predict(features)

        # Get the predicted genre (label with the highest probability)
        predicted_genre = genre_labels[np.argmax(prediction)]

        return jsonify({'predicted_genre': predicted_genre})

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002)
