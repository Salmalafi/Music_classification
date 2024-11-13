from flask import Flask, request, jsonify, render_template
import joblib
import numpy as np
import pandas as pd
import base64
from io import BytesIO
import librosa
import logging
from pydub import AudioSegment
from flask_cors import CORS
# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the pre-trained SVM model and scaler
model = joblib.load('svm_genre_model.pkl')
scaler = joblib.load('scaler.pkl')  # Ensure this path is correct

# Example genre list
genre_labels = ['classical', 'rock', 'pop', 'hiphop', 'jazz', 'blues', 'metal', 'reggae', 'disco', 'country']

def extract_features_from_audio(audio_segment):
    # Convert audio segment to a numpy array
    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=float)
    sr = audio_segment.frame_rate

    # Feature Extraction
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sr, n_mfcc=20)
    mfccs_mean = np.mean(mfccs, axis=1)  # Mean of MFCCs
    mfccs_std = np.std(mfccs, axis=1)    # Std of MFCCs
    mfccs_features = np.concatenate((mfccs_mean, mfccs_std))

    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sr)
    chroma_mean = np.mean(chroma, axis=1)
    chroma_features = chroma_mean

    spectral_centroid = np.mean(librosa.feature.spectral_centroid(y=audio_data, sr=sr))
    spectral_bandwidth = np.mean(librosa.feature.spectral_bandwidth(y=audio_data, sr=sr))
    spectral_rolloff = np.mean(librosa.feature.spectral_rolloff(y=audio_data, sr=sr))

    zcr = np.mean(librosa.feature.zero_crossing_rate(y=audio_data))
    rms = np.mean(librosa.feature.rms(y=audio_data))

    features = np.concatenate((
        mfccs_features,
        chroma_features,
        [spectral_centroid, spectral_bandwidth, spectral_rolloff],
        [zcr, rms]
    ))

    if features.size != 57:
        raise ValueError(f'Expected 57 features, but got {features.size}. Features: {features}')

    return pd.DataFrame([features])

@app.route('/predict', methods=['POST'])
def predict_genre():
    try:
        data = request.json

        # Check if audio data exists
        if 'wav_music' not in data:
            return jsonify({'error': 'No audio data provided'}), 400

        # Decode audio from base64 and load it into AudioSegment
        audio_data = base64.b64decode(data['wav_music'])
        audio_segment = AudioSegment.from_file(BytesIO(audio_data), format='wav')

        # Extract features from the audio segment
        features = extract_features_from_audio(audio_segment)

        # Ensure the correct feature shape
        expected_shape = (1, 57)
        if features.shape != expected_shape:
            return jsonify({'error': f'Expected input shape: {expected_shape}, but got {features.shape}'}), 400

        # Scale the features
        scaled_features = scaler.transform(features)

        # Predict the genre (ensure it returns the genre name, not an index)
        prediction = model.predict(scaled_features)
        
        # If the prediction returns a genre name directly, use it
        predicted_genre = prediction[0]  # Directly access the genre string if it's already a label

        return jsonify({'predicted_genre': predicted_genre})

    except Exception as e:
        # Log the error and return the error message
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500


if __name__ == '__main__':
    app.run(port=5000)
