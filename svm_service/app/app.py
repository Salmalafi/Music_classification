import tempfile
import os
from flask import Flask, request, jsonify
import base64
import librosa
import numpy as np
from tensorflow.keras.models import load_model
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Get the absolute path to the directory of this script
base_path = os.path.abspath(os.path.dirname(__file__))

# Load the pre-trained model
model_path = os.path.join(base_path, "svm-classification-model.h5")
try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model: {str(e)}")

# Define genre dictionary
genre_dict = {
    0: 'blues', 1: 'classical', 2: 'country', 3: 'disco',
    4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'
}

def extract_features(file_path):
    """
    Extract MFCC features from the audio file.
    """
    try:
        audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
        mfccs_processed = np.mean(mfccs.T, axis=0)
        return mfccs_processed
    except Exception as e:
        raise ValueError(f"Error extracting features: {str(e)}")

def predict_genre(file_path):
    """
    Predict the genre of the given audio file.
    """
    try:
        features = extract_features(file_path)
        features = np.reshape(features, (1, -1))  # Reshape for model input
        prediction = model.predict(features)
        predicted_genre = np.argmax(prediction)
        return genre_dict.get(predicted_genre, "Unknown Genre")
    except Exception as e:
        raise ValueError(f"Error during prediction: {str(e)}")

@app.route('/classify', methods=['POST'])
def classify():
    """
    Handle POST requests to classify music genre from audio data.
    """
    try:
        data = request.get_json()
        
        # Validate request body
        if not data or "music_data" not in data:
            return jsonify({
                "received_message": "No music file received",
                "response": "Error"
            }), 400
        
        # Decode the base64 audio data
        encoded_music_data = data["music_data"]
        try:
            decoded_music_data = base64.b64decode(encoded_music_data)
        except Exception as e:
            return jsonify({
                "received_message": "Invalid base64 data provided",
                "error": str(e)
            }), 400

        # Use a temporary file for the audio data
        with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as temp_wav_file:
            temp_wav_file.write(decoded_music_data)
            temp_file_path = temp_wav_file.name

        # Predict genre
        genre = predict_genre(temp_file_path)

        # Clean up temporary file
        os.remove(temp_file_path)

        return jsonify({
            "received_message": "Music file received and processed successfully",
            "response": genre
        }), 200

    except ValueError as e:
        return jsonify({
            "received_message": "An error occurred during processing",
            "error": str(e)
        }), 400

    except Exception as e:
        return jsonify({
            "received_message": "Unexpected error occurred",
            "error": str(e)
        }), 500

if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
