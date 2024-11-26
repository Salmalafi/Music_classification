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
model = load_model(model_path)

genre_dict = {0: 'blues', 1: 'classical', 2: 'country', 3: 'disco',
              4: 'hiphop', 5: 'jazz', 6: 'metal', 7: 'pop', 8: 'reggae', 9: 'rock'}

def extract_features(file_path):
    # Load audio file with librosa
    audio, sample_rate = librosa.load(file_path, res_type='kaiser_fast')
    # Extract MFCCs and other features...
    mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=20)
    mfccs_processed = np.mean(mfccs.T, axis=0)
    return mfccs_processed

def predict_genre(file_path):
    # Extract features from the audio file
    features = extract_features(file_path)
    
    # Reshape features to fit the model input format
    features = np.reshape(features, (1, -1))  # Reshape for model
    
    # Predict genre
    prediction = model.predict(features)
    print("Raw Prediction:", prediction)
    
    # Convert prediction to genre
    predicted_genre = np.argmax(prediction)
    predicted_genre_name = genre_dict.get(predicted_genre, "Unknown Genre")
    
    return predicted_genre_name

@@app.route('/classify', methods=['POST'])
def classify():
    data = request.get_json()
    
    if data and "music_data" in data:
        encoded_music_data = data["music_data"]
        try:
            # Attempt to decode base64
            decoded_music_data = base64.b64decode(encoded_music_data)
        except base64.binascii.Error as e:
            # Handle invalid base64 input
            return jsonify({
                "received_message": "An error occurred during prediction",
                "error": "Invalid base64 data provided"
            }), 200

        # Use a temporary directory for the audio file
        temp_dir = tempfile.mkdtemp()
        temp_wav_file = os.path.join(temp_dir, 'temp_audio.wav')

        try:
            # Write decoded data to a temporary file
            with open(temp_wav_file, 'wb') as temp_file:
                temp_file.write(decoded_music_data)

            # Make the prediction
            genre = predict_genre(temp_wav_file)

            response_data = {
                "received_message": "Music file received and processed successfully",
                "response": genre,
            }
        except Exception as e:
            # Handle errors during prediction
            response_data = {
                "received_message": "An error occurred during prediction",
                "error": str(e),
            }
        finally:
            os.remove(temp_wav_file)
            os.rmdir(temp_dir)
    else:
        response_data = {
            "received_message": "No music file received",
            "response": "Error"
        }

    return jsonify(response_data)


if __name__ == '__main__':
    # Run the Flask app
    app.run(host='0.0.0.0', port=5000)
