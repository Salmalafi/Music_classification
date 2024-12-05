import os
import numpy as np
import base64
from io import BytesIO
from flask import Flask, request, jsonify
from pydub import AudioSegment
from flask_cors import CORS
import logging
import tensorflow as tf

# Disable GPU (use CPU only)
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# Initialize Flask app
app = Flask(__name__)
CORS(app, resources={r"/*": {"origins": "*"}})

# Set up logging
logging.basicConfig(level=logging.DEBUG)
base_path = os.path.abspath(os.path.dirname(__file__))

# Load the pre-trained TensorFlow Lite model
tflite_model_path = os.path.join(base_path, "vgg_model.tflite")

# Try loading the TFLite model and log the error if it fails
try:
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    logging.info("TFLite model loaded successfully.")
except Exception as e:
    logging.error(f"Error loading TFLite model: {e}")
    interpreter = None  # Set interpreter to None if it fails to load

# Genre labels for classification (ensure these match your model's training labels)
genre_labels = ['classical', 'rock', 'pop', 'hiphop', 'jazz', 'blues', 'metal', 'reggae', 'disco', 'country']

def preprocess_audio(audio_segment):
    """
    Preprocess audio to match the input shape expected by the model.
    """
    # Convert the audio segment to a numpy array (samples are in int16 format)
    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=np.float32)

    # Get the sample rate
    sr = audio_segment.frame_rate

    # Normalize audio data
    audio_data = audio_data / np.max(np.abs(audio_data))

    # Make sure the audio is 150x150 (model's expected size)
    # You can either trim or pad the audio
    if len(audio_data) < 150 * 150:
        padding = 150 * 150 - len(audio_data)
        audio_data = np.pad(audio_data, (0, padding), 'constant')
    else:
        audio_data = audio_data[:150 * 150]

    # Reshape the audio data to match the input shape expected by the model
    audio_data = audio_data.reshape((1, 150, 150, 1))

    return audio_data

def predict_genre_with_tflite(features):
    """
    Predict the genre using the TensorFlow Lite model.
    """
    try:
        # Set the input tensor
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        logging.debug(f"Input details: {input_details}")
        logging.debug(f"Output details: {output_details}")

        interpreter.set_tensor(input_details[0]['index'], features)
        interpreter.invoke()

        # Get the prediction output
        output_data = interpreter.get_tensor(output_details[0]['index'])

        # Get the predicted genre (label with the highest probability)
        predicted_genre = genre_labels[np.argmax(output_data)]

        return predicted_genre
    except Exception as e:
        logging.error(f"Error in TFLite prediction: {str(e)}")
        return None

@app.route('/classify', methods=['POST'])
def classify_music():
    try:
        data = request.json

        if 'wav_music' not in data:
            return jsonify({'error': 'No audio data provided'}), 400

        # Decode the base64-encoded audio data
        audio_data = base64.b64decode(data['wav_music'])
        audio_segment = AudioSegment.from_file(BytesIO(audio_data), format='wav')

        # Preprocess the audio to match the model's expected input shape
        features = preprocess_audio(audio_segment)

        if interpreter is None:
            return jsonify({'error': 'Model not loaded, unable to predict genre'}), 500

        # Predict the genre using the TensorFlow Lite model
        predicted_genre = predict_genre_with_tflite(features)

        if predicted_genre is None:
            return jsonify({'error': 'Error during prediction'}), 500

        return jsonify({'predicted_genre': predicted_genre})

    except Exception as e:
        logging.error(f"Error in classification: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5002)
