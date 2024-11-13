from flask import Flask, request, jsonify
import joblib
import numpy as np
import base64
from io import BytesIO
import librosa
import librosa.display
from tensorflow.keras.applications import VGG19
from tensorflow.keras.preprocessing import image
from tensorflow.keras.applications.vgg19 import preprocess_input
import logging
from pydub import AudioSegment
from flask_cors import CORS

# Initialize Flask app
app = Flask(__name__)
CORS(app)  # Enable CORS for all routes

# Set up logging
logging.basicConfig(level=logging.DEBUG)

# Load the VGG19 model (ensure you have the appropriate model files)
vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

# Example genre list (you may need to adjust this to match your dataset)
genre_labels = ['classical', 'rock', 'pop', 'hiphop', 'jazz', 'blues', 'metal', 'reggae', 'disco', 'country']

def extract_features_from_audio(audio_segment):
    # Convert audio segment to a numpy array
    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=float)
    sr = audio_segment.frame_rate

    # Generate a spectrogram
    spectrogram = librosa.stft(audio_data)  # Compute short-time Fourier transform
    spectrogram = np.abs(spectrogram)  # Get the magnitude of the STFT

    # Convert spectrogram to log scale
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)

    # Convert spectrogram to an image
    # This will plot the spectrogram, which will be saved as an image for VGG19 input
    fig, ax = plt.subplots(figsize=(2, 2))  # Set figure size to match VGG19 input size
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.axis('off')  # Remove axes for a clean image

    # Save the spectrogram to a BytesIO object
    img_io = BytesIO()
    fig.savefig(img_io, format='png')
    img_io.seek(0)
    
    # Load the image from BytesIO into a format that VGG19 can process
    img = image.load_img(img_io, target_size=(224, 224))  # Resize to match VGG19 input size
    img_array = image.img_to_array(img)  # Convert the image to an array
    img_array = np.expand_dims(img_array, axis=0)  # Add batch dimension
    img_array = preprocess_input(img_array)  # Preprocess the image for VGG19
    
    return img_array

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

        # Extract features from the audio segment for VGG19
        features = extract_features_from_audio(audio_segment)

        # Predict the genre using VGG19
        prediction = vgg_model.predict(features)

        # You will need to process the prediction to map it to genre labels
        predicted_genre = genre_labels[np.argmax(prediction)]  # Example processing

        return jsonify({'predicted_genre': predicted_genre})

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5001)
