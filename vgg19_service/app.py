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
app = Flask(__name__)
CORS(app) 

# Set up logging
logging.basicConfig(level=logging.DEBUG)

vgg_model = VGG19(weights='imagenet', include_top=False, input_shape=(224, 224, 3))

genre_labels = ['classical', 'rock', 'pop', 'hiphop', 'jazz', 'blues', 'metal', 'reggae', 'disco', 'country']

def extract_features_from_audio(audio_segment):
    audio_data = np.array(audio_segment.get_array_of_samples(), dtype=float)
    sr = audio_segment.frame_rate
    spectrogram = librosa.stft(audio_data)  
    spectrogram = np.abs(spectrogram)  
    log_spectrogram = librosa.amplitude_to_db(spectrogram, ref=np.max)
    fig, ax = plt.subplots(figsize=(2, 2)) 
    librosa.display.specshow(log_spectrogram, sr=sr, x_axis='time', y_axis='log', ax=ax)
    ax.axis('off') 

    img_io = BytesIO()
    fig.savefig(img_io, format='png')
    img_io.seek(0)
    img = image.load_img(img_io, target_size=(224, 224)) 
    img_array = image.img_to_array(img) 
    img_array = np.expand_dims(img_array, axis=0)  
    img_array = preprocess_input(img_array)  
    
    return img_array

@app.route('/predict', methods=['POST'])
def predict_genre():
    try:
        data = request.json
    
        if 'wav_music' not in data:
            return jsonify({'error': 'No audio data provided'}), 400
        audio_data = base64.b64decode(data['wav_music'])
        audio_segment = AudioSegment.from_file(BytesIO(audio_data), format='wav')
        features = extract_features_from_audio(audio_segment)
        prediction = vgg_model.predict(features)
        predicted_genre = genre_labels[np.argmax(prediction)] 

        return jsonify({'predicted_genre': predicted_genre})

    except Exception as e:
        logging.error(f"Error in prediction: {str(e)}")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    app.run(port=5002)
