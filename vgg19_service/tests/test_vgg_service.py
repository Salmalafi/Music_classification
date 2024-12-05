import unittest
import requests
import base64
from xmlrunner import XMLTestRunner

BASE_URL = "http://localhost:5002/classify"  # Update with your actual URL

class TestMusicGenreClassification(unittest.TestCase):

    def test_valid_audio(self):
        """Test with valid audio data"""
        with open("tests/pop.00002.wav", "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        # The correct payload format as per the Flask API
        data = {"wav_music": encoded_audio}
        response = requests.post(BASE_URL, json=data)
        
        self.assertEqual(response.status_code, 200)
        response_json = response.json()

        # Checking if the response contains the predicted genre
        self.assertIn("predicted_genre", response_json)
        self.assertIsInstance(response_json["predicted_genre"], str)

    def test_invalid_audio(self):
        """Test with invalid audio data"""
        # Send corrupted or invalid base64 audio data
        data = {"wav_music": "invalid_base64_audio_data"}
        response = requests.post(BASE_URL, json=data)
        
        self.assertEqual(response.status_code, 400)
        response_json = response.json()

        # Check that the error message is related to the invalid audio
        self.assertIn("error", response_json)
        self.assertEqual(response_json["error"], "Error during prediction")

    def test_missing_audio(self):
        """Test with no audio data"""
        response = requests.post(BASE_URL, json={})
        
        self.assertEqual(response.status_code, 400)
        response_json = response.json()

        # Check that the error message mentions missing audio data
        self.assertIn("error", response_json)
        self.assertEqual(response_json["error"], "No audio data provided")

if __name__ == "__main__":
    # Run the tests and save the results in an XML file
    with open("test_results.xml", "wb") as output:
        unittest.main(testRunner=XMLTestRunner(output=output))

