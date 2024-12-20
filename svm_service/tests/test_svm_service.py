import unittest
import requests
import base64
from xmlrunner import XMLTestRunner

BASE_URL = "http://localhost:5000/classify" 

class TestSVMService(unittest.TestCase):

    def test_valid_audio(self):
        """Test with valid audio data"""
        with open("tests/pop.00002.wav", "rb") as audio_file:
            encoded_audio = base64.b64encode(audio_file.read()).decode('utf-8')

        data = {"music_data": encoded_audio}
        response = requests.post(BASE_URL, json=data)
        
        self.assertEqual(response.status_code, 200)  
        response_json = response.json()
        self.assertIn("received_message", response_json)
        self.assertEqual(response_json["received_message"], "Music file received and processed successfully")
        self.assertIn("response", response_json)

    def test_invalid_audio(self):
        """Test with invalid audio data"""
        data = {"music_data": "invalid_base64_audio_data"} 
        response = requests.post(BASE_URL, json=data)
        
        self.assertEqual(response.status_code, 400) 
        response_json = response.json()
        self.assertIn("received_message", response_json)
        self.assertEqual(response_json["received_message"], "An error occurred during prediction")
        self.assertIn("error", response_json)

    def test_missing_audio(self):
        """Test with no audio data"""
        response = requests.post(BASE_URL, json={})
        
        self.assertEqual(response.status_code, 400)  
        response_json = response.json()
        self.assertIn("received_message", response_json)
        self.assertEqual(response_json["received_message"], "No music file received")
        self.assertIn("response", response_json)
        self.assertEqual(response_json["response"], "Error") 

if __name__ == "__main__":
    with open("test_results.xml", "wb") as output:
        unittest.main(testRunner=XMLTestRunner(output=output))
