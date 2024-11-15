<<<<<<< HEAD
=======
// Function to predict the genre for music using VGG19
>>>>>>> b0a666358a5f0dc76159d0990b7a14afd01b7b46
async function predictGenre() {
    const reader = new FileReader();
    reader.onload = async function(event) {
        const base64Audio = event.target.result.split(',')[1];

        try {
<<<<<<< HEAD
            const response = await fetch('http://localhost:9000/predict_vgg', { // Correct API for VGG19
=======
            const response = await fetch('http://localhost:5001/predict', { // Correct API for VGG19
>>>>>>> b0a666358a5f0dc76159d0990b7a14afd01b7b46
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ wav_music: base64Audio })
            });

            const result = await response.json();
            if (response.ok) {
                document.getElementById('image-result').textContent = "Predicted Genre (VGG19): " + result.predicted_genre;
            } else {
                document.getElementById('image-result').textContent = "Error: " + result.error;
            }
        } catch (error) {
            console.error("Error:", error);
            document.getElementById('image-result').textContent = "Error: " + error.message;
        }
    };

    reader.readAsDataURL(audioFile);
}

<<<<<<< HEAD
=======
// Function to predict the genre for music using the SVM model
>>>>>>> b0a666358a5f0dc76159d0990b7a14afd01b7b46
async function predictSvmGenre() {
    const audioFile = document.getElementById('audio-file').files[0];
    if (!audioFile) {
        alert("Please select an audio file.");
        return;
    }

    const reader = new FileReader();
    reader.onload = async function(event) {
        const base64Audio = event.target.result.split(',')[1];

        try {
<<<<<<< HEAD
            const response = await fetch('http://localhost:5000/predict', { 
=======
            const response = await fetch('http://localhost:5000/predict', { // Correct API for SVM
>>>>>>> b0a666358a5f0dc76159d0990b7a14afd01b7b46
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify({ wav_music: base64Audio })
            });

            const result = await response.json();
            if (response.ok) {
                document.getElementById('music-result').textContent = "Predicted Genre (SVM): " + result.predicted_genre;
            } else {
                document.getElementById('music-result').textContent = "Error: " + result.error;
            }
        } catch (error) {
            console.error("Error:", error);
            document.getElementById('music-result').textContent = "Error: " + error.message;
        }
    };

    reader.readAsDataURL(audioFile);
}
