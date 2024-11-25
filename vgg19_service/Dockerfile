# Base image with Python 3.9
FROM python:3.9-slim-buster

# Set the working directory in the container
WORKDIR ./

# Set environment variables
ENV STATIC_URL /static
ENV STATIC_PATH /app/static

# Copy requirements file and install dependencies
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt
# Copy the pre-trained model
COPY /app/vgg19_music_classifier.h5 /app/vgg19_music_classifier.h5

# Copy the application code into the container
COPY . .

# Expose the application port
EXPOSE 5002

# Create a shared directory for spectrogram images
RUN mkdir -p /shared/images

# Set the default command to run the Flask app
CMD ["python", "/app/app.py"]
