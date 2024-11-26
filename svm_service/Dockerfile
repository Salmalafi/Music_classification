FROM python:3.9-slim-buster

# Set the working directory to the current directory
WORKDIR ./

# Set environment variables for static files (if needed)
ENV STATIC_URL /static
ENV STATIC_PATH /static

# Copy the requirements file and install dependencies
COPY ./requirements.txt ./requirements.txt
RUN pip install -r ./requirements.txt

# Copy the model file
COPY ./app/svm-classification-model.h5 ./app/svm-classification-model.h5

# Copy the tests directory into the container
COPY ./tests ./tests

# Copy the entire application code into the current directory
COPY . .

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "/app/app.py"]
