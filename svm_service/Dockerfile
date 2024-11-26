FROM python:3.9-slim-buster

# Set the working directory to /app
WORKDIR /app

# Set environment variables for static files (if needed)
ENV STATIC_URL /static
ENV STATIC_PATH /app/static

# Copy the requirements file and install dependencies
COPY ./requirements.txt /app/requirements.txt
RUN pip install -r /app/requirements.txt

# Copy the model file
COPY ./app/svm-classification-model.h5 /app/svm-classification-model.h5

# Copy the tests directory into /app/tests in the container
COPY ./tests /app/tests

# Copy the entire application code into /app
COPY . .

# Ensure correct permissions (optional but useful in case of permission issues)
RUN chmod -R 755 /app

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "/app/app.py"]

