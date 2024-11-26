FROM python:3.9-slim-buster

WORKDIR ./ 

ENV STATIC_URL /static
ENV STATIC_PATH /app/static

# Copy the requirements file and install dependencies
COPY ./requirements.txt /requirements.txt
RUN pip install -r /requirements.txt

# Copy the model file
COPY ./app/svm-classification-model.h5 /app/svm-classification-model.h5
COPY ./tests ./tests

# Copy the entire application code
COPY . .

# Ensure correct permissions (optional but useful in case of permission issues)
RUN chmod -R 755 /app

# Expose the port the app will run on
EXPOSE 5000

# Command to run the application
CMD ["python", "/app/app.py"]
