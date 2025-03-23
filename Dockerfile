# Use a base image with Python
FROM python:3.8-slim

# Set environment variables to prevent buffering issues in Python
ENV PYTHONUNBUFFERED 1

# Install required system dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    libssl-dev \
    libffi-dev \
    libmysqlclient-dev \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory to /app
WORKDIR /app

# Copy your project requirements file
COPY requirements.txt /app/

# Install Python dependencies
RUN pip install --upgrade pip && pip install -r requirements.txt

# Copy your application code into the container
COPY . /app/

# Run the application (assuming you're using Flask, FastAPI, or another web framework)
CMD ["python", "app.py"]
