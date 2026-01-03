# Use Python 3.10 slim image as base
FROM python:3.10.19-slim

# Set working directory
WORKDIR /app

# Install system dependencies
RUN apt update -y && apt install awscli -y

# Copy requirements first for better caching
COPY requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy the application code
COPY . /app

# Expose the port the app runs on
EXPOSE 8000

# Command to run the application
CMD ["python", "application.py"]