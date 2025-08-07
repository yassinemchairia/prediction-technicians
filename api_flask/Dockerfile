# Use a lightweight Python base image
FROM python:3.9-slim

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY api_prediction.py .
COPY technicien_model.pkl .
COPY model_features.txt .

# Expose port 5001
EXPOSE 5001

# Command to run the Flask app
CMD ["python", "api_prediction.py"]
