# Use lightweight official Python image
FROM python:3.10-slim

# Install system dependencies needed for compiling certain libraries (if necessary)
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory inside the container
WORKDIR /app

# Copy dependency file first to leverage Docker build cache
COPY requirements.txt .

# Install Python packages
RUN pip install --no-cache-dir -r requirements.txt

# Copy the core code and execution scripts
COPY . .

# Default command: Runs the scenario comparison
CMD ["python", "scripts/run_all_scenarios.py"]
