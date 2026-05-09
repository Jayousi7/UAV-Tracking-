FROM python:3.10-slim

# Install system dependencies required by OpenCV
RUN apt-get update && apt-get install -y \
    libgl1 \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the requirements file and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
# Set library paths so ONNX Runtime can find the pip-installed CUDA/cuDNN libraries
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.10/site-packages/nvidia/cudnn/lib:/usr/local/lib/python3.10/site-packages/nvidia/cublas/lib:${LD_LIBRARY_PATH}

# Copy the rest of the application code
COPY . .

# Expose the FastAPI port
EXPOSE 8000

# Start the application
CMD ["python", "-m", "uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
