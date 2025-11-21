# Use a slim Python base image
FROM python:3.10-slim

# Set working directory
WORKDIR /app

# Install system dependencies
# tesseract-ocr is for OCR
# libgl1 is the modern replacement for libgl1-mesa-glx
RUN apt-get update && apt-get install -y \
    tesseract-ocr \
    libgl1 \
    && rm -rf /var/lib/apt/lists/*

# Copy and install Python requirements
COPY requirements.txt ./

# --- This is the fix for small /tmp partitions ---
# We create our own temp dir, tell pip to use it, then delete it.
RUN mkdir /pip_tmp && \
    TMPDIR=/pip_tmp pip install --no-cache-dir -r requirements.txt && \
    rm -rf /pip_tmp

# Copy the rest of the application code
COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the app using uvicorn
# --host 0.0.0.0 makes it accessible from other containers
CMD ["uvicorn", "fastapi_app:app", "--host", "0.0.0.0", "--port", "8000"]