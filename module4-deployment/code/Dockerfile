FROM agrigorev/zoomcamp-model:mlops-2024-3.10.13-slim

WORKDIR /app

# Install dependencies
RUN pip install --no-cache-dir pandas==2.2.2 scikit-learn==1.5.0 pyarrow==17.0.0 numpy==1.26.4

# Copy the script
COPY homework.py .

# Command to run the script, arguments to be provided at runtime
ENTRYPOINT ["python", "homework.py"]