FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime AS modelbase

# Set the working directory
WORKDIR /app/model

# Install system dependencies
RUN apt-get update && apt-get install -y \
    git \
    wget \
    && rm -rf /var/lib/apt/lists/*

# Install huggingface_hub and the model
RUN pip install --no-cache-dir "huggingface_hub[cli]"
RUN huggingface-cli download Wan-AI/Wan2.1-I2V-14B-720P --local-dir ./Wan2.1-I2V-14B-720P

FROM pytorch/pytorch:2.0.0-cuda11.7-cudnn8-runtime

# Set the working directory
WORKDIR /app

# Install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the model from the modelbase stage in different steps to optimize for paralel docker layer download
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/config.json /app/model/Wan2.1-I2V-14B-720P/config.json
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00001-of-00007.safetensors
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00002-of-00007.safetensors
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00003-of-00007.safetensors
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00004-of-00007.safetensors
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00005-of-00007.safetensors
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00006-of-00007.safetensors
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model-00007-of-00007.safetensors
COPY --from=modelbase /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors.inde.json /app/model/Wan2.1-I2V-14B-720P/diffusion_pytorch_model.safetensors.inde.json

# Copy the application code
COPY ./api /app/api

# Expose the port for FastAPI
EXPOSE 8000

# Set environment variables
ENV PYTHONPATH=/app

# Run the FastAPI application
CMD ["uvicorn", "api.ai_model_service:app", "--host", "0.0.0.0", "--port", "8000"]