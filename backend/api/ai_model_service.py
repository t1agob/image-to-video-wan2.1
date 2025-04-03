from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import FileResponse
from typing import List
import os
import torch
from torch import autocast
from PIL import Image
import uuid
import shutil
from pathlib import Path

app = FastAPI()

# Set up model paths
MODEL_PATH = "./Wan2.1-I2V-14B-720P"
OUTPUT_DIR = "./output"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Load model on startup
model = None
device = "cuda" if torch.cuda.is_available() else "cpu"

@app.on_event("startup")
async def load_model():
    global model
    try:
        # Import here to avoid loading until startup
        from diffusers import WanI2VPipeline
        
        # Load the model
        model = WanI2VPipeline.from_pretrained(
            MODEL_PATH,
            torch_dtype=torch.float16,
            variant="fp16"
        )
        model = model.to(device)
        print(f"Model loaded on {device}")
    except Exception as e:
        print(f"Error loading model: {e}")
        model = None

@app.get("/")
def read_root():
    return {"message": "Hello, World!"}

@app.post("/generate_video")
async def generate_video(
    images: List[UploadFile] = File(...),
    num_frames: int = 16,
    fps: int = 8,
):
    if model is None:
        raise HTTPException(status_code=500, detail="Model not loaded")
    
    if len(images) == 0:
        raise HTTPException(status_code=400, detail="No images provided")
    
    # Create unique ID for this generation
    generation_id = str(uuid.uuid4())
    temp_dir = os.path.join(OUTPUT_DIR, generation_id)
    os.makedirs(temp_dir, exist_ok=True)
    
    try:
        # Save uploaded images to temporary directory
        image_paths = []
        for i, img_file in enumerate(images):
            img_path = os.path.join(temp_dir, f"input_{i}.png")
            with open(img_path, "wb") as buffer:
                shutil.copyfileobj(img_file.file, buffer)
            image_paths.append(img_path)
        
        # Load the first image (or you can process multiple images if the model supports it)
        input_image = Image.open(image_paths[0]).convert("RGB")
        
        # Generate video with Wan2.1
        with autocast(device):
            frames = model(
                input_image, 
                num_frames=num_frames,
                fps=fps,
                height=720,  # This model is trained for 720p
                width=1280,
            ).frames
        
        # Save frames as video
        output_path = os.path.join(OUTPUT_DIR, f"{generation_id}.mp4")
        
        # Convert frames to video
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            format="MP4",
            duration=1000//fps,
            loop=0
        )
        
        # Return video file
        return FileResponse(
            path=output_path, 
            media_type="video/mp4", 
            filename=f"generated_video.mp4"
        )
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video generation failed: {str(e)}")
    finally:
        # Clean up temporary files
        if os.path.exists(temp_dir):
            shutil.rmtree(temp_dir)