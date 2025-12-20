#!/usr/bin/env python3
"""
FastAPI server for Image-to-3D Gaussian Splat generation
Competition submission for subnet architecture
"""
import os
os.environ['OPENCV_IO_ENABLE_OPENEXR'] = '1'
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

import io
import time
import asyncio
from pathlib import Path
from typing import Optional

import cv2
import torch
import numpy as np
from PIL import Image
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import StreamingResponse, JSONResponse
import uvicorn

from trellis2.pipelines import Trellis2ImageTo3DPipeline
from trellis2.utils import render_utils
from trellis2.renderers import EnvMap
import o_voxel

app = FastAPI(title="Image-to-3D Gaussian Splat Generator")

# Global state
pipeline: Optional[Trellis2ImageTo3DPipeline] = None
is_ready = False
generation_lock = asyncio.Lock()


def mesh_to_gaussian_splat_ply(mesh_with_voxel, max_points: int = 1000000) -> bytes:
    """
    Convert MeshWithVoxel to Gaussian Splat PLY format using o-voxel.
    Uses the voxel representation directly (more accurate than mesh sampling).
    """
    # Extract voxel data from MeshWithVoxel object
    coords = mesh_with_voxel.coords  # Voxel coordinates (N, 3)
    attrs = mesh_with_voxel.attrs    # Voxel attributes (N, C)
    layout = mesh_with_voxel.layout  # Attribute layout dict
    
    # Limit number of voxels if necessary
    if len(coords) > max_points:
        # Random sampling
        indices = torch.randperm(len(coords))[:max_points]
        coords = coords[indices]
        attrs = attrs[indices]
    
    # Convert voxel coordinates to world positions
    # coords are in voxel space, convert to [-0.5, 0.5] range
    positions = coords.float() * mesh_with_voxel.voxel_size + torch.tensor(
        mesh_with_voxel.origin, 
        device=coords.device, 
        dtype=torch.float32
    )
    
    # Prepare attributes dictionary for PLY export
    attr_dict = {}
    
    # Extract base color (RGB) from attributes
    if 'base_color' in layout:
        base_color_slice = layout['base_color']
        base_color = attrs[:, base_color_slice]
        # Convert to 0-255 range if normalized
        if base_color.max() <= 1.0:
            base_color = (base_color * 255).clamp(0, 255).byte()
        attr_dict['base_color'] = base_color.cpu()
    
    # Extract opacity/alpha if available
    if 'alpha' in layout:
        alpha_slice = layout['alpha']
        alpha = attrs[:, alpha_slice]
        if alpha.max() <= 1.0:
            alpha = (alpha * 255).clamp(0, 255).byte()
        attr_dict['alpha'] = alpha.cpu()
    
    # Extract metallic and roughness if available (for PBR)
    if 'metallic' in layout:
        metallic_slice = layout['metallic']
        metallic = attrs[:, metallic_slice]
        if metallic.max() <= 1.0:
            metallic = (metallic * 255).clamp(0, 255).byte()
        attr_dict['metallic'] = metallic.cpu()
    
    if 'roughness' in layout:
        roughness_slice = layout['roughness']
        roughness = attrs[:, roughness_slice]
        if roughness.max() <= 1.0:
            roughness = (roughness * 255).clamp(0, 255).byte()
        attr_dict['roughness'] = roughness.cpu()
    
    # Write to PLY format using o_voxel
    ply_buffer = io.BytesIO()
    o_voxel.io.write_ply(ply_buffer, positions.cpu(), attr_dict)
    ply_buffer.seek(0)
    
    return ply_buffer.read()


@app.on_event("startup")
async def startup_event():
    """Initialize the model on startup"""
    global pipeline, is_ready
    
    print("🚀 Starting model initialization...")
    start_time = time.time()
    
    try:
        # Load pipeline
        pipeline = Trellis2ImageTo3DPipeline.from_pretrained("serpentine-b/t2")
        pipeline.cuda()
        
        # Warmup with a dummy image
        dummy_image = Image.new('RGB', (512, 512), color='white')
        _ = pipeline.run(
            dummy_image,
            seed=100,
            sparse_structure_sampler_params={"steps": 9},  # Reduced steps for warmup
            shape_slat_sampler_params={"steps": 8},
            tex_slat_sampler_params={"steps": 8}
        )
        
        is_ready = True
        elapsed = time.time() - start_time
        print(f"✅ Model ready! Initialization took {elapsed:.2f}s")
        
    except Exception as e:
        print(f"❌ Failed to initialize model: {e}")
        raise


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if is_ready:
        return JSONResponse(
            content={"status": "ready"},
            status_code=200
        )
    else:
        return JSONResponse(
            content={"status": "initializing"},
            status_code=503
        )


@app.post("/generate")
async def generate_model(prompt_image_file: UploadFile = File(...)):
    """
    Generate 3D Gaussian Splat from input image.
    
    Returns:
        StreamingResponse: PLY file stream
    """
    if not is_ready:
        raise HTTPException(status_code=503, detail="Service not ready")
    
    # Ensure sequential processing
    async with generation_lock:
        start_time = time.time()
        
        try:
            # Read and validate image
            image_data = await prompt_image_file.read()
            image = Image.open(io.BytesIO(image_data)).convert('RGB')
            
            # Resize if too large (for speed)
            max_size = 512
            if max(image.size) > max_size:
                image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
            
            # Generate mesh with optimized parameters for speed
            outputs = pipeline.run(
                image,
                seed=100,
                sparse_structure_sampler_params={"steps": 9},  # Reduced for speed
                shape_slat_sampler_params={"steps": 8},  # Reduced for speed
                tex_slat_sampler_params={"steps": 8}  # Reduced for speed
            )
            mesh_with_voxel = outputs[0]
            
            ply_data = mesh_to_gaussian_splat_ply(mesh_with_voxel, max_points=299999)
            
            generation_time = time.time() - start_time
            ply_size_mb = len(ply_data) / (1024 * 1024)
            
            # Return PLY file as stream
            return StreamingResponse(
                io.BytesIO(ply_data),
                media_type="application/octet-stream",
                headers={
                    "Content-Disposition": f"attachment; filename=model.ply",
                    "X-Generation-Time": str(generation_time),
                    "X-File-Size-MB": str(ply_size_mb)
                }
            )
            
        except Exception as e:
            print(f"❌ Generation failed: {e}")
            import traceback
            traceback.print_exc()
            raise HTTPException(status_code=500, detail=str(e))


@app.get("/")
async def root():
    """Root endpoint with API info"""
    return {
        "name": "Image-to-3D Gaussian Splat Generator",
        "status": "ready" if is_ready else "initializing",
        "endpoints": {
            "/generate": "POST - Generate 3D model from image",
            "/health": "GET - Health check",
        },
        "constraints": {
            "max_generation_time": "30 seconds",
            "max_ply_size": "200 MB",
            "max_image_size": "512x512 (auto-resized)"
        }
    }


if __name__ == "__main__":
    # Run server on 0.0.0.0:10006
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=10006,
        log_level="info"
    )

