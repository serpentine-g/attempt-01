# Image-to-3D Gaussian Splat Mining Solution

Competition submission for subnet architecture Image-to-3D generation.

## Quick Start

### Build Docker Image
```bash
docker build -t image-to-3d-miner -f docker/Dockerfile .
```

### Run Container
```bash
docker run --gpus all -p 10006:10006 image-to-3d-miner
```

### Test the API

#### Health Check
```bash
curl http://localhost:10006/health
```

#### Generate 3D Model
```bash
curl -X POST http://localhost:10006/generate \
  -F "prompt_image_file=@path/to/your/image.jpg" \
  --output model.ply
```

## API Endpoints

### `POST /generate`
Generate 3D Gaussian Splat from image.

**Input:**
- `prompt_image_file`: Image file (multipart/form-data)

**Output:**
- PLY file stream (application/octet-stream)

**Headers:**
- `X-Generation-Time`: Time taken to generate (seconds)
- `X-File-Size-MB`: Size of generated PLY file (MB)

### `GET /health`
Health check endpoint.

**Returns:**
```json
{"status": "ready"}
```

### `GET /`
API information and status.

## Performance Specifications

- **Docker Build Time:** < 3 hours
- **Container Startup + Warmup:** < 1 hour
- **PLY File Size:** < 200 MB
- **Generation Time:** < 30 seconds per request

## Architecture

This solution uses:
- **TRELLIS.2-4B** model for image-to-3D generation
- **FastAPI** for HTTP server
- **Optimized sampling** (10 steps) for speed
- **Surface sampling** for Gaussian Splat PLY conversion
- **Sequential request handling** with async locks

## Model Details

The model is automatically downloaded from HuggingFace Hub on first startup:
- `tao-hunter/TRELLIS.2-4B`

## Requirements

- NVIDIA GPU with CUDA 12.4+ support
- Docker with NVIDIA Container Runtime
- At least 16GB GPU memory recommended

## Development

### Local Testing Without Docker
```bash
# Install dependencies
bash setup.sh --new-env --basic --flash-attn --cumesh --o-voxel --flexgemm --nvdiffrast --nvdiffrec

# Run server
python mining_server.py
```

### Test with Python
```python
import requests

# Health check
response = requests.get("http://localhost:10006/health")
print(response.json())

# Generate model
with open("test_image.jpg", "rb") as f:
    files = {"prompt_image_file": f}
    response = requests.post("http://localhost:10006/generate", files=files)
    
with open("output.ply", "wb") as f:
    f.write(response.content)
```

## Troubleshooting

### Container fails to start
- Check GPU availability: `nvidia-smi`
- Check Docker GPU support: `docker run --rm --gpus all nvidia/cuda:12.4.1-base-ubuntu22.04 nvidia-smi`

### Generation takes too long
- Image is automatically resized to 512x512
- Sampling steps are optimized to 10 (can be reduced further)
- Consider using smaller model variant if available

### Out of memory
- Reduce `max_points` in `mesh_to_gaussian_splat_ply()`
- Reduce image resolution
- Clear CUDA cache between requests

## License

See LICENSE file in repository root.

