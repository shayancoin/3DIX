# 3DIX SAM-3D Object Reconstruction Service

Service for reconstructing 3D objects from images using SAM-3D Objects model.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8002 --reload
```

## API Endpoints

- `GET /health` - Health check
- `POST /reconstruct` - Reconstruct single object from image + mask
- `POST /reconstruct/multi` - Reconstruct multiple objects from image + multiple masks

## Environment Variables

- `PORT` - Service port (default: 8002)
- `HOST` - Service host (default: 0.0.0.0)
- `SAM3D_CHECKPOINT_TAG` - Checkpoint tag (default: "hf")

## Setup

To use real SAM-3D models:

1. Download SAM-3D checkpoints from the [official repository](https://github.com/facebookresearch/sam-3d-objects)
2. Place checkpoints in `research/sam-3d-objects/checkpoints/{tag}/`
3. Install SAM-3D dependencies (see research/sam-3d-objects/requirements.txt)
4. Set `SAM3D_CHECKPOINT_TAG` environment variable
