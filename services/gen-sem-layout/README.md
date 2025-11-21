# 3DIX Layout Generation Service

ML microservice for generating semantic room layouts using SemLayoutDiff.

## Development

```bash
# Install dependencies
pip install -r requirements.txt

# Run the service
python main.py
# Or with uvicorn
uvicorn main:app --host 0.0.0.0 --port 8001 --reload
```

## API Endpoints

- `GET /health` - Health check
- `POST /generate` - Generate layout from vibe specification

## Environment Variables

- `PORT` - Service port (default: 8001)
- `HOST` - Service host (default: 0.0.0.0)
- `SLDN_CHECKPOINT_PATH` - Path to SLDN checkpoint
- `APM_CHECKPOINT_PATH` - Path to APM checkpoint
- `SEMLAYOUTDIFF_CONFIG_PATH` - Path to SemLayoutDiff config files
- `THREED_FUTURE_DATASET_PATH` - Path to 3D-FUTURE dataset JSON file
- `THREED_FUTURE_MODEL_INFO_PATH` - Path to 3D-FUTURE model_info.json
- `ASSET_BASE_URL` - Base URL for serving asset files (default: http://localhost:8001/assets)

## Asset Library Setup

To use real 3D-FUTURE assets:

1. Download 3D-FUTURE dataset from [official website](https://tianchi.aliyun.com/specials/promotion/alibaba-3d-future)
2. Extract the dataset
3. Generate dataset JSON file using the preprocessing scripts in `research/sem-layout-diff`:
   ```bash
   cd research/sem-layout-diff
   python preprocess/scripts/json_threed_future_dataset.py \
     --path_to_3d_future_dataset_directory /path/to/3D-FUTURE \
     --path_to_3d_future_model_info /path/to/model_info.json \
     --output_path /path/to/output/threed_future_model_unified.json
   ```
4. Convert OBJ files to glTF format (optional, for web compatibility):
   - Use tools like [obj2gltf](https://github.com/CesiumGS/obj2gltf) or [Blender](https://www.blender.org/)
   - Organize assets by model_id in a directory structure
5. Set environment variables:
   - `THREED_FUTURE_DATASET_PATH`: Path to the JSON dataset file
   - `THREED_FUTURE_MODEL_INFO_PATH`: Path to model_info.json
   - `ASSET_BASE_URL`: URL where asset files are served (e.g., CDN or storage service URL)

## Asset Quality Levels

The service supports multiple quality levels for assets:
- `high`: Full quality glTF models with textures
- `medium`: Medium quality (reduced polygon count)
- `low`: Low quality (minimal polygons, no textures)
