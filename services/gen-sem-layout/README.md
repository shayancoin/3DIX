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
