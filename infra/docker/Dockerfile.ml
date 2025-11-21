FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    git \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY services/gen-sem-layout/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy the ML service code
COPY services/gen-sem-layout /app

# Copy research code (SemLayoutDiff)
COPY research/sem-layout-diff /app/research/sem-layout-diff

# Expose port
EXPOSE 8001

ENV PYTHONPATH=/app:/app/research/sem-layout-diff

# Command to run the application
CMD ["python", "-m", "uvicorn", "main:app", "--host", "0.0.0.0", "--port", "8001"]
