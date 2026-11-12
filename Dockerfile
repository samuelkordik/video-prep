FROM python:3.11-slim

RUN apt-get update \
 && apt-get install -y --no-install-recommends ffmpeg fonts-dejavu-core \
 && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Model caches live on a bind mount (./models) so they survive rebuilds.
ENV HF_HOME=/models/hf \
    TORCH_HOME=/models/torch \
    VIDEOPREP_MODELS=/models \
    PYTHONUNBUFFERED=1 \
    PYTHONPATH=/app \
    OMP_NUM_THREADS=8

COPY videoprep /app/videoprep
WORKDIR /data
ENTRYPOINT ["python", "-m", "videoprep"]
CMD ["--help"]
