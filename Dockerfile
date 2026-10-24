FROM python:3.12-slim

# LightGBM needs the OpenMP runtime
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements-serve.txt .
RUN pip install --no-cache-dir -r requirements-serve.txt

COPY sql sql
COPY src src
# Generate data, build SQLite and train the calibrated LightGBM inside the image (seed=42)
RUN python src/build_serving.py

ENV PORT=8000
EXPOSE 8000
CMD ["sh", "-c", "uvicorn app:app --app-dir src --host 0.0.0.0 --port ${PORT}"]
