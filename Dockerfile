FROM python:3.12-slim

# libgl/libglib are needed by opencv-python
RUN apt-get update && \
    apt-get install -y --no-install-recommends libgl1 libglib2.0-0 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.in .
RUN pip install --no-cache-dir -r requirements.in

COPY . .

# InsightFace models (~300MB) download to this volume on first run
VOLUME /root/.insightface
# Scan cache, face crops, uploads
VOLUME /root/.facesorter

EXPOSE 8501
CMD ["streamlit", "run", "facesorter/app.py", "--server.address=0.0.0.0"]
