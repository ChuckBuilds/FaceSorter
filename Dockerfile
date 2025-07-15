# Stage 1: The Builder
# This stage installs dependencies, including compiling dlib.
FROM python:3.11-slim AS builder

# Install system dependencies required for dlib and opencv
RUN apt-get update && apt-get install -y \
    build-essential \
    cmake \
    libopenblas-dev \
    liblapack-dev \
    libjpeg-dev \
    libx11-dev \
    libgtk-3-dev \
    pkg-config \
    git \
    && rm -rf /var/lib/apt/lists/*

# Set up a virtual environment in the builder stage
ENV VIRTUAL_ENV=/opt/venv
RUN python3 -m venv $VIRTUAL_ENV
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy and install requirements
# This is done in a separate layer to leverage Docker's cache
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt


# Stage 2: The Final Image
# This stage creates the final, lean image.
FROM python:3.11-slim

# Install all identified runtime libraries required by the dependencies
RUN apt-get update && apt-get install -y \
    libopenblas0 \
    libx11-6 \
    libpng16-16 \
    libjpeg62-turbo \
    libwebp7 \
    libtiff6 \
    libopenjp2-7 \
    libgl1-mesa-glx \
    libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

# Set the working directory
WORKDIR /app

# Copy the virtual environment with all the installed packages from the builder stage
COPY --from=builder /opt/venv /opt/venv

# Activate the virtual environment for subsequent commands
ENV VIRTUAL_ENV=/opt/venv
ENV PATH="$VIRTUAL_ENV/bin:$PATH"

# Copy the application source code
COPY . .

# Expose the port Streamlit runs on
EXPOSE 8501

# Set the entrypoint to run the Streamlit app
# Using `CMD` in this format allows for passing additional arguments to streamlit run
CMD ["streamlit", "run", "facesorter/app.py", "--server.port=8501", "--server.address=0.0.0.0"] 