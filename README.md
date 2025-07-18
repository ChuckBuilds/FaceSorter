# FaceSorter

FaceSorter is an application that automatically organizes photos and videos by detecting and clustering faces, then distributing media files to appropriate folders for easy sharing with guests.

## Prerequisites (for Windows)

Before installing the Python packages, you will need to install the following tools:

1.  **Microsoft C++ Build Tools**:
    -   Download the "Build Tools for Visual Studio" from the [Visual Studio downloads page](https://visualstudio.microsoft.com/downloads/#build-tools-for-visual-studio-2022).
    -   During installation, select the "Desktop development with C++" workload.

2.  **CMake**:
    -   Download and install CMake from the [official website](https://cmake.org/download/).
    -   **Important**: During installation, make sure to select the option to "Add CMake to the system PATH for all users".

## Installation

After installing the prerequisites, create and activate a virtual environment:

```bash
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Then, install the required Python packages:

```bash
pip install -r requirements.txt
```

## Enabling GPU Acceleration (Optional, for NVIDIA GPUs)

For a significant performance increase in face detection, you can enable GPU acceleration if you have a compatible NVIDIA GPU. This requires `dlib` to be compiled from source with CUDA support.

### 1. Install NVIDIA CUDA Toolkit

- **Download**: Go to the [NVIDIA CUDA Toolkit Archive](https://developer.nvidia.com/cuda-toolkit-archive) to download a version compatible with your GPU drivers. We have tested this with version 12.x.
- **Install**: Run the installer and follow the on-screen instructions.

### 2. Install NVIDIA cuDNN

The `dlib` library requires the NVIDIA CUDA Deep Neural Network library (cuDNN) for its CNN-based models.

- **Download**: Go to the [NVIDIA cuDNN Archive](https://developer.nvidia.com/rdp/cudnn-archive). You will need a free NVIDIA Developer account. Download the version that matches your CUDA Toolkit version (e.g., for CUDA 12.x). Choose the "Local Installer for Windows (Zip)" package.
- **Install**:
    1. Unzip the downloaded file. You will find three folders: `bin`, `include`, and `lib`.
    2. Copy the contents of these folders into the corresponding folders in your CUDA Toolkit installation directory (e.g., `C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9`).

### 3. Re-install dlib and face-recognition

To force the libraries to build with CUDA support, they must be re-installed.

1.  **Uninstall existing versions**:
    ```bash
    pip uninstall dlib face-recognition -y
    ```
2.  **Install from source**:
    ```bash
    pip install dlib --no-cache-dir
    pip install face-recognition
    ```
    This process will compile `dlib` from scratch, which may take several minutes. The build script should automatically detect your CUDA and cuDNN installations.

### 4. Verify the Installation

Run the provided test script to confirm that `dlib` can detect your GPU:

```bash
python check_gpu.py
```

A successful output will look like this:

```
Is dlib compiled with CUDA support? True
Number of CUDA devices found: 1

Success! Your dlib installation can see the GPU.
The 'cnn' model should now be running on your NVIDIA GPU.
```

## Understanding the Settings

To get the best results, it's helpful to understand what the key settings in the sidebar do.

### Face Detection Model
This setting determines the core algorithm used to find faces in your photos.
- **`hog` (Histogram of Oriented Gradients):** This is the faster, CPU-based model. It works well for clear, well-lit, mostly front-facing photos. It is less accurate for faces at odd angles (e.g., profiles).
- **`cnn` (Convolutional Neural Network):** This is a much more accurate deep learning model that excels at finding faces in a wide variety of conditions, including different angles, lighting, and obstructions. It is significantly slower and requires a properly configured NVIDIA GPU to be effective.

### Clustering Sensitivity (`eps`)
This is the **most important setting** for ensuring people are grouped correctly. It controls how similar two face "fingerprints" must be to be automatically placed in the same folder.

- **Analogy: The "Huddle" Rule.** Think of `eps` as the maximum distance someone can be from a group at a party and still be considered part of that "huddle."
- A **low `eps` value (e.g., 0.40)** is a *strict* rule. It's like saying, "You must be shoulder-to-shoulder to be in the same group." This creates many small, tight, well-defined groups and is best for the highly consistent `cnn` model.
- A **high `eps` value (e.g., 0.60)** is a *loose* rule. It's like saying, "If you're in the same half of the room, you're in the same group." This is better for the less consistent `hog` model, but if set too high, it can cause unrelated people to be lumped into the same folder.

The application will suggest a smart default based on the detection model you choose, but you may need to adjust it:
- If you find the **same person is being split** into multiple different folders, their photos are slightly too far apart. **Increase** the `eps` value (e.g., from 0.40 to 0.45).
- If you find **different people are being grouped together** in the same folder, the rule is too loose. **Decrease** the `eps` value (e.g., from 0.45 to 0.40).

*Note: There is no longer a separate setting for "Merge Suggestions." The application may still offer to merge folders after the initial clustering is complete if the average "faceprints" of two folders are extremely similar. This is an automatic secondary check.*

### Parallel Workers
This slider controls how many CPU processes are used to prepare images (loading from disk, resizing) to be fed to the face detection model. It helps create an efficient pipeline to keep the GPU or CPU busy.
- For **GPU processing (`cnn`)**, the optimal number is usually a small value (2-4). This is just enough to prepare the next image while the GPU is working on the current one, ensuring the GPU is never idle.
- For **CPU processing (`hog`)**, you can set this to the number of available CPU cores for maximum throughput.

## Usage

To run the web interface:

```
