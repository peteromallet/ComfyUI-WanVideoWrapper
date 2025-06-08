# ComfyUI wrapper nodes for [WanVideo](https://github.com/Wan-Video/Wan2.1)

# WORK IN PROGRESS

# Installation
1. Clone this repo into `custom_nodes` folder.
2. Install dependencies: `pip install -r requirements.txt`
   or if you use the portable install, run this in ComfyUI_windows_portable -folder:

  `python_embeded\python.exe -m pip install -r ComfyUI\custom_nodes\ComfyUI-WanVideoWrapper\requirements.txt`

## Models

https://huggingface.co/Kijai/WanVideo_comfy/tree/main

Text encoders to `ComfyUI/models/text_encoders`

Clip vision to `ComfyUI/models/clip_vision`

Transformer to `ComfyUI/models/diffusion_models`

Vae to `ComfyUI/models/vae`

You can also use the native ComfyUI text encoding and clip vision loader with the wrapper instead of the original models:

![image](https://github.com/user-attachments/assets/6a2fd9a5-8163-4c93-b362-92ef34dbd3a4)

---

Examples:
---

[ReCamMaster](https://github.com/KwaiVGI/ReCamMaster):

https://github.com/user-attachments/assets/c58a12c2-13ba-4af8-8041-e283dbef197e


TeaCache (with the old temporary WIP naive version, I2V):

**Note that with the new version the threshold values should be 10x higher**

Range of 0.25-0.30 seems good when using the coefficients, start step can be 0, with more aggressive threshold values it may make sense to start later to avoid any potential step skips early on, that generally ruin the motion.

https://github.com/user-attachments/assets/504a9a50-3337-43d2-97b8-8e1661f29f46


Context window test:

1025 frames using window size of 81 frames, with 16 overlap. With the 1.3B T2V model this used under 5GB VRAM and took 10 minutes to gen on a 5090:

https://github.com/user-attachments/assets/89b393af-cf1b-49ae-aa29-23e57f65911e

---


This very first test was 512x512x81

~16GB used with 20/40 blocks offloaded

https://github.com/user-attachments/assets/fa6d0a4f-4a4d-4de5-84a4-877cc37b715f

Vid2vid example:


with 14B T2V model:

https://github.com/user-attachments/assets/ef228b8a-a13a-4327-8a1b-1eb343cf00d8

with 1.3B T2V model

https://github.com/user-attachments/assets/4f35ba84-da7a-4d5b-97ee-9641296f391e

# Standalone WanVideo Image-to-Video Runner

This project provides a standalone Python script (`img2vid.py`) to run an image-to-video generation workflow using the ComfyUI WanVideoWrapper, without needing to run the full ComfyUI application.

## Features

- **Standalone Execution**: Runs a complex ComfyUI workflow from a single Python script.
- **Self-Contained**: Clones the necessary ComfyUI source code into `comfy_src` and uses a local `models` directory.
- **Automatic Model Downloader**: On the first run, the script automatically checks for and downloads all required models from Hugging Face, showing a progress bar.

---

## Setup and Installation

Follow these steps to set up the project environment.

### 1. Clone this Repository

```bash
git clone <your-repo-url>
cd <your-repo-name>
```

### 2. Clone ComfyUI Source Code

The script needs access to the core ComfyUI source files. Clone the official repository into a directory named `comfy_src` inside your project folder.

```bash
git clone https://github.com/comfyanonymous/ComfyUI.git comfy_src
```

### 3. Install Python Dependencies

Install all the required Python packages using the provided `requirements.txt` file. It is recommended to do this in a virtual environment.

```bash
pip install -r requirements.txt
```

---

## How to Run

After completing the setup, you can generate a video.

### 1. Place an Input Image

Add a PNG or JPEG image that you want to animate into the `examples/` directory. If the directory doesn't exist, the script will create it for you.

### 2. Execute the Script

Run the `img2vid.py` script from your terminal.

```bash
python img2vid.py
```

**Note:** The first time you run the script, it will download several large model files (totaling over 25 GB). This will take a while depending on your internet connection. Subsequent runs will be much faster as they will use the locally saved models.

### 3. Find Your Video

The output video will be saved in the `output/` directory, named `my_video.mp4`.

---

## Customization

You can easily customize the generation by editing the `if __name__ == '__main__':` block at the bottom of the `img2vid.py` script. Here you can change:
- The input image path.
- The positive and negative text prompts.
- The output video path.



