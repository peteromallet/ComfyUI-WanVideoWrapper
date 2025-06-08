import os
import sys

# Add the comfy_src directory to Python's import path BEFORE other imports
script_directory = os.path.dirname(os.path.abspath(__file__))
comfy_src_path = os.path.join(script_directory, 'comfy_src')
sys.path.append(comfy_src_path)

import torch
from PIL import Image
import numpy as np
import imageio
import folder_paths # Import for model paths
import urllib.request
from tqdm import tqdm

# Import the node classes from nodes.py
from nodes import (
    WanVideoModelLoader,
    WanVideoVAELoader,
    LoadWanVideoClipTextEncoder,
    WanVideoClipVisionEncode,
    WanVideoImageToVideoEncode,
    LoadWanVideoT5TextEncoder,
    WanVideoTextEncode,
    WanVideoSampler,
    WanVideoDecode,
    WanVideoBlockSwap,
    WanVideoEnhanceAVideo,
    WanVideoTeaCache,
    WanVideoImageResizeToClosest,
)

def setup_local_model_paths():
    """
    Overrides ComfyUI's model pathing to use a local 'models' directory.
    """
    project_root = os.path.dirname(os.path.abspath(__file__))
    models_dir = os.path.join(project_root, 'models')
    
    # Define model type mappings to their subdirectories
    model_path_map = {
        "diffusion_models": "diffusion_models",
        "vae": "vae",
        "clip_vision": "clip_vision",
        "text_encoders": "text_encoders",
        "loras": "loras",
        "vae_approx": "vae_approx",
    }
    
    # Clear any existing paths and apply our local ones
    folder_paths.folder_names_and_paths = {}
    
    for model_type, sub_dir in model_path_map.items():
        folder_paths.add_model_folder_path(model_type, os.path.join(models_dir, sub_dir))
        
    print(f"Set model paths to local directory: {models_dir}")


def download_model(url, file_path):
    """
    Downloads a file from a URL to a given path with a progress bar.
    """
    class TqdmUpTo(tqdm):
        def update_to(self, b=1, bsize=1, tsize=None):
            if tsize is not None:
                self.total = tsize
            self.update(b * bsize - self.n)

    if os.path.exists(file_path):
        print(f"Model already exists: {os.path.basename(file_path)}")
        return

    print(f"Downloading model: {os.path.basename(file_path)}")
    os.makedirs(os.path.dirname(file_path), exist_ok=True)
    
    try:
        with TqdmUpTo(unit='B', unit_scale=True, miniters=1, desc=os.path.basename(file_path)) as t:
            urllib.request.urlretrieve(url, file_path, reporthook=t.update_to)
        print(f"Successfully downloaded {os.path.basename(file_path)}")
    except Exception as e:
        print(f"Error downloading {url}: {e}")
        if os.path.exists(file_path):
            os.remove(file_path) # Clean up partial download


def img2vid(
    image_path: str,
    positive_prompt: str,
    negative_prompt: str,
    output_path: str = "output.mp4",
):
    """
    This function generates a video from an image based on the provided ComfyUI workflow,
    using the actual node classes from nodes.py.
    """
    # --- Download Models ---
    models_to_download = {
        "diffusion_models/WanVideo/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        "vae/wanvideo/Wan2_1_VAE_bf16.safetensors": "https://huggingface.co/Kijai/WanVideo_comfy/resolve/main/Wan2_1_VAE_bf16.safetensors",
        "clip_vision/clip_vision_h.safetensors": "https://huggingface.co/Comfy-Org/Wan_2.1_ComfyUI_repackaged/resolve/main/split_files/clip_vision/clip_vision_h.safetensors",
        "text_encoders/umt5_xxl_fp16.safetensors": "https://huggingface.co/comfyanonymous/flux_text_encoders/resolve/main/t5xxl_fp16.safetensors"
    }
    
    models_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models')
    for relative_path, url in models_to_download.items():
        download_model(url, os.path.join(models_dir, relative_path))

    # This function needs access to the ComfyUI model paths.
    # We'll initialize it here to use our local models directory.
    setup_local_model_paths()
    
    # Instantiate the node classes
    wvm_loader = WanVideoModelLoader()
    wvv_loader = WanVideoVAELoader()
    lwvcte_loader = LoadWanVideoClipTextEncoder()
    wvcve_encoder = WanVideoClipVisionEncode()
    wvir_resizer = WanVideoImageResizeToClosest()
    wvitv_encoder = WanVideoImageToVideoEncode()
    lwvt5_loader = LoadWanVideoT5TextEncoder()
    wvte_encoder = WanVideoTextEncode()
    wv_sampler = WanVideoSampler()
    wv_decoder = WanVideoDecode()
    
    # --- Replicate the workflow ---

    # 1. Load models
    print("Loading models...")
    # NOTE: The model names are hardcoded based on the workflow.
    # You may need to change these if your file names are different.
    model = wvm_loader.loadmodel(
        model="WanVideo/Wan2_1-I2V-14B-480P_fp8_e4m3fn.safetensors",
        base_precision="fp16",
        quantization="fp8_e4m3fn",
        load_device="offload_device"
    )[0]
    
    vae = wvv_loader.loadmodel(
        model_name="wanvideo/Wan2_1_VAE_bf16.safetensors",
        precision="bf16"
    )[0]
    
    # This corresponds to the CLIP model used for image encoding
    clip_vision = lwvcte_loader.loadmodel(
        model_name="clip_vision_h.safetensors",
        precision="fp16"
    )[0]

    # This corresponds to the T5 model used for text encoding
    t5_text_encoder = lwvt5_loader.loadmodel(
        model_name="umt5_xxl_fp16.safetensors",
        precision="bf16"
    )[0]

    # 2. Process the image
    print("Processing image...")
    # Replicate LoadImage
    with Image.open(image_path) as i:
        i = i.convert("RGB")
        image = np.array(i).astype(np.float32) / 255.0
        image = torch.from_numpy(image)[None,]

    # Replicate ImageResizeKJ using the node from the wrapper
    resized_image, width, height = wvir_resizer.process(
        image=image, generation_width=624, generation_height=624, aspect_ratio_preservation="crop_to_new"
    )

    clip_embeds = wvcve_encoder.process(
        clip_vision=clip_vision,
        image_1=resized_image,
        strength_1=1.0,
        strength_2=1.0,
        crop="center",
        combine_embeds="average",
        force_offload=True,
        ratio=0.2
    )[0]
    
    image_embeds = wvitv_encoder.process(
        vae=vae,
        clip_embeds=clip_embeds,
        start_image=resized_image,
        width=width,
        height=height,
        num_frames=81,
        noise_aug_strength=0.03,
        start_latent_strength=1.0,
        end_latent_strength=1.0,
        force_offload=True,
        fun_or_fl2v_model=False
    )[0]

    # 3. Handle text prompts
    print("Encoding text prompts...")
    text_embeds = wvte_encoder.process(
        t5=t5_text_encoder,
        positive_prompt=positive_prompt,
        negative_prompt=negative_prompt
    )[0]

    # 4. Set up optional arguments for the sampler
    block_swap_args = WanVideoBlockSwap().setargs(blocks_to_swap=10, offload_img_emb=False, offload_txt_emb=False, use_non_blocking=True)[0]
    feta_args = WanVideoEnhanceAVideo().setargs(weight=2.0, start_percent=0.0, end_percent=1.0)[0]
    teacache_args = WanVideoTeaCache().process(rel_l1_thresh=0.25, start_step=1, end_step=-1, cache_device="offload_device", use_coefficients=True, mode="e")[0]

    # 5. Run the sampler
    print("Sampling...")
    latent_samples = wv_sampler.process(
        model=model,
        text_embeds=text_embeds,
        image_embeds=image_embeds,
        steps=25,
        cfg=6.0,
        shift=5.0,
        seed=1057359483639287,
        force_offload=True,
        scheduler="unipc",
        riflex_freq_index=0,
        feta_args=feta_args,
        teacache_args=teacache_args,
        rope_function="comfy"
    )[0]

    # 6. Decode the video
    print("Decoding video...")
    decoded_images = wv_decoder.decode(
        vae=vae,
        samples=latent_samples,
        enable_vae_tiling=False,
        tile_x=272,
        tile_y=272,
        tile_stride_x=144,
        tile_stride_y=128
    )[0]

    # 7. Combine frames into a video
    print(f"Saving video to {output_path}...")
    # Ensure output directory exists
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    # Convert to uint8 for video saving
    video_frames = (decoded_images.numpy() * 255).astype(np.uint8)
    # Save video using imageio
    imageio.mimsave(output_path, video_frames, fps=16)
    
    print(f"Video generation complete.")


if __name__ == '__main__':
    # This is an example of how to run the function.
    # You will need to ensure that:
    # 1. You have an image at the specified `image_path`.
    # 2. Your ComfyUI `models` directory is structured correctly so that
    #    the script can find the models (e.g., `ComfyUI/models/diffusion_models/WanVideo/...`)
    #    The script assumes it is located in a subfolder of your ComfyUI-WanVideoWrapper project,
    #    and that the ComfyUI directory is a sibling to it.
    #    The script now looks for a 'models' folder in the same directory as the script,
    #    and will download them if they are missing.
    
    # Create an 'examples' and 'output' directory if they don't exist
    if not os.path.exists("examples"):
        os.makedirs("examples")
        print("Created 'examples' directory. Please place an image inside it to run the script.")
    if not os.path.exists("output"):
        os.makedirs("output")

    example_image_path = "examples/example.png"
    if not os.path.exists(example_image_path):
        print(f"Example image not found at '{example_image_path}'. Please add an image to run this example.")
    else:
        img2vid(
            image_path=example_image_path,
            positive_prompt="an old man is stroking his beard thoughtfully",
            negative_prompt="色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走",
            output_path="output/my_video.mp4"
        )