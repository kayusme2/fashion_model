import os
import re
from PIL import Image
import cv2
import mediapipe as mp
from rembg import remove
from diffusers import StableDiffusionControlNetPipeline, ControlNetModel, UniPCMultistepScheduler
import torch
import numpy as np
from huggingface_hub import configure_http_backend, HfFolder
from huggingface_hub.utils import HfHubHTTPError
from fastapi import FastAPI, File, UploadFile, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
import urllib3
import io
import tempfile

# Configure HTTP backend with retries for network timeouts
def http_backend_factory():
    return urllib3.PoolManager(retries=urllib3.Retry(total=5, backoff_factor=1))

configure_http_backend(http_backend_factory)

# Initialize FastAPI app
app = FastAPI()

# Add CORS middleware to allow frontend requests
app.add_middleware(
    CORSMiddleware,
    allow_origins=["https://fashion-model-front.onrender.com", "http://localhost:8000"],  # Replace with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Check HF_TOKEN
HF_TOKEN = os.getenv("HF_TOKEN", "YOUR_HUGGING_FACE_TOKEN_HERE")
if not HF_TOKEN or HF_TOKEN == "YOUR_HUGGING_FACE_TOKEN_HERE":
    raise ValueError("Hugging Face token must be set via HF_TOKEN environment variable.")

# Define OpenPose skeleton connections (based on OpenPose body_25 model)
POSE_CONNECTIONS = [
    (0, 1), (1, 2), (2, 3), (3, 4),  # Head to right arm
    (1, 5), (5, 6), (6, 7),  # Head to left arm
    (1, 8), (8, 9), (9, 10),  # Torso to right leg
    (8, 11), (11, 12), (12, 13),  # Torso to left leg
    (1, 14), (14, 15), (15, 16),  # Head to right hand
    (1, 17), (17, 18), (18, 19),  # Head to left hand
]

# Define keypoint coordinates for each pose (normalized 0-1, for a 512x512 image)
POSE_KEYPOINTS = {
    "standing_neutral": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.5),  # 3: Right elbow
        (0.8, 0.7),  # 4: Right wrist
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.5),  # 6: Left elbow
        (0.2, 0.7),  # 7: Left wrist
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.8, 0.7),  # 15: Right hand (simplified)
        (0.8, 0.7),  # 16: Right fingers
        (0.2, 0.7),  # 17: Left hand
        (0.2, 0.7),  # 18: Left fingers
        (0.2, 0.7),  # 19: Left fingers (placeholder)
    ],
    "standing_arms_raised": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.2),  # 2: Right shoulder
        (0.7, 0.1),  # 3: Right elbow
        (0.8, 0.0),  # 4: Right wrist
        (0.4, 0.2),  # 5: Left shoulder
        (0.3, 0.1),  # 6: Left elbow
        (0.2, 0.0),  # 7: Left wrist
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.8, 0.0),  # 15: Right hand
        (0.8, 0.0),  # 16: Right fingers
        (0.2, 0.0),  # 17: Left hand
        (0.2, 0.0),  # 18: Left fingers
        (0.2, 0.0),  # 19: Left fingers
    ],
    "arms_crossed": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.5, 0.4),  # 3: Right elbow (crossed over chest)
        (0.4, 0.5),  # 4: Right wrist (near left side)
        (0.4, 0.3),  # 5: Left shoulder
        (0.5, 0.4),  # 6: Left elbow (crossed over chest)
        (0.6, 0.5),  # 7: Left wrist (near right side)
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.4, 0.5),  # 15: Right hand
        (0.4, 0.5),  # 16: Right fingers
        (0.6, 0.5),  # 17: Left hand
        (0.6, 0.5),  # 18: Left fingers
        (0.6, 0.5),  # 19: Left fingers
    ],
    "standing_holding_right": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.4),  # 3: Right elbow (bent, holding)
        (0.6, 0.5),  # 4: Right wrist (near torso)
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.5),  # 6: Left elbow
        (0.2, 0.7),  # 7: Left wrist
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.6, 0.5),  # 15: Right hand
        (0.6, 0.5),  # 16: Right fingers
        (0.2, 0.7),  # 17: Left hand
        (0.2, 0.7),  # 18: Left fingers
        (0.2, 0.7),  # 19: Left fingers
    ],
    "standing_holding_left": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.5),  # 3: Right elbow
        (0.8, 0.7),  # 4: Right wrist
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.4),  # 6: Left elbow (bent, holding)
        (0.4, 0.5),  # 7: Left wrist (near torso)
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.8, 0.7),  # 15: Right hand
        (0.8, 0.7),  # 16: Right fingers
        (0.4, 0.5),  # 17: Left hand
        (0.4, 0.5),  # 18: Left fingers
        (0.4, 0.5),  # 19: Left fingers
    ],
    "standing_holding_both": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.6, 0.4),  # 3: Right elbow (bent, holding)
        (0.5, 0.5),  # 4: Right wrist (near center torso)
        (0.4, 0.3),  # 5: Left shoulder
        (0.4, 0.4),  # 6: Left elbow (bent, holding)
        (0.5, 0.5),  # 7: Left wrist (near center torso)
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.5, 0.5),  # 15: Right hand
        (0.5, 0.5),  # 16: Right fingers
        (0.5, 0.5),  # 17: Left hand
        (0.5, 0.5),  # 18: Left fingers
        (0.5, 0.5),  # 19: Left fingers
    ],
    "sitting_neutral": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.5),  # 3: Right elbow
        (0.8, 0.7),  # 4: Right wrist
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.5),  # 6: Left elbow
        (0.2, 0.7),  # 7: Left wrist
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.5),  # 9: Right hip
        (0.6, 0.6),  # 10: Right knee (bent, sitting)
        (0.6, 0.7),  # 11: Right ankle
        (0.4, 0.5),  # 12: Left hip
        (0.4, 0.6),  # 13: Left knee (bent, sitting)
        (0.4, 0.7),  # 14: Left ankle
        (0.8, 0.7),  # 15: Right hand
        (0.8, 0.7),  # 16: Right fingers
        (0.2, 0.7),  # 17: Left hand
        (0.2, 0.7),  # 18: Left fingers
        (0.2, 0.7),  # 19: Left fingers
    ],
    "sitting_holding_right": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.4),  # 3: Right elbow (bent, holding)
        (0.6, 0.5),  # 4: Right wrist (near torso)
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.5),  # 6: Left elbow
        (0.2, 0.7),  # 7: Left wrist
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.5),  # 9: Right hip
        (0.6, 0.6),  # 10: Right knee (bent, sitting)
        (0.6, 0.7),  # 11: Right ankle
        (0.4, 0.5),  # 12: Left hip
        (0.4, 0.6),  # 13: Left knee (bent, sitting)
        (0.4, 0.7),  # 14: Left ankle
        (0.6, 0.5),  # 15: Right hand
        (0.6, 0.5),  # 16: Right fingers
        (0.2, 0.7),  # 17: Left hand
        (0.2, 0.7),  # 18: Left fingers
        (0.2, 0.7),  # 19: Left fingers
    ],
    "sitting_holding_left": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.5),  # 3: Right elbow
        (0.8, 0.7),  # 4: Right wrist
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.4),  # 6: Left elbow (bent, holding)
        (0.4, 0.5),  # 7: Left wrist (near torso)
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.5),  # 9: Right hip
        (0.6, 0.6),  # 10: Right knee (bent, sitting)
        (0.6, 0.7),  # 11: Right ankle
        (0.4, 0.5),  # 12: Left hip
        (0.4, 0.6),  # 13: Left knee (bent, sitting)
        (0.4, 0.7),  # 14: Left ankle
        (0.8, 0.7),  # 15: Right hand
        (0.8, 0.7),  # 16: Right fingers
        (0.4, 0.5),  # 17: Left hand
        (0.4, 0.5),  # 18: Left fingers
        (0.4, 0.5),  # 19: Left fingers
    ],
    "walking_right_leg_forward": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.5),  # 3: Right elbow
        (0.8, 0.7),  # 4: Right wrist
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.5),  # 6: Left elbow
        (0.2, 0.7),  # 7: Left wrist
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.7),  # 10: Right knee (forward)
        (0.6, 0.8),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee (back)
        (0.4, 0.9),  # 14: Left ankle
        (0.8, 0.7),  # 15: Right hand
        (0.8, 0.7),  # 16: Right fingers
        (0.2, 0.7),  # 17: Left hand
        (0.2, 0.7),  # 18: Left fingers
        (0.2, 0.7),  # 19: Left fingers
    ],
    "walking_left_leg_forward": [
        (0.5, 0.1),  # 0: Head
        (0.5, 0.2),  # 1: Neck
        (0.6, 0.3),  # 2: Right shoulder
        (0.7, 0.5),  # 3: Right elbow
        (0.8, 0.7),  # 4: Right wrist
        (0.4, 0.3),  # 5: Left shoulder
        (0.3, 0.5),  # 6: Left elbow
        (0.2, 0.7),  # 7: Left wrist
        (0.5, 0.4),  # 8: Mid torso
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee (back)
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.7),  # 13: Left knee (forward)
        (0.4, 0.8),  # 14: Left ankle
        (0.8, 0.7),  # 15: Right hand
        (0.8, 0.7),  # 16: Right fingers
        (0.2, 0.7),  # 17: Left hand
        (0.2, 0.7),  # 18: Left fingers
        (0.2, 0.7),  # 19: Left fingers
    ],
    "leaning_forward": [
        (0.5, 0.2),  # 0: Head (forward)
        (0.5, 0.3),  # 1: Neck
        (0.6, 0.4),  # 2: Right shoulder
        (0.7, 0.6),  # 3: Right elbow
        (0.8, 0.8),  # 4: Right wrist
        (0.4, 0.4),  # 5: Left shoulder
        (0.3, 0.6),  # 6: Left elbow
        (0.2, 0.8),  # 7: Left wrist
        (0.5, 0.5),  # 8: Mid torso (bent forward)
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.8, 0.8),  # 15: Right hand
        (0.8, 0.8),  # 16: Right fingers
        (0.2, 0.8),  # 17: Left hand
        (0.2, 0.8),  # 18: Left fingers
        (0.2, 0.8),  # 19: Left fingers
    ],
    "leaning_backward": [
        (0.5, 0.0),  # 0: Head (backward)
        (0.5, 0.1),  # 1: Neck
        (0.6, 0.2),  # 2: Right shoulder
        (0.7, 0.4),  # 3: Right elbow
        (0.8, 0.6),  # 4: Right wrist
        (0.4, 0.2),  # 5: Left shoulder
        (0.3, 0.4),  # 6: Left elbow
        (0.2, 0.6),  # 7: Left wrist
        (0.5, 0.3),  # 8: Mid torso (bent backward)
        (0.6, 0.6),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.6),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee
        (0.4, 0.9),  # 14: Left ankle
        (0.8, 0.6),  # 15: Right hand
        (0.8, 0.6),  # 16: Right fingers
        (0.2, 0.6),  # 17: Left hand
        (0.2, 0.6),  # 18: Left fingers
        (0.2, 0.6),  # 19: Left fingers
    ],
    "crouching": [
        (0.5, 0.3),  # 0: Head
        (0.5, 0.4),  # 1: Neck
        (0.6, 0.5),  # 2: Right shoulder
        (0.7, 0.6),  # 3: Right elbow
        (0.8, 0.7),  # 4: Right wrist
        (0.4, 0.5),  # 5: Left shoulder
        (0.3, 0.6),  # 6: Left elbow
        (0.2, 0.7),  # 7: Left wrist
        (0.5, 0.6),  # 8: Mid torso
        (0.6, 0.7),  # 9: Right hip
        (0.6, 0.8),  # 10: Right knee (bent, crouching)
        (0.6, 0.9),  # 11: Right ankle
        (0.4, 0.7),  # 12: Left hip
        (0.4, 0.8),  # 13: Left knee (bent, crouching)
        (0.4, 0.9),  # 14: Left ankle
        (0.8, 0.7),  # 15: Right hand
        (0.8, 0.7),  # 16: Right fingers
        (0.2, 0.7),  # 17: Left hand
        (0.2, 0.7),  # 18: Left fingers
        (0.2, 0.7),  # 19: Left fingers
    ],
    "turning_sideways": [
        (0.6, 0.1),  # 0: Head (turned right)
        (0.6, 0.2),  # 1: Neck
        (0.7, 0.3),  # 2: Right shoulder (visible)
        (0.8, 0.5),  # 3: Right elbow
        (0.9, 0.7),  # 4: Right wrist
        (0.5, 0.3),  # 5: Left shoulder (partially visible)
        (0.4, 0.5),  # 6: Left elbow
        (0.3, 0.7),  # 7: Left wrist
        (0.6, 0.4),  # 8: Mid torso
        (0.7, 0.6),  # 9: Right hip
        (0.7, 0.8),  # 10: Right knee
        (0.7, 0.9),  # 11: Right ankle
        (0.5, 0.6),  # 12: Left hip
        (0.5, 0.8),  # 13: Left knee
        (0.5, 0.9),  # 14: Left ankle
        (0.9, 0.7),  # 15: Right hand
        (0.9, 0.7),  # 16: Right fingers
        (0.3, 0.7),  # 17: Left hand
        (0.3, 0.7),  # 18: Left fingers
        (0.3, 0.7),  # 19: Left fingers
    ],
}

def generate_pose_image(pose_name, image_size=(512, 512)):
    """Generate an OpenPose-compatible skeleton image for the given pose."""
    if pose_name not in POSE_KEYPOINTS:
        raise ValueError(f"Pose '{pose_name}' not supported.")

    keypoints = POSE_KEYPOINTS[pose_name]
    img = np.zeros((image_size[0], image_size[1], 3), dtype=np.uint8)

    # Scale keypoints to image size
    scaled_keypoints = [(int(x * image_size[1]), int(y * image_size[0])) for x, y in keypoints]

    # Draw keypoints
    for x, y in scaled_keypoints:
        cv2.circle(img, (x, y), 5, (0, 255, 0), -1)

    # Draw connections
    for start, end in POSE_CONNECTIONS:
        x1, y1 = scaled_keypoints[start]
        x2, y2 = scaled_keypoints[end]
        cv2.line(img, (x1, y1), (x2, y2), (255, 0, 0), 2)

    # Convert to PIL Image for compatibility with diffusers
    img_pil = Image.fromarray(img)
    return img_pil

def parse_prompt(prompt):
    """Parse the text prompt to extract item type, pose, side/hand, and background."""
    prompt = prompt.lower().strip()
    
    item_types = {
        "belt": "belt", "glasses": "glasses", "trousers": "trousers", "pants": "trousers",
        "hat": "hat", "pencil": "pencil", "gadget": "gadget", "wristwatch": "wristwatch",
        "watch": "wristwatch", "gloves": "gloves", "shoes": "shoes", "product": "product"
    }
    item_type = None
    for key, value in item_types.items():
        if key in prompt:
            item_type = value
            break
    if not item_type:
        item_type = "product"
    
    poses = {
        "standing neutral": "standing_neutral",
        "standing arms raised": "standing_arms_raised",
        "standing holding right": "standing_holding_right",
        "standing holding left": "standing_holding_left",
        "standing holding both": "standing_holding_both",
        "sitting neutral": "sitting_neutral",
        "sitting holding right": "sitting_holding_right",
        "sitting holding left": "sitting_holding_left",
        "walking right leg forward": "walking_right_leg_forward",
        "walking left leg forward": "walking_left_leg_forward",
        "leaning forward": "leaning_forward",
        "leaning backward": "leaning_backward",
        "crouching": "crouching",
        "turning sideways": "turning_sideways",
        "arms crossed": "arms_crossed",
        "standing": "standing_neutral",
        "sitting": "sitting_neutral",
        "walking": "walking_right_leg_forward",
        "leaning": "leaning_forward",
        "crouching": "crouching",
        "turning": "turning_sideways"
    }
    pose = "standing_neutral"
    for key, value in poses.items():
        if key in prompt:
            pose = value
            break
    
    side = "right"
    if "left hand" in prompt or "left wrist" in prompt or "left foot" in prompt:
        side = "left"
    elif "right hand" in prompt or "right wrist" in prompt or "right foot" in prompt:
        side = "right"
    
    background_keywords = list(item_types.keys()) + list(poses.keys()) + ["left hand", "right hand", "left wrist", "right wrist", "left foot", "right foot"]
    background = prompt
    for keyword in background_keywords:
        background = background.replace(keyword, "")
    background = re.sub(r'\s+', ' ', background).strip()
    if not background:
        background = "in a studio"
    
    return {
        "item_type": item_type,
        "pose": pose,
        "side": side,
        "background": background
    }

def remove_background(image_data):
    """Remove background from the uploaded image."""
    try:
        input_image = Image.open(io.BytesIO(image_data)).convert("RGBA")
        output_image = remove(input_image)
        return output_image
    except Exception as e:
        raise ValueError(f"Background removal failed: {str(e)}")

def generate_model_image(pose, background_desc):
    """Generate a model image with Stable Diffusion and ControlNet."""
    try:
        os.environ["HUGGING_FACE_HUB_TOKEN"] = HF_TOKEN
        HfFolder.save_token(HF_TOKEN)
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cpu":
            print("Warning: Running on CPU, which may be slow.")
        
        controlnet = ControlNetModel.from_pretrained(
            "lllyasviel/control_v11p_sd15_openpose",
            torch_dtype=torch.float16,  # Use float16 to reduce memory usage
            low_cpu_mem_usage=True
        )
        pipeline = StableDiffusionControlNetPipeline.from_pretrained(
            "stabilityai/stable-diffusion-2-1-base",
            controlnet=controlnet,
            torch_dtype=torch.float16,  # Use float16 to reduce memory usage
            low_cpu_mem_usage=True
        )
        pipeline.scheduler = UniPCMultistepScheduler.from_config(pipeline.scheduler.config)
        pipeline.safety_checker = None  # Disable safety checker to save memory
        
        if device == "cuda":
            pipeline.to(device)
            torch.cuda.empty_cache()
        else:
            pipeline.to("cpu")
        
        # Generate the pose image dynamically
        openpose_image = generate_pose_image(pose)
        prompt = f"A professional model {pose.replace('_', ' ')}, {background_desc}, high quality, realistic, neutral clothing"
        image = pipeline(prompt, image=openpose_image, num_inference_steps=10, guidance_scale=7.5).images[0]
        
        output_path = os.path.join(tempfile.gettempdir(), f"model_{pose}_{background_desc.replace(' ', '_')}.png")
        image.save(output_path)
        return output_path
    except HfHubHTTPError as e:
        raise RuntimeError(f"Failed to download models due to network error: {str(e)}")
    except Exception as e:
        raise RuntimeError(f"Model generation failed: {str(e)}")

def detect_body_parts(image_path):
    """Detect face, hands, and pose landmarks using MediaPipe."""
    try:
        mp_face_mesh = mp.solutions.face_mesh
        mp_hands = mp.solutions.hands
        mp_pose = mp.solutions.pose
        
        face_mesh = mp_face_mesh.FaceMesh(static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5)
        hands = mp_hands.Hands(static_image_mode=True, max_num_hands=2, min_detection_confidence=0.5)
        pose = mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5)
        
        image = cv2.imread(image_path)
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        face_results = face_mesh.process(rgb_image)
        hand_results = hands.process(rgb_image)
        pose_results = pose.process(rgb_image)
        
        face_mesh.close()
        hands.close()
        pose.close()
        
        return {
            "face": face_results.multi_face_landmarks,
            "hands": hand_results.multi_hand_landmarks,
            "pose": pose_results.pose_landmarks
        }
    except Exception as e:
        raise RuntimeError(f"Body part detection failed: {str(e)}")

def calculate_face_position(face_landmarks, image_width, image_height):
    """Calculate position and orientation for glasses and hat."""
    if not face_landmarks:
        return None
    landmarks = face_landmarks[0].landmark
    nose_tip = landmarks[1]
    left_eye = landmarks[33]
    right_eye = landmarks[263]
    
    face_x, face_y = int(nose_tip.x * image_width), int(nose_tip.y * image_height)
    eye_distance = np.sqrt((right_eye.x - left_eye.x)**2 + (right_eye.y - left_eye.y)**2) * image_width
    face_size = eye_distance * 2
    angle = np.degrees(np.arctan2(right_eye.y - left_eye.y, right_eye.x - left_eye.x))
    return (face_x, face_y), face_size, angle

def calculate_wrist_position(hand_landmarks, image_width, image_height, side="right"):
    """Calculate wrist position and orientation."""
    if not hand_landmarks:
        return None
    for hand in hand_landmarks:
        handedness = "right" if hand.classification[0].label == "Right" else "left"
        if handedness == side:
            wrist = hand.landmark[0]
            forearm = hand.landmark[5]
            wrist_x, wrist_y = int(wrist.x * image_width), int(wrist.y * image_height)
            forearm_x, forearm_y = int(forearm.x * image_width), int(forearm.y * image_height)
            wrist_size = np.sqrt((forearm_x - wrist_x)**2 + (forearm_y - wrist_y)**2)
            angle = np.degrees(np.arctan2(forearm_y - wrist_y, forearm_x - wrist_x))
            return (wrist_x, wrist_y), wrist_size, angle
    return None

def calculate_lower_body_position(pose_landmarks, image_width, image_height, side="right"):
    """Calculate position for trousers and shoes."""
    if not pose_landmarks:
        return None
    landmarks = pose_landmarks.landmark
    hip = landmarks[23 if side == "right" else 24]
    ankle = landmarks[27 if side == "right" else 28]
    
    leg_x = int((hip.x + ankle.x) * image_width / 2)
    leg_y = int((hip.y + ankle.y) * image_height / 2)
    leg_size = np.sqrt((ankle.x - hip.x)**2 + (ankle.y - hip.y)**2) * image_height
    
    foot_x, foot_y = int(ankle.x * image_width), int(ankle.y * image_height)
    foot_size = leg_size * 0.2
    angle = np.degrees(np.arctan2(ankle.y - hip.y, ankle.x - hip.x))
    return {"leg": ((leg_x, leg_y), leg_size, angle), "foot": ((foot_x, foot_y), foot_size, angle)}

def calculate_waist_position(pose_landmarks, image_width, image_height):
    """Calculate position for belt."""
    if not pose_landmarks:
        return None
    landmarks = pose_landmarks.landmark
    left_hip = landmarks[23]
    right_hip = landmarks[24]
    
    waist_x = int((left_hip.x + right_hip.x) * image_width / 2)
    waist_y = int((left_hip.y + right_hip.y) * image_height / 2)
    waist_size = np.sqrt((right_hip.x - left_hip.x)**2 + (right_hip.y - left_hip.y)**2) * image_width
    angle = np.degrees(np.arctan2(right_hip.y - left_hip.y, right_hip.x - left_hip.x))
    return (waist_x, waist_y), waist_size, angle

def calculate_hand_position(hand_landmarks, image_width, image_height, side="right"):
    """Calculate hand position for pencil/gadget/product."""
    if not hand_landmarks:
        return None
    for hand in hand_landmarks:
        handedness = "right" if hand.classification[0].label == "Right" else "left"
        if handedness == side:
            wrist = hand.landmark[0]
            middle_finger_base = hand.landmark[9]
            palm_x = int((wrist.x + middle_finger_base.x) * image_width / 2)
            palm_y = int((wrist.y + middle_finger_base.y) * image_height / 2)
            hand_size = np.sqrt((middle_finger_base.x - wrist.x)**2 + (middle_finger_base.y - wrist.y)**2) * image_width
            angle = np.degrees(np.arctan2(middle_finger_base.y - wrist.y, middle_finger_base.x - wrist.x))
            return (palm_x, palm_y), hand_size, angle
    return None

def place_item(model_image, item_image, position, item_size, angle, scale_factor=0.8):
    """Generic function to place an item on the model."""
    try:
        scale = item_size * scale_factor / max(item_image.width, item_image.height)
        new_size = (int(item_image.width * scale), int(item_image.height * scale))
        item_image = item_image.resize(new_size, Image.LANCZOS)
        item_image = item_image.rotate(-angle, expand=True)
        
        paste_x = int(position[0] - item_image.width // 2)
        paste_y = int(position[1] - item_image.height // 2)
        model_image.paste(item_image, (paste_x, paste_y), item_image)
        return model_image
    except Exception as e:
        print(f"Warning: Failed to place item: {str(e)}")
        return model_image

def generate_ad_image(item_image_data, item_type, pose, side, background_desc):
    """Generate the final advertising image with the specified item."""
    model_image_path = None
    try:
        model_image_path = generate_model_image(pose, background_desc)
        
        body_parts = detect_body_parts(model_image_path)
        final_image = Image.open(model_image_path).convert("RGBA")
        item_image = remove_background(item_image_data)
        
        if item_type == "glasses" and body_parts["face"]:
            result = calculate_face_position(body_parts["face"], final_image.width, final_image.height)
            if result:
                pos, size, angle = result
                final_image = place_item(final_image, item_image, pos, size, angle, scale_factor=0.7)
        
        elif item_type == "hat" and body_parts["face"]:
            result = calculate_face_position(body_parts["face"], final_image.width, final_image.height)
            if result:
                pos, size, angle = result
                final_image = place_item(final_image, item_image, (pos[0], pos[1] - int(size * 0.2)), size, angle, scale_factor=0.9)
        
        elif item_type == "wristwatch" and body_parts["hands"]:
            result = calculate_wrist_position(body_parts["hands"], final_image.width, final_image.height, side)
            if result:
                pos, size, angle = result
                final_image = place_item(final_image, item_image, pos, size, angle, scale_factor=0.8)
        
        elif item_type == "gloves" and body_parts["hands"]:
            result = calculate_wrist_position(body_parts["hands"], final_image.width, final_image.height, side)
            if result:
                pos, size, angle = result
                final_image = place_item(final_image, item_image, pos, size, angle, scale_factor=0.9)
        
        elif item_type == "belt" and body_parts["pose"]:
            result = calculate_waist_position(body_parts["pose"], final_image.width, final_image.height)
            if result:
                pos, size, angle = result
                final_image = place_item(final_image, item_image, pos, size, angle, scale_factor=0.8)
        
        elif item_type == "trousers" and body_parts["pose"]:
            result = calculate_lower_body_position(body_parts["pose"], final_image.width, final_image.height, "right")
            if result:
                pos, size, angle = result["leg"]
                final_image = place_item(final_image, item_image, pos, size, angle, scale_factor=1.0)
        
        elif item_type == "shoes" and body_parts["pose"]:
            result = calculate_lower_body_position(body_parts["pose"], final_image.width, final_image.height, side)
            if result:
                pos, size, angle = result["foot"]
                final_image = place_item(final_image, item_image, pos, size, angle, scale_factor=0.8)
        
        elif item_type in ["pencil", "gadget", "product"] and body_parts["hands"]:
            result = calculate_hand_position(body_parts["hands"], final_image.width, final_image.height, side)
            if result:
                pos, size, angle = result
                scale_factor = 0.4 if item_type == "pencil" else 0.5
                final_image = place_item(final_image, item_image, pos, size, angle, scale_factor=scale_factor)
        
        output_path = os.path.join(tempfile.gettempdir(), "ad_image.png")
        final_image.save(output_path)
        return output_path
    except Exception as e:
        raise RuntimeError(f"Ad image generation failed: {str(e)}")
    finally:
        # Clean up intermediate file
        if model_image_path and os.path.exists(model_image_path):
            try:
                os.remove(model_image_path)
            except Exception as e:
                print(f"Warning: Failed to delete temp file {model_image_path}: {str(e)}")

@app.post("/generate")
async def generate_ad_image_endpoint(file: UploadFile = File(...), prompt: str = Form(...)):
    """Generate an ad image based on the uploaded item image and prompt."""
    try:
        # Validate file
        if not file.content_type.startswith("image/"):
            raise HTTPException(status_code=400, detail="Uploaded file must be an image.")
        if file.size > 5 * 1024 * 1024:  # 5MB limit
            raise HTTPException(status_code=400, detail="Image size must be less than 5MB.")
        
        # Read image data
        image_data = await file.read()
        
        # Parse prompt
        params = parse_prompt(prompt)
        
        # Generate image
        output_path = generate_ad_image(
            item_image_data=image_data,
            item_type=params["item_type"],
            pose=params["pose"],
            side=params["side"],
            background_desc=params["background"]
        )
        
        return FileResponse(output_path, media_type="image/png", filename="ad_image.png")
    except Exception as e:
        # Log the full error for debugging
        print(f"Error in /generate endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail="Failed to generate image. Please try again or contact support.")


if __name__ == "__main__":
    import uvicorn
    print("Starting server.py...")
    port = int(os.getenv("PORT", 8000))
    print(f"Binding to port: {port}")
    uvicorn.run(app, host="0.0.0.0", port=port)
    print("Uvicorn started successfully")
