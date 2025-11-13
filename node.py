import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import copy
import torch
import numpy as np
from PIL import Image
import logging
from torch.hub import download_url_to_file
from urllib.parse import urlparse
import folder_paths
import comfy.model_management
from sam2.build_sam import build_sam2
from sam2.sam2_image_predictor import SAM2ImagePredictor
from local_groundingdino.datasets import transforms as T
from local_groundingdino.util.utils import (
    clean_state_dict as local_groundingdino_clean_state_dict,
)
from local_groundingdino.util.slconfig import SLConfig as local_groundingdino_SLConfig
from local_groundingdino.models import build_model as local_groundingdino_build_model
import glob
import folder_paths
from hydra import initialize
from hydra.core.global_hydra import GlobalHydra

# Import the new point extraction utilities
try:
    import point_extraction_utils
except ImportError:
    print("Error: Could not import 'point_extraction_utils.py'. Make sure it is in the same directory as 'node.py'.")
    # As a fallback, create dummy functions so the rest of the file can load
    class DummyPointExtraction:
        def get_positive_points(self, boxes, num_points_per_box=1):
            return np.empty((0, 2), dtype=np.float32)
        def get_negative_points(self, boxes, image_shape, num_points=5):
            return np.empty((0, 2), dtype=np.float32)
        def combine_points_and_labels(self, positive_points, negative_points):
            return None, None
    point_extraction_utils = DummyPointExtraction()


logger = logging.getLogger("ComfyUI-SAM2")

sam_model_dir_name = "sam2"
sam_model_list = {
    "sam2_hiera_tiny": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_tiny.pt"
    },
    "sam2_hiera_small.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_small.pt"
    },
    "sam2_hiera_base_plus.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_base_plus.pt"
    },
    "sam2_hiera_large.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/072824/sam2_hiera_large.pt"
    },
    "sam2_1_hiera_tiny.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_tiny.pt"
    },
    "sam2_1_hiera_small.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_small.pt"
    },
    "sam2_1_hiera_base_plus.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_base_plus.pt"
    },
    "sam2_1_hiera_large.pt": {
        "model_url": "https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt"
    },
}

groundingdino_model_dir_name = "grounding-dino"
groundingdino_model_list = {
    "GroundingDINO_SwinT_OGC (694MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinT_OGC.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swint_ogc.pth",
    },
    "GroundingDINO_SwinB (938MB)": {
        "config_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/GroundingDINO_SwinB.cfg.py",
        "model_url": "https://huggingface.co/ShilongLiu/GroundingDINO/resolve/main/groundingdino_swinb_cogcoor.pth",
    },
}


def get_bert_base_uncased_model_path():
    comfy_bert_model_base = os.path.join(folder_paths.models_dir, "bert-base-uncased")
    if glob.glob(
        os.path.join(comfy_bert_model_base, "**/model.safetensors"), recursive=True
    ):
        print("grounding-dino is using models/bert-base-uncased")
        return comfy_bert_model_base
    return "bert-base-uncased"


def list_files(dirpath, extensions=[]):
    return [
        f
        for f in os.listdir(dirpath)
        if os.path.isfile(os.path.join(dirpath, f)) and f.split(".")[-1] in extensions
    ]


def list_sam_model():
    return list(sam_model_list.keys())


def load_sam_model(model_name):
    sam2_checkpoint_path = get_local_filepath(
        sam_model_list[model_name]["model_url"], sam_model_dir_name
    )
    model_file_name = os.path.basename(sam2_checkpoint_path)
    model_file_name = model_file_name.replace("2.1", "2_1")
    model_type = model_file_name.split(".")[0]

    if GlobalHydra().is_initialized():
        GlobalHydra.instance().clear()

    config_path = "sam2_configs"
    initialize(config_path=config_path)
    model_cfg = f"{model_type}.yaml"

    sam_device = comfy.model_management.get_torch_device()
    sam = build_sam2(model_cfg, sam2_checkpoint_path, device=sam_device)
    sam.model_name = model_file_name
    return sam


def get_local_filepath(url, dirname, local_file_name=None):
    if not local_file_name:
        parsed_url = urlparse(url)
        local_file_name = os.path.basename(parsed_url.path)

    destination = folder_paths.get_full_path(dirname, local_file_name)
    if destination:
        logger.warn(f"using extra model: {destination}")
        return destination

    folder = os.path.join(folder_paths.models_dir, dirname)
    if not os.path.exists(folder):
        os.makedirs(folder)

    destination = os.path.join(folder, local_file_name)
    if not os.path.exists(destination):
        logger.warn(f"downloading {url} to {destination}")
        download_url_to_file(url, destination)
    return destination


def load_groundingdino_model(model_name):
    dino_model_args = local_groundingdino_SLConfig.fromfile(
        get_local_filepath(
            groundingdino_model_list[model_name]["config_url"],
            groundingdino_model_dir_name,
        ),
    )

    if dino_model_args.text_encoder_type == "bert-base-uncased":
        dino_model_args.text_encoder_type = get_bert_base_uncased_model_path()

    dino = local_groundingdino_build_model(dino_model_args)
    checkpoint = torch.load(
        get_local_filepath(
            groundingdino_model_list[model_name]["model_url"],
            groundingdino_model_dir_name,
        ),
    )
    dino.load_state_dict(
        local_groundingdino_clean_state_dict(checkpoint["model"]), strict=False
    )
    device = comfy.model_management.get_torch_device()
    dino.to(device=device)
    dino.eval()
    return dino


def list_groundingdino_model():
    return list(groundingdino_model_list.keys())


def groundingdino_predict(dino_model, image, prompt, threshold):
    def load_dino_image(image_pil):
        transform = T.Compose(
            [
                T.RandomResize([800], max_size=1333),
                T.ToTensor(),
                T.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
            ]
        )
        image, _ = transform(image_pil, None)  # 3, h, w
        return image

    def get_grounding_output(model, image, caption, box_threshold):
        caption = caption.lower()
        caption = caption.strip()
        if not caption.endswith("."):
            caption = caption + "."
        device = comfy.model_management.get_torch_device()
        image = image.to(device)
        with torch.no_grad():
            outputs = model(image[None], captions=[caption])
        logits = outputs["pred_logits"].sigmoid()[0]  # (nq, 256)
        boxes = outputs["pred_boxes"][0]  # (nq, 4)
        
        # filter output
        logits_filt = logits.clone()
        boxes_filt = boxes.clone()
        filt_mask = logits_filt.max(dim=1)[0] > box_threshold
        logits_filt = logits_filt[filt_mask]  # num_filt, 256
        boxes_filt = boxes_filt[filt_mask]  # num_filt, 4
        
        # Return boxes_filt.cpu() and also logits_filt.cpu()
        # We don't use logits_filt for now, but this is where you'd get them.
        return boxes_filt.cpu()

    dino_image = load_dino_image(image.convert("RGB"))
    boxes_filt = get_grounding_output(dino_model, dino_image, prompt, threshold)
    H, W = image.size[1], image.size[0]
    for i in range(boxes_filt.size(0)):
        boxes_filt[i] = boxes_filt[i] * torch.Tensor([W, H, W, H])
        boxes_filt[i][:2] -= boxes_filt[i][2:] / 2
        boxes_filt[i][2:] += boxes_filt[i][:2]
    return boxes_filt


def create_pil_output(image_np, masks, boxes_filt):
    output_masks, output_images = [], []
    boxes_filt = boxes_filt.numpy().astype(int) if boxes_filt is not None else None
    for mask in masks:
        output_masks.append(Image.fromarray(np.any(mask, axis=0)))
        image_np_copy = copy.deepcopy(image_np)
        image_np_copy[~np.any(mask, axis=0)] = np.array([0, 0, 0, 0])
        output_images.append(Image.fromarray(image_np_copy))
    return output_images, output_masks


def create_tensor_output(image_np, masks):
    """
    Create a batch of preview images and masks for each detected object.
    
    Args:
        image_np (np.ndarray): Original RGBA or RGB image array.
        masks (np.ndarray): Array of shape (N, H, W) or (N, 1, H, W) where N is number of masks.
        boxes_filt (torch.Tensor or np.ndarray): Filtered bounding boxes (optional, not always used).

    Returns:
        tuple[list[Image.Image], list[Image.Image]]:
            output_images: preview images showing segmentation
            output_masks: grayscale or binary mask images
    """
    list_length = len(masks)
    output_masks, output_images = [], []

    # Handle 4D inputs (N, 1, H, W) by squeezing the channel dimension
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks.squeeze(1)

    # Ensure masks are boolean for indexing, but keep grayscale values if desired
    for i, mask in enumerate(masks):
        mask = np.asarray(mask)
        if mask.dtype != bool:
            # If it's a probability or grayscale mask, threshold lightly for visualization only
            vis_mask = mask > 0.5
        else:
            vis_mask = mask

        image_np_copy = image_np.copy()
        image_np_copy[~vis_mask] = np.array([0, 0, 0, 0])

        output_image, output_mask = split_image_mask(Image.fromarray(image_np_copy))
        output_images.append(output_image)
        output_masks.append(output_mask)

    return output_images, output_masks


def split_image_mask(image):
    image_rgb = image.convert("RGB")
    image_rgb = np.array(image_rgb).astype(np.float32) / 255.0
    image_rgb = torch.from_numpy(image_rgb)[None,]
    if "A" in image.getbands():
        mask = np.array(image.getchannel("A")).astype(np.float32) / 255.0
        mask = torch.from_numpy(mask)[None,]
    else:
        mask = torch.zeros((64, 64), dtype=torch.float32, device="cpu")
    return (image_rgb, mask)


def sam_segment(sam_model, image, boxes, 
                use_point_prompts=False, positive_points_per_box=1, negative_points_count=0):
    """
    Updated segmentation function to optionally use point prompts.
    
    Args:
        sam_model: The loaded SAM2 model.
        image: The input PIL Image.
        boxes: A torch.Tensor of bounding boxes (N, 4).
        use_point_prompts (bool): Whether to generate and use point prompts.
        positive_points_per_box (int): Number of positive points inside each box.
        negative_points_count (int): Number of negative points outside all boxes.
    """
    if boxes.shape[0] == 0:
        return None
    
    predictor = SAM2ImagePredictor(sam_model)
    image_np = np.array(image)
    image_np_rgb = image_np[..., :3]
    predictor.set_image(image_np_rgb)
    
    point_coords, point_labels = None, None
    boxes_to_pass = boxes
    
    if use_point_prompts:
        # Convert boxes to numpy for utility functions
        boxes_np = boxes.cpu().numpy()
        num_boxes = boxes_np.shape[0]
        
        # Generate points per box
        all_point_coords = []
        all_point_labels = []
        
        for i in range(num_boxes):
            box = boxes_np[i:i+1]  # Keep as (1, 4) for the utility function
            
            # 1. Get positive points (inside this box)
            pos_points = point_extraction_utils.get_positive_points(
                box, 
                num_points_per_box=positive_points_per_box
            )
            
            # 2. Get negative points (outside this box)
            # For simplicity, we generate negative points outside ALL boxes
            # and use the same set for each box
            if i == 0:  # Only generate once
                image_shape_hw = image_np_rgb.shape[:2]
                neg_points = point_extraction_utils.get_negative_points(
                    boxes_np,  # Pass all boxes to avoid them
                    image_shape=image_shape_hw, 
                    num_points=negative_points_count
                )
            
            # 3. Combine points for this box
            box_point_coords, box_point_labels = point_extraction_utils.combine_points_and_labels(
                pos_points, 
                neg_points
            )
            
            if box_point_coords is not None:
                all_point_coords.append(box_point_coords)
                all_point_labels.append(box_point_labels)
        
        # Stack all points into batched format: (B, N, 2) and (B, N)
        if all_point_coords:
            point_coords = np.stack(all_point_coords, axis=0)
            point_labels = np.stack(all_point_labels, axis=0)
            print(f"Using {point_coords.shape[0]} boxes with {point_coords.shape[1]} points each "
                  f"(total {point_coords.shape[1]} points per box: {positive_points_per_box} pos, {negative_points_count} neg shared).")
        else:
            point_coords = None
            point_labels = None

    sam_device = comfy.model_management.get_torch_device()
    masks, scores, _ = predictor.predict(
        point_coords=point_coords,  # Shape: (B, N, 2) or None
        point_labels=point_labels,  # Shape: (B, N) or None
        box=boxes_to_pass,          # Shape: (B, 4)
        multimask_output=True
    )
    
    if masks.ndim == 3:
        masks = np.expand_dims(masks, axis=0)    
    return create_tensor_output(image_np, masks)   


class SAM2ModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_sam_model(),),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("SAM2_MODEL",)

    def main(self, model_name):
        sam_model = load_sam_model(model_name)
        return (sam_model,)


class GroundingDinoModelLoader:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "model_name": (list_groundingdino_model(),),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("GROUNDING_DINO_MODEL",)

    def main(self, model_name):
        dino_model = load_groundingdino_model(model_name)
        return (dino_model,)


class GroundingDinoSAM2Segment:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "sam_model": ("SAM2_MODEL", {}),
                "grounding_dino_model": ("GROUNDING_DINO_MODEL", {}),
                "image": ("IMAGE", {}),
                "prompt": ("STRING", {}),
                "threshold": (
                    "FLOAT",
                    {"default": 0.3, "min": 0, "max": 1.0, "step": 0.01},
                ),
            },
            "optional": {
                "use_point_prompts": ("BOOLEAN", {"default": True}),
                "positive_points_per_box": ("INT", {"default": 1, "min": 0, "max": 10, "step": 1}),
                "negative_points_count": ("INT", {"default": 5, "min": 0, "max": 50, "step": 1}),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("IMAGE", "MASK")

    def main(self, grounding_dino_model, sam_model, image, prompt, threshold,
             use_point_prompts=True, positive_points_per_box=1, negative_points_count=5):
        
        res_images = []
        res_masks = []
        for item in image:
            item = Image.fromarray(
                np.clip(255.0 * item.cpu().numpy(), 0, 255).astype(np.uint8)
            ).convert("RGBA")
            
            boxes = groundingdino_predict(grounding_dino_model, item, prompt, threshold)
            
            if boxes.shape[0] == 0:
                # No boxes found, skip segmentation
                continue
                
            (images, masks) = sam_segment(
                sam_model, 
                item, 
                boxes,
                use_point_prompts=use_point_prompts,
                positive_points_per_box=positive_points_per_box,
                negative_points_count=negative_points_count
            )
            
            if images:
                res_images.extend(images)
                res_masks.extend(masks)
                
        if len(res_images) == 0:
            # Handle case where no boxes were found or segmentation failed
            print("No segments found. Returning empty masks.")
            # Get original image shape for empty mask
            _, height, width, _ = image.shape
            empty_mask = torch.zeros(
                (1, height, width), dtype=torch.float32, device="cpu"
            )
            empty_image = torch.zeros(
                (1, height, width, 3), dtype=torch.float32, device="cpu"
            )
            return (empty_image, empty_mask)
            
        imageStack = torch.stack(res_images, dim=0)
        imageStack = torch.squeeze(imageStack, dim=1)
        maskStack = torch.stack(res_masks, dim=0)
        return (imageStack, maskStack)


class InvertMask:
    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "mask": ("MASK",),
            }
        }

    CATEGORY = "segment_anything2"
    FUNCTION = "main"
    RETURN_TYPES = ("MASK",)

    def main(self, mask):
        out = 1.0 - mask
        return (out,)


class IsMaskEmptyNode:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ["NUMBER"]
    RETURN_NAMES = ["boolean_number"]

    FUNCTION = "main"
    CATEGORY = "segment_anything2"

    def main(self, mask):
        return (torch.all(mask == 0).int().item(),)
