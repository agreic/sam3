import os
import glob
import argparse
from pathlib import Path

import torch
import numpy as np
import torchvision
from PIL import Image
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader

# SAM3 Imports
from sam3 import build_sam3_image_model
from sam3.eval.postprocessors import PostProcessImage
from sam3.train.transforms.basic_for_api import (
    ComposeAPI,
    RandomResizeAPI,
    ToTensorAPI,
    NormalizeAPI,
    CLAHETransformAPI,
)
from sam3.train.data.sam3_image_dataset import (
    InferenceMetadata,
    FindQueryLoaded,
    Image as SAMImage,
    Datapoint,
)
from sam3.train.data.collator import collate_fn_api as collate_fn
from sam3.model.utils.misc import copy_data_to_device

# --- Constants & Configuration (aligned with SEGMENTATION.ipynb) ---
TILE_SIZE = 1008
# Hardcoded for 2304x2304 images based on notebook logic
# Stride calculation: (2304 - 1008) / 2 = 648
STRIDES = [0, 648, 2304 - TILE_SIZE]

# NOTE: This counter is intentionally global (like the notebook) so that
# every datapoint/query has a UNIQUE id. Reusing the same id for multiple
# tiles of one image causes SAM3's postprocessor to drop/overwrite results.
GLOBAL_COUNTER = 1


def create_empty_datapoint():
    """Create an empty Datapoint, matching SEGMENTATION.ipynb semantics."""
    return Datapoint(find_queries=[], images=[])


def set_image(datapoint, pil_image):
    """Attach the image to the datapoint, like in the notebook."""
    w, h = pil_image.size
    datapoint.images = [SAMImage(data=pil_image, objects=[], size=[h, w])]


def add_text_prompt(datapoint, text_query):
    """Add a text query to the datapoint, using a unique global id."""
    global GLOBAL_COUNTER
    assert len(datapoint.images) == 1, "please set the image first"

    w, h = datapoint.images[0].size
    datapoint.find_queries.append(
        FindQueryLoaded(
            query_text=text_query,
            image_id=0,
            object_ids_output=[],
            is_exhaustive=True,
            query_processing_order=0,
            inference_metadata=InferenceMetadata(
                coco_image_id=GLOBAL_COUNTER,
                original_image_id=GLOBAL_COUNTER,
                original_category_id=1,
                original_size=[w, h],
                object_id=0,
                frame_index=0,
            ),
        )
    )
    GLOBAL_COUNTER += 1
    return GLOBAL_COUNTER - 1

class TiledSAM3Dataset(Dataset):
    def __init__(self, image_files, text_prompt, transform=None):
        self.image_files = image_files
        self.text_prompt = text_prompt
        self.transform = transform

        # Pre-calculate all tile tasks
        self.tasks = []
        for img_path in image_files:
            base_name = Path(img_path).stem
            # Create a task for every tile position
            for top in STRIDES:
                for left in STRIDES:
                    self.tasks.append(
                        {
                            "path": img_path,
                            "base_name": base_name,
                            "top": top,
                            "left": left,
                        }
                    )

    def __len__(self):
        return len(self.tasks)

    def __getitem__(self, idx):
        task = self.tasks[idx]

        # 1. Load Image
        try:
            full_img = Image.open(task["path"])

            # 2. Crop Tile
            tile = full_img.crop(
                (
                    task["left"],
                    task["top"],
                    task["left"] + TILE_SIZE,
                    task["top"] + TILE_SIZE,
                )
            )

            # 3. Standardize to RGB (incl. 16-bit → 8-bit)
            if tile.mode != "RGB":
                if tile.mode in ["I;16", "I"]:
                    arr = np.array(tile)
                    # Normalize 16-bit to 8-bit
                    arr = (
                        (arr - arr.min())
                        / (arr.max() - arr.min() + 1e-6)
                        * 255
                    ).astype(np.uint8)
                    tile = Image.fromarray(arr)
                tile = tile.convert("RGB")

            # 4. Create Datapoint
            dp = create_empty_datapoint()
            set_image(dp, tile)
            qid = add_text_prompt(dp, self.text_prompt)

            # 5. Apply Transforms
            if self.transform:
                dp = self.transform(dp)

            # Metadata for reconstruction
            metadata = {
                "base_name": task["base_name"],
                "offset": torch.tensor(
                    [task["left"], task["top"], task["left"], task["top"]]
                ),
                "query_id": qid,
            }

            return dp, metadata

        except Exception as e:
            print(f"Error processing {task['path']}: {e}")
            return None, None

def custom_collate(batch):
    # Filter out failed samples
    batch = [b for b in batch if b[0] is not None]
    if not batch:
        return None

    datapoints, metadatas = zip(*batch)

    # Use SAM3's internal collator for the datapoints
    batched_input = collate_fn(datapoints, dict_key="dummy")["dummy"]

    return batched_input, metadatas

def save_image_results(base_name, data, output_folder):
    """Stitches tiles, runs NMS, and saves the final image."""
    if not data["boxes"]:
        return

    # 1. Concatenate all tiles
    all_boxes = torch.cat(data["boxes"]).detach()
    all_scores = torch.cat(data["scores"]).detach()
    all_masks = torch.cat(data["masks"]).detach()
    all_offsets = torch.cat(data["mask_offsets"]).detach()

    # 2. Cast to float32 for NMS
    all_boxes = all_boxes.to(dtype=torch.float32)
    all_scores = all_scores.to(dtype=torch.float32)

    # 3. Global NMS
    keep_indices = torchvision.ops.nms(all_boxes, all_scores, iou_threshold=0.5)

    # 4. Filter results
    final_boxes = all_boxes[keep_indices].cpu().numpy()
    final_scores = all_scores[keep_indices].cpu().numpy()
    final_masks = all_masks[keep_indices] # Keep on GPU
    final_offsets = all_offsets[keep_indices]

    # 5. Reconstruct Mask Image (match notebook: 2304x2304, instance-id labels)
    full_mask = torch.zeros((2304, 2304), dtype=torch.int32, device="cuda")
    
    for i, (mask_tensor, offset) in enumerate(zip(final_masks, final_offsets)):
        if mask_tensor.ndim == 3: mask_tensor = mask_tensor.squeeze(0)
        
        binary_mask = mask_tensor > 0.0 if mask_tensor.min() < 0 else mask_tensor > 0.5
        
        x_start, y_start = int(offset[0]), int(offset[1])
        x_end = min(int(x_start + TILE_SIZE), 2304)
        y_end = min(int(y_start + TILE_SIZE), 2304)
        
        mask_h, mask_w = y_end - y_start, x_end - x_start
        valid_binary_mask = binary_mask[:mask_h, :mask_w].to(full_mask.device)

        current_slice = full_mask[y_start:y_end, x_start:x_end]
        full_mask[y_start:y_end, x_start:x_end] = torch.where(
            valid_binary_mask, 
            torch.tensor(i + 1, dtype=torch.int32, device=full_mask.device), 
            current_slice
        )

    # 6. Save Files
    full_mask_np = full_mask.cpu().numpy().astype(np.uint16)
    Image.fromarray(full_mask_np).save(os.path.join(output_folder, f"{base_name}_masks.png"))
    np.savetxt(os.path.join(output_folder, f"{base_name}_boxes.txt"), final_boxes, fmt="%.2f")
    np.savetxt(os.path.join(output_folder, f"{base_name}_scores.txt"), final_scores, fmt="%.4f")

def main():
    parser = argparse.ArgumentParser(
        description="SAM3 High Performance Inference (SEGMENTATION.ipynb as script)"
    )
    parser.add_argument("--input_folder", type=str, required=True, help="Path to input images")
    parser.add_argument("--output_folder", type=str, required=True, help="Path to save results")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default="/workspaces/sam3-main/assets/models/sam3.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--bpe_path",
        type=str,
        default="/workspaces/sam3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz",
        help="Path to BPE vocab",
    )
    parser.add_argument(
        "--text_prompt",
        type=str,
        default="cells",
        help="Text prompt for segmentation",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=4,
        help="Batch size for inference (tiles, not full images)",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="Number of data loading workers",
    )
    parser.add_argument(
        "--enable_sanity_checks",
        action="store_true",
        help="Run optional, lightweight sanity checks on the first batch.",
    )
    args = parser.parse_args()

    # Setup
    os.makedirs(args.output_folder, exist_ok=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this high-performance inference script.")
    device = torch.device("cuda")

    # Match notebook mixed-precision / TF32 settings
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    # Load Model
    print(f"Loading model from {args.checkpoint}...")
    model = (
        build_sam3_image_model(
            bpe_path=args.bpe_path,
            checkpoint_path=args.checkpoint,
        )
        .to(device)
        .eval()
    )

    postprocessor = PostProcessImage(
        max_dets_per_img=-1,
        iou_type="segm",
        use_original_sizes_box=True,
        use_original_sizes_mask=True,
        convert_mask_to_rle=False,
        detection_threshold=0.65,
        to_cpu=True,
    )

    # Find Images
    image_extensions = ['*.png', '*.jpg', '*.jpeg', '*.tif', '*.tiff', '*.bmp']
    image_files = []
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(args.input_folder, ext)))
        image_files.extend(glob.glob(os.path.join(args.input_folder, ext.upper())))
    
    image_files = sorted(list(set(image_files)))
    # Filter for 'w00' if needed, or remove this line to process everything
    image_files = [i for i in image_files if "w00" in i] 
    
    print(f"Found {len(image_files)} images.")

    # Dataset & Loader
    # Match SEGMENTATION.ipynb: CLAHE → resize 1008 → tensor → normalize 0.5/0.5
    transform = ComposeAPI(
        transforms=[
            CLAHETransformAPI(clip_limit=3.0),
            RandomResizeAPI(sizes=1008, max_size=1008, square=True, consistent_transform=False),
            ToTensorAPI(),
            NormalizeAPI(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ]
    )

    dataset = TiledSAM3Dataset(image_files, args.text_prompt, transform=transform)
    dataloader = DataLoader(
        dataset, 
        batch_size=args.batch_size, 
        shuffle=False, # IMPORTANT: False ensures tiles for one image come together
        num_workers=args.num_workers, 
        collate_fn=custom_collate,
        pin_memory=True
    )

    # Buffer for results
    current_image_results = {"boxes": [], "scores": [], "masks": [], "mask_offsets": []}
    current_base_name = None

    print("Starting Inference Loop...")

    sanity_done = False
    with torch.inference_mode():
        for batch in tqdm(dataloader, desc="Batches"):
            if batch is None:
                continue

            batch_input, metadatas = batch

            # Move to GPU
            batch_input = copy_data_to_device(batch_input, device, non_blocking=True)

            # Inference with autocast for throughput (similar to notebook usage)
            with torch.autocast(device_type="cuda", dtype=torch.float16):
                output = model(batch_input)
            
            # Post-process
            processed_batch = postprocessor.process_results(output, batch_input.find_metadatas)
            if isinstance(processed_batch, dict):
                results_list = list(processed_batch.values())
            else:
                results_list = processed_batch

            # Optional, opt-in sanity checks on first batch only
            if args.enable_sanity_checks and not sanity_done:
                first_res = results_list[0]
                print("[sanity] First result keys:", list(first_res.keys()))
                for key in ("boxes", "scores", "masks"):
                    if key in first_res and torch.is_tensor(first_res[key]):
                        t = first_res[key]
                        print(f"[sanity] {key} shape={tuple(t.shape)}, nan_any={torch.isnan(t).any().item()}")
                sanity_done = True

            # # Process Results
            # for i, meta in enumerate(metadatas):
            #     res = results_list[i]
            #     base_name = meta["base_name"]

            for res, meta in zip(results_list, metadatas): 
                base_name = meta["base_name"]
                
                # Check if we moved to a new image
                if current_base_name is not None and base_name != current_base_name:
                    # Save previous image results
                    save_image_results(current_base_name, current_image_results, args.output_folder)
                    # Clear buffer
                    current_image_results = {"boxes": [], "scores": [], "masks": [], "mask_offsets": []}
                
                current_base_name = base_name

                # Filter & Accumulate (Edge Rejection Logic)
                if "boxes" in res and len(res["boxes"]) > 0:
                    boxes = res["boxes"]
                    scores = res["scores"]
                    masks = res["masks"]

                    # Edge Logic
                    margin = 5
                    offset = meta["offset"].to(boxes.device)
                    # offset is [left, top, left, top]

                    is_left = offset[0] == 0
                    is_top = offset[1] == 0
                    is_right = offset[0] + TILE_SIZE >= 2304
                    is_bottom = offset[1] + TILE_SIZE >= 2304

                    keep_mask = torch.ones(len(boxes), dtype=torch.bool, device=boxes.device)

                    if not is_left:
                        keep_mask &= boxes[:, 0] > margin
                    if not is_top:
                        keep_mask &= boxes[:, 1] > margin
                    if not is_right:
                        keep_mask &= boxes[:, 2] < TILE_SIZE - margin
                    if not is_bottom:
                        keep_mask &= boxes[:, 3] < TILE_SIZE - margin

                    if keep_mask.sum() > 0:
                        current_image_results["boxes"].append(boxes[keep_mask] + offset)
                        current_image_results["scores"].append(scores[keep_mask])
                        current_image_results["masks"].append(masks[keep_mask])
                        # Store offsets for reconstruction
                        n_dets = keep_mask.sum()
                        current_image_results["mask_offsets"].append(offset[:2].repeat(n_dets, 1))

        # Save the very last image
        if current_base_name is not None:
             save_image_results(current_base_name, current_image_results, args.output_folder)

    print("Done!")

if __name__ == "__main__":
    main()
