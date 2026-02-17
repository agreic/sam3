import os
import subprocess
import glob
from pathlib import Path

# --- Configuration ---
# The folder containing all your position subfolders (e.g. 230624DS30_p0001, etc.)
BASE_INPUT_DIR = "/root/data/230624DS30"

# Where you want the results for each position to go
BASE_OUTPUT_DIR = "/root/data/Segmentation_SAM3/230624DS30"

# Path to your inference script
SCRIPT_PATH = "scripts/run_inference.py"

# Constant arguments for the model
CHECKPOINT = "/workspaces/sam3-main/assets/models/sam3.pt"
BPE_PATH = "/workspaces/sam3-main/sam3/assets/bpe_simple_vocab_16e6.txt.gz"
TEXT_PROMPT = "cells"
BATCH_SIZE = "16"
NUM_WORKERS = "16"

def main():
    # 1. Find all position folders (looking for pattern *_p*)
    # This finds folders like /root/data/230624DS30/230624DS30_p0001
    search_pattern = os.path.join(BASE_INPUT_DIR, "*_p*")
    position_folders = sorted(glob.glob(search_pattern))

    # Filter to ensure they are actually directories
    position_folders = [p for p in position_folders if os.path.isdir(p)]

    if not position_folders:
        print(f"No position folders found in {BASE_INPUT_DIR}!")
        return

    print(f"Found {len(position_folders)} positions to process.")
    print("-" * 50)

    # 2. Iterate and Run
    for input_path in position_folders:
        folder_name = os.path.basename(input_path)  # e.g., "230624DS30_p0001"
        
        # Construct the specific output path for this position
        output_path = os.path.join(BASE_OUTPUT_DIR, folder_name)

        print(f"Processing: {folder_name}")
        print(f"Input:  {input_path}")
        print(f"Output: {output_path}")

        # Construct the command
        cmd = [
            "python", SCRIPT_PATH,
            "--input_folder", input_path,
            "--output_folder", output_path,
            "--checkpoint", CHECKPOINT,
            "--bpe_path", BPE_PATH,
            "--text_prompt", TEXT_PROMPT,
            "--batch_size", BATCH_SIZE,
            "--num_workers", NUM_WORKERS
        ]

        try:
            # Run the command and wait for it to finish
            # check=True raises an error if the script fails, stopping the batch
            subprocess.run(cmd, check=True)
            print(f"✅ Finished {folder_name}\n")
            
        except subprocess.CalledProcessError as e:
            print(f"❌ Error processing {folder_name}. Stopping batch.")
            print(f"Error details: {e}")
            break
        except KeyboardInterrupt:
            print("\n🛑 Batch processing stopped by user.")
            break

    print("-" * 50)
    print("Batch processing complete.")

if __name__ == "__main__":
    main()