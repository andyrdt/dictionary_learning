#!/bin/bash

# Exit immediately if a command exits with a non-zero status.
set -e

# --- Configuration ---
# Base path where your SAEs are located and will be organized
BASE_SAE_PATH="/workspace/sae/llama-3-8b-instruct"

# This part of the subpath is static
STATIC_PART_OF_RUNS_SUBPATH_PREFIX="runs/2_l" # Layer number will be inserted here
STATIC_PART_OF_RUNS_SUBPATH_SUFFIX="meta-llama_Llama-3.1-8B-Instruct_batch_top_k"

# Name of the target directory to consolidate SAEs
CONSOLIDATED_SAES_DIR_NAME="saes"

# Layer numbers to process
LAYER_NUMBERS=(3 7 11 15 19 23 27)

# --- Hugging Face Configuration (IMPORTANT: UPDATE THESE VALUES) ---
# Replace with your Hugging Face username and the desired repository name
HF_REPO_ID="andyrdt/sae-llama-3-8b-instruct"
# Specify the type of the repository, e.g., "model", "dataset", "space"
HF_REPO_TYPE="model" # Assuming these are model artifacts

# --- Script Logic ---

# 1. Define full destination parent path
FULL_DEST_PARENT_DIR="${BASE_SAE_PATH}/${CONSOLIDATED_SAES_DIR_NAME}"

# 2. Create the main destination directory if it doesn't exist
echo "Ensuring main destination directory exists: ${FULL_DEST_PARENT_DIR}"
mkdir -p "${FULL_DEST_PARENT_DIR}"
echo "Main destination directory ready."
echo ""

# 3. Loop through layer numbers and copy directories using rsync for exclusion
echo "Starting to copy layer directories (excluding checkpoints)..."
for layer_num in "${LAYER_NUMBERS[@]}"; do
  # Format layer number to be two digits (e.g., 3 -> 03, 11 -> 11)
  formatted_layer_num=$(printf "%02d" "${layer_num}")

  # Construct the dynamic runs subpath for the current layer
  DYNAMIC_RUNS_SUBPATH="${STATIC_PART_OF_RUNS_SUBPATH_PREFIX}${formatted_layer_num}/${STATIC_PART_OF_RUNS_SUBPATH_SUFFIX}"
  
  SOURCE_LAYER_DIR_NAME="resid_post_layer_${layer_num}"
  
  # Construct the full source path for the current layer's SAE data
  FULL_SOURCE_LAYER_PATH="${BASE_SAE_PATH}/${DYNAMIC_RUNS_SUBPATH}/${SOURCE_LAYER_DIR_NAME}"
  
  # Define the specific destination path for this layer's SAEs
  # e.g., /workspace/sae/llama-3-8b-instruct/saes/resid_post_layer_3
  TARGET_DEST_LAYER_PATH="${FULL_DEST_PARENT_DIR}/${SOURCE_LAYER_DIR_NAME}"

  echo "Processing layer ${layer_num} (run subpath using l${formatted_layer_num}):"
  if [ -d "${FULL_SOURCE_LAYER_PATH}" ]; then
    echo "  Source: ${FULL_SOURCE_LAYER_PATH}/"
    echo "  Destination: ${TARGET_DEST_LAYER_PATH}/"
    
    # Ensure the specific target directory for the layer exists
    mkdir -p "${TARGET_DEST_LAYER_PATH}"
    
    echo "  Copying contents using rsync, excluding 'trainer_*/checkpoints/'..."
    rsync -av --exclude='trainer_*/checkpoints/' "${FULL_SOURCE_LAYER_PATH}/" "${TARGET_DEST_LAYER_PATH}/"
    
    echo "  Successfully copied contents of ${SOURCE_LAYER_DIR_NAME} (excluding specified checkpoints) to ${TARGET_DEST_LAYER_PATH}."
  else
    echo "  Warning: Source directory ${FULL_SOURCE_LAYER_PATH} not found. Skipping."
  fi
  echo ""
done

echo "All specified layer directories processed."
echo "Consolidated SAEs (with checkpoints excluded from trainer folders) are now in: ${FULL_DEST_PARENT_DIR}"
echo ""

# 4. Upload to Hugging Face
echo "--- Hugging Face Upload ---"
echo "The directory to be uploaded is: ${FULL_DEST_PARENT_DIR}"
echo "This will upload its contents (e.g., resid_post_layer_3, resid_post_layer_7, etc., with checkpoints excluded as specified) to the root of the HF repo."
echo ""
echo "Please ensure:"
echo "  1. You have updated HF_REPO_ID ('${HF_REPO_ID}') and HF_REPO_TYPE ('${HF_REPO_TYPE}') in this script."
echo "  2. You have 'huggingface-cli' installed and are logged in (run 'huggingface-cli login')."
echo "  3. 'rsync' is installed on your system."
echo ""
echo "The Hugging Face upload command that would run is:"
echo "huggingface-cli upload \"${HF_REPO_ID}\" \"${FULL_DEST_PARENT_DIR}\" --repo-type \"${HF_REPO_TYPE}\" --commit-message \"Add SAEs for layers: ${LAYER_NUMBERS[*]} (checkpoints excluded)\""
echo ""
echo "If you are ready to upload, uncomment the following lines in the script and run it again:"
echo ""
echo "Attempting to upload to Hugging Face repository: ${HF_REPO_ID}..."
huggingface-cli upload-large-folder "${HF_REPO_ID}" "${FULL_DEST_PARENT_DIR}" \
  --repo-type "${HF_REPO_TYPE}"
echo "Hugging Face upload command executed. Please check your repository on huggingface.co."

echo ""
echo "Script finished."
echo "Review the output above. If copying was successful and you've configured your HF details, you can uncomment the upload section."