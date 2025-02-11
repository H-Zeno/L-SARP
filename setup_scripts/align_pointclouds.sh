#!/bin/bash

# Configuration
LSARP_DIR="${LSARP%/}"  # Remove trailing slash if present
CONFIG_FILE="configs/config.yaml"

# Function to check if directory exists
check_dir() {
    if [ ! -d "$1" ]; then
        echo "Creating directory: $1"
        mkdir -p "$1"
    fi
}

# Function to read from config.yaml
read_config() {
    # Use yq or python to properly parse YAML
    if command -v yq >/dev/null 2>&1; then
        # If yq is available
        LOW_RES_NAME=$(yq eval '.pre_scanned_graphs.low_res' "$CONFIG_FILE")
        HIGH_RES_NAME=$(yq eval '.pre_scanned_graphs.high_res' "$CONFIG_FILE")
    else
        # Fallback to Python if yq is not available
        LOW_RES_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['pre_scanned_graphs']['low_res'])")
        HIGH_RES_NAME=$(python3 -c "import yaml; print(yaml.safe_load(open('$CONFIG_FILE'))['pre_scanned_graphs']['high_res'])")
    fi
    
    echo "Using configuration:"
    echo "Low resolution scan: $LOW_RES_NAME"
    echo "High resolution scan: $HIGH_RES_NAME"
}

# Setup required directories
check_dir "${LSARP_DIR}/data/autowalk"
check_dir "${LSARP_DIR}/data/point_clouds"
check_dir "${LSARP_DIR}/data/prescans"
check_dir "${LSARP_DIR}/data/aligned_point_clouds"

# Read scan names from config
read_config

# Process low-res point cloud
echo "Processing low resolution point cloud..."
if [ -f "${LSARP_DIR}/data/autowalk/${LOW_RES_NAME}.walk/point_cloud.ply" ]; then
    cp "${LSARP_DIR}/data/autowalk/${LOW_RES_NAME}.walk/point_cloud.ply" \
       "${LSARP_DIR}/data/point_clouds/${LOW_RES_NAME}.ply"
else
    echo "Error: Low resolution point cloud not found"
    exit 1
fi

# Process high-res point cloud
echo "Processing high resolution point cloud..."
check_dir "${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}"
cp "${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}/${HIGH_RES_NAME}.ply" "${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}/pcd.ply"

# Run alignment script
echo "Running point cloud alignment..."
python3 "${LSARP_DIR}/source/scripts/point_cloud_scripts/full_align.py"
