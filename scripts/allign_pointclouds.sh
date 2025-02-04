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
    # Read values from config using grep and cut
    LOW_RES_NAME=$(grep "low_res:" "$CONFIG_FILE" | cut -d'"' -f2)
    HIGH_RES_NAME=$(grep "high_res:" "$CONFIG_FILE" | cut -d'"' -f2)
    
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
