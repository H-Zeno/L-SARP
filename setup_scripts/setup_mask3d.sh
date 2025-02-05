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

# Function to read high_res name from config
get_high_res_name() {
    grep "high_res:" "$CONFIG_FILE" | cut -d'"' -f2
}

# Setup Mask3D repository and checkpoints
echo "Setting up Mask3D..."
cd "${LSARP_DIR}/source" || exit 1

if [ ! -d "Mask3D" ]; then
    git submodule add https://github.com/behretj/Mask3D.git
fi

check_dir "Mask3D/checkpoints"
cd "Mask3D/checkpoints" || exit 1

if [ ! -f "mask3d_scannet200_demo.ckpt" ]; then
    wget "https://zenodo.org/records/10422707/files/mask3d_scannet200_demo.ckpt"
fi

cd "${LSARP_DIR}" || exit 1

# Get high-res name from config
HIGH_RES_NAME=$(get_high_res_name)
echo "Using high resolution scan: ${HIGH_RES_NAME}"

# Pull and run Mask3D Docker container
echo "Pulling Mask3D Docker image..."
docker pull rupalsaxena/mask3d_docker:latest

echo "Running Mask3D processing..."

# Rename the point cloud PLY file from pcd to mesh, since mesh is required for 3D Mask 
mv "${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}/pcd.ply" "${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}/mesh.ply"

# First remove any existing container with the same name
docker rm -f mask3d_docker 2>/dev/null || true

# Run Mask3D in it's docker environment
docker run --gpus all -it \
    --name mask3d_docker \
    -v "${LSARP_DIR}:${LSARP_DIR}" \
    -e LSARP="${LSARP_DIR}" \
    -w "${LSARP_DIR}/source/Mask3D" \
    rupalsaxena/mask3d_docker:latest \
    -c "python mask3d.py --seed 42 --workspace ${LSARP_DIR}/data/prescans/${HIGH_RES_NAME} && \
                  chmod -R 777 ${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}"


# Check if Docker command succeeded
if [ $? -ne 0 ]; then
    echo "Error: Docker command failed"
    exit 1
fi

# Copying files (these file locations are expected by the SpotLight repo)
echo "Copying files..."
cp "${LSARP_DIR}/data/aligned_point_clouds/${HIGH_RES_NAME}/pose/icp_tform_ground.txt" "${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}/icp_tform_ground.txt"
cp "${LSARP_DIR}/mask3d_label_mapping.csv" "${LSARP_DIR}/data/prescans/${HIGH_RES_NAME}/mask3d_label_mapping.csv"

echo "Mask3D setup and processing complete!" 