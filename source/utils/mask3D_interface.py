"""
Util functions for segmenting point clouds with Mask3D.
"""

from __future__ import annotations

from glob import glob
import logging
import os.path

import numpy as np

import open3d as o3d
import pandas as pd
from utils import recursive_config
from utils.importer import PointCloud
from utils.scannet_200_labels import CLASS_LABELS_200, VALID_CLASS_IDS_200
from utils.vis import generate_distinct_colors

# Set up logger
logger = logging.getLogger("main")

def is_valid_label(item: str) -> bool:
    """
    Check whether a label is valid (within the specified possible labels).
    """
    return item in CLASS_LABELS_200


def _get_list_of_items(folder_path: str) -> pd.DataFrame:
    """
    Read and combine multiple prediction files (predictions.txt, predictions_light_switches.txt,
    and predictions_drawers.txt) that include the output from Mask3D.
    Each file consists of 3 columns which specify the file that indexes the points 
    belonging to the object, its label, and the confidence. 
    This function outputs all the items/objects that have been detected in the point cloud.
    
    :param folder_path: path to the folder containing prediction files
    :return: pandas dataframe with the combined parsed data
    """
    # Check if folder_path is a directory or a file
    if os.path.isdir(folder_path):
        # If it's a directory, look for all prediction files
        prediction_files = [
            os.path.join(folder_path, "predictions.txt"),
            os.path.join(folder_path, "predictions_light_switches.txt"),
            os.path.join(folder_path, "predictions_drawers.txt")
        ]
        logger.info(f"Looking for prediction files in directory: {folder_path}")
    else:
        # If it's already a file, use it directly
        prediction_files = [folder_path]
        logger.info(f"Using prediction file: {folder_path}")
    
    # Initialize empty DataFrame to store combined results
    combined_df = pd.DataFrame(columns=["path_ending", "class_label", "confidence"])
    
    # Process each prediction file
    for pred_file in prediction_files:
        if not os.path.exists(pred_file):
            logger.debug(f"Prediction file not found at {pred_file}")
            continue
        
        try:
            # Read the file content first to clean it
            with open(pred_file, 'r') as f:
                lines = f.readlines()
            
            # Filter out empty lines and strip whitespace
            clean_lines = [line.strip() for line in lines if line.strip()]
            
            if not clean_lines:
                logger.debug(f"Prediction file {pred_file} is empty")
                continue
            
            # Create a DataFrame from the cleaned lines
            data = []
            for line in clean_lines:
                parts = line.split()
                if len(parts) >= 3:  # Ensure we have all three columns
                    data.append(parts[:3])  # Take only first 3 columns if there are more
            
            if not data:
                logger.debug(f"No valid data found in {pred_file}")
                continue
            
            # Create DataFrame for this file
            file_df = pd.DataFrame(data, columns=["path_ending", "class_label", "confidence"])
            
            # Convert class_label and confidence to appropriate types
            file_df["class_label"] = file_df["class_label"].astype(int)
            file_df["confidence"] = file_df["confidence"].astype(float)
            
            # Append to combined DataFrame
            combined_df = pd.concat([combined_df, file_df], ignore_index=True)
            
            logger.info(f"Successfully processed {pred_file} with {len(file_df)} entries")
            
        except Exception as e:
            logger.error(f"Error reading prediction file {pred_file}: {str(e)}")
            continue
    
    if combined_df.empty:
        logger.warning("No data was successfully loaded from any prediction files")
    else:
        # Log the first few rows to help with debugging
        logger.info(f"Combined {len(combined_df)} total entries from all prediction files")
        logger.debug(f"First few rows of combined DataFrame:\n{combined_df.head()}")
        logger.debug(f"Available class labels: {combined_df['class_label'].unique()}")
    
    return combined_df


def get_coordinates_from_item(
    item: str,
    folder_path: str | bytes,
    point_cloud_path: str | bytes,
    index: int = 0,
) -> (PointCloud, PointCloud):
    """
    Given an item description, we extract all points that are part of this item.
    Returns two point clouds, one representing the item, the other the rest.
    :param item: name of item to extract
    :param folder_path: base folder path of the Mask3D output
    :param point_cloud_path: path for the point cloud
    :param index: if there are multiple objects for a given label, which one to focus on
    """
    logger.info(f"Getting coordinates from item: {item}")
    if not is_valid_label(item):
        raise ValueError(f"Item {item} is not a valid label")
    # convert string label to numeric
    idx = CLASS_LABELS_200.index(item)
    label = VALID_CLASS_IDS_200[idx]

    df = _get_list_of_items(str(folder_path))
    # get all entries for our item label
    entries = df[df["class_label"] == label]
    if index > len(entries) or index < (-len(entries) + 1):
        index = 0
    # get "index" numbered object
    entry = entries.iloc[index]
    path_ending = entry["path_ending"]

    # get the mask of the item
    mask_file_path = os.path.join(folder_path, path_ending)
    with open(mask_file_path, "r", encoding="UTF-8") as file:
        lines = file.readlines()
    good_points_bool = np.asarray([bool(int(line)) for line in lines])

    # read the point cloud
    pc = o3d.io.read_point_cloud(point_cloud_path)
    
    # IMPORTANT: Mask3D uses voxel quantization on the point cloud before generating masks 
    # We need to replicate that process here to ensure mask indices match point cloud indices
    points = np.asarray(pc.points)
    
    # Apply the same voxel size (0.02) that Mask3D uses for quantization
    voxel_size = 0.02
    logger.info(f"Original point cloud has {len(points)} points")
    
    # This creates a quantized version of the point cloud using the same method as Mask3D
    pc_quantized = pc.voxel_down_sample(voxel_size)
    logger.info(f"Quantized point cloud has {len(pc_quantized.points)} points")
    
    # Check if the sizes match
    if len(pc_quantized.points) != len(good_points_bool):
        logger.warning(f"Size mismatch! Mask has {len(good_points_bool)} entries but quantized point cloud has {len(pc_quantized.points)} points")
        # If sizes are close, we can try to use only the valid indices
        if len(good_points_bool) > len(pc_quantized.points):
            logger.warning("Truncating mask to match point cloud size")
            good_points_bool = good_points_bool[:len(pc_quantized.points)]
        else:
            logger.warning("Padding mask with False values to match point cloud size")
            padding = np.zeros(len(pc_quantized.points) - len(good_points_bool), dtype=bool)
            good_points_bool = np.concatenate([good_points_bool, padding])
    
    # Get indices of mask points
    good_points_idx = np.where(good_points_bool)[0]
    
    # Use the quantized point cloud instead of the original
    environment_cloud = pc_quantized.select_by_index(good_points_idx, invert=True)
    item_cloud = pc_quantized.select_by_index(good_points_idx)

    return item_cloud, environment_cloud


def get_all_item_point_clouds(
    folder_path: str | bytes,
    point_cloud_path: str | bytes,
) -> list[PointCloud]:
    """
    TODO
    """
    df = _get_list_of_items(str(folder_path))
    pcd = o3d.io.read_point_cloud(point_cloud_path)
    colors = generate_distinct_colors(len(df))

    pcds = []
    for row in df.iterrows():
        idx, (path_ending, class_label, confidence) = row
        mask_file_path = os.path.join(folder_path, path_ending)
        with open(mask_file_path, "r", encoding="UTF-8") as file:
            lines = file.readlines()
        good_points_bool = np.asarray([bool(int(line)) for line in lines])
        current_pcd = pcd.select_by_index(np.where(good_points_bool)[0])
        current_pcd.paint_uniform_color(colors[idx])
        pcds.append(current_pcd)
    return pcds


def _test() -> None:
    config = recursive_config.Config()

    mask_path = config.get_subpath("masks")
    ending = config["pre_scanned_graphs"]["high_res"]
    mask_path = os.path.join(mask_path, ending)

    pc_path = config.get_subpath("aligned_point_clouds")
    ending = config["pre_scanned_graphs"]["high_res"]
    pc_path = os.path.join(str(pc_path), ending, "scene.ply")

    _get_list_of_items(mask_path)
    # res = get_all_item_point_clouds(mask_path, pc_path)
    # o3d.visualization.draw_geometries(res)
    # print(res)


if __name__ == "__main__":
    _test()
