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
        logging.info(f"Looking for prediction files in directory: {folder_path}")
    else:
        # If it's already a file, use it directly
        prediction_files = [folder_path]
        logging.info(f"Using prediction file: {folder_path}")
    
    # Initialize empty DataFrame to store combined results
    combined_df = pd.DataFrame(columns=["path_ending", "class_label", "confidence"])
    
    # Process each prediction file
    for pred_file in prediction_files:
        if not os.path.exists(pred_file):
            logging.debug(f"Prediction file not found at {pred_file}")
            continue
        
        try:
            # Read the file content first to clean it
            with open(pred_file, 'r') as f:
                lines = f.readlines()
            
            # Filter out empty lines and strip whitespace
            clean_lines = [line.strip() for line in lines if line.strip()]
            
            if not clean_lines:
                logging.debug(f"Prediction file {pred_file} is empty")
                continue
            
            # Create a DataFrame from the cleaned lines
            data = []
            for line in clean_lines:
                parts = line.split()
                if len(parts) >= 3:  # Ensure we have all three columns
                    data.append(parts[:3])  # Take only first 3 columns if there are more
            
            if not data:
                logging.debug(f"No valid data found in {pred_file}")
                continue
            
            # Create DataFrame for this file
            file_df = pd.DataFrame(data, columns=["path_ending", "class_label", "confidence"])
            
            # Convert class_label and confidence to appropriate types
            file_df["class_label"] = file_df["class_label"].astype(int)
            file_df["confidence"] = file_df["confidence"].astype(float)
            
            # Append to combined DataFrame
            combined_df = pd.concat([combined_df, file_df], ignore_index=True)
            
            logging.info(f"Successfully processed {pred_file} with {len(file_df)} entries")
            
        except Exception as e:
            logging.error(f"Error reading prediction file {pred_file}: {str(e)}")
            continue
    
    if combined_df.empty:
        logging.warning("No data was successfully loaded from any prediction files")
    else:
        # Log the first few rows to help with debugging
        logging.info(f"Combined {len(combined_df)} total entries from all prediction files")
        logging.debug(f"First few rows of combined DataFrame:\n{combined_df.head()}")
        logging.debug(f"Available class labels: {combined_df['class_label'].unique()}")
    
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

    # read the point cloud, select by indices specified in the file
    pc = o3d.io.read_point_cloud(point_cloud_path)

    good_points_idx = np.where(good_points_bool)[0]
    environment_cloud = pc.select_by_index(good_points_idx, invert=True)
    item_cloud = pc.select_by_index(good_points_idx)

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
