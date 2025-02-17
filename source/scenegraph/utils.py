import open3d as o3d
import glob, os, cv2, json
import numpy as np
from projectaria_tools.core import data_provider, calibration
import imageio
from moviepy.editor import VideoFileClip, clips_array

def get_all_images(scan_dir, name):
    vrs_files = glob.glob(os.path.join(scan_dir + name, '*.vrs'))
    assert vrs_files is not None, "No vrs files found in directory"
    for vrs_file in vrs_files:
        provider = data_provider.create_vrs_data_provider(vrs_file)
        assert provider is not None, "Cannot open file"

        deliver_option = provider.get_default_deliver_queued_options()

        deliver_option.deactivate_stream_all()
        camera_label = "camera-rgb"
        stream_id = provider.get_stream_id_from_label(camera_label)
        calib = provider.get_device_calibration().get_camera_calib(camera_label)
        w, h = calib.get_image_size()
        pinhole = calibration.get_linear_camera_calibration(w, h, calib.get_focal_lengths()[0])

        image_dir = os.path.join(scan_dir, os.path.basename(vrs_file)[:-4] + "_images")
        os.makedirs(image_dir, exist_ok=True)

        for i in range(provider.get_num_data(stream_id)):
            image_data = provider.get_image_data_by_index(stream_id, i)
            img = image_data[0].to_numpy_array()
            undistorted_image = calibration.distort_by_calibration(img, pinhole, calib)
            aruco_image = cv2.cvtColor(undistorted_image, cv2.COLOR_RGB2BGR)
            aruco_image = cv2.rotate(aruco_image, cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite(os.path.join(image_dir, f"frame_{i:05}.jpg"), aruco_image)


def mask3d_labels(label_path):
    with open(label_path, 'r') as file:
        lines = file.readlines()
    
    file_paths = []
    values = []
    confidences = []

    for line in lines:
        parts = line.split()
        file_paths.append(parts[0])
        values.append(int(parts[1]))
        confidences.append(float(parts[2]))
    
    base_dir = os.path.dirname(os.path.abspath(label_path))
    first_pred_mask_path = os.path.join(base_dir, file_paths[0])

    with open(first_pred_mask_path, 'r') as file:
        num_lines = len(file.readlines())

    mask3d_labels = np.zeros(num_lines, dtype=int)

    for i, relative_path in enumerate(file_paths):
        value = values[i]
        file_path = os.path.join(base_dir, relative_path)
        with open(file_path, 'r') as file:
            for j, line in enumerate(file):
                if line.strip() == '1':
                    mask3d_labels[j] = value

    output_path = os.path.join(base_dir, 'mask3d_labels.txt')
    with open(output_path, 'w') as file:
        for label in mask3d_labels:
            file.write(f"{label}\n")
            
def compute_pose(points: np.ndarray, centroid: np.ndarray) -> np.ndarray:
    """
    Computes the pose of an object given its 3D points and centroid.

    This function calculates the orientation and position of an object in 3D space by aligning 
    the principal axes of its points to the global axes using PCA. It returns
    a 4x4 transformation matrix representing the object's pose.

    :param points: Array of 3D points representing the object's geometry.
    :param centroid: The centroid of the object as a 3D point.
    :return: 4x4 numpy array representing the object's pose in the world frame.
    """
    points_centered = points - centroid
    covariance_matrix = np.cov(points_centered, rowvar=False)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance_matrix)
    sorted_idx = np.argsort(eigenvalues)[::-1]
    eigenvectors = eigenvectors[:, sorted_idx]
    R = eigenvectors
    if np.linalg.det(R) < 0:
        R[:, -1] *= -1
    object_pose = np.eye(4)
    object_pose[:3, :3] = R
    object_pose[:3, 3] = centroid
    return object_pose

def parse_json(file_path: str) -> tuple[np.ndarray, np.ndarray]:
    """
    Parses a JSON file to extract camera intrinsics and pose.

    :param file_path: Path to the JSON file containing camera parameters.
    :return: Tuple containing:
        - intrinsics: 3x3 numpy array of camera intrinsic parameters.
        - camera_pose: 4x4 numpy array of the camera pose in the AR frame.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)
    
    intrinsics = np.array(data["intrinsics"]).reshape(3, 3)
    # projection_matrix = np.array(data["projectionMatrix"]).reshape(4, 4)
    camera_pose = np.array(data["cameraPoseARFrame"]).reshape(4, 4)
    return intrinsics, camera_pose

def parse_txt(file_path: str) -> np.ndarray:
    """
    Parses a text file to extract extrinsic camera parameters.

    :param file_path: Path to the text file containing extrinsic parameters.
    :return: Numpy array representing the extrinsic parameters.
    """
    with open(file_path, 'r') as file:
        extrinsics = file.readlines()
        extrinsics = [parts.split() for parts in extrinsics]
        extrinsics = np.array(extrinsics).astype(float)

    return extrinsics
    

def vis_detections(coords: np.ndarray, color: list = [0, 0, 1]) -> list[o3d.geometry.TriangleMesh]:
    """
    Creates a list of spheres to visualize detection points in 3D space.

    This function takes an array of 3D coordinates and generates a list of sphere geometries 
    at those points, each painted in the specified color for visualization in Open3D.

    :param coords: Array of 3D coordinates to visualize as spheres.
    :param color: RGB color for the spheres, defaults to blue.
    :return: List of Open3D TriangleMesh objects representing spheres at each detection point.
    """
    coords = np.asarray(coords)
    spheres = []

    if len(coords) == 0:
        return spheres
    
    if len(coords.shape) == 1:
        coords = np.array([coords])
    
    for coord in coords:
        sphere = o3d.geometry.TriangleMesh.create_sphere(radius=0.02)
        sphere.paint_uniform_color(color)
        sphere.compute_vertex_normals()
        sphere.translate(coord)
        spheres += [sphere]
    return spheres

def calculate_center(bb: list[float]) -> list[float]:
    """
    Calculates the center point of a bounding box.

    :param bb: Bounding box coordinates as [x_min, y_min, x_max, y_max].
    :return: List of x and y coordinates for the center point.
    """
    return [(bb[0] + bb[2])/2, (bb[1] + bb[3])/2]

def filter_object(obj_dets: np.ndarray, hand_dets: np.ndarray) -> list[int]:
    """
    Matches hands in contact with the nearest detected object based on offset.

    This function takes in arrays of object and hand detections, using the hand offset vector 
    to determine the closest object to each hand. It returns a list of object indices matched 
    to each hand.

    :param obj_dets: Array of object detections with bounding boxes.
    :param hand_dets: Array of hand detections with bounding boxes and contact status.
    :return: List of indices of objects closest to each hand in contact.
    """
    object_cc_list = [] # object center list
    for j in range(obj_dets.shape[0]):
        object_cc_list.append(calculate_center(obj_dets[j,:4]))
    object_cc_list = np.array(object_cc_list)

    img_obj_id = [] # matching list
    for i in range(hand_dets.shape[0]):
        if hand_dets[i, 5] <= 0: # if hand is non-contact
            img_obj_id.append(-1)
            continue
        else: # hand is in-contact
            hand_cc = np.array(calculate_center(hand_dets[i,:4])) # hand center points
            # caculates, using the hand offset vector, which object is the closest to this object
            point_cc = np.array([(hand_cc[0]+hand_dets[i,6]*10000*hand_dets[i,7]), (hand_cc[1]+hand_dets[i,6]*10000*hand_dets[i,8])]) # extended points (hand center + offset)
            dist = np.sum((object_cc_list - point_cc)**2,axis=1)
            dist_min = np.argmin(dist) # find the nearest 
            img_obj_id.append(dist_min)
        
    return img_obj_id


def crop_image(image: np.ndarray) -> np.ndarray:
    """
    Crops the input image by a margin of 8% on each side. Default behaviour
    to get rid off the black borders due to fish-eye distortion.

    :param image: Input image as a numpy array.
    :return: Cropped image as a numpy array.
    """
    height, width = image.shape[:2]
    
    ratio=0.08
    margin_x = int(width * (ratio) / 2)
    margin_y = int(height * (ratio) / 2)

    # Define the cropping rectangle
    x_start = margin_x
    x_end = width - margin_x
    y_start = margin_y
    y_end = height - margin_y

    # Crop the image
    cropped_image = image[y_start:y_end, x_start:x_end]

    return cropped_image

def create_video(image_dir: str, output_path: str = "output.mp4", fps: int = 30) -> None:
    """
    Creates a video from a directory of images. This function takes images from the specified
    directory, sorted by filename, and compiles them into a video at the specified frame rate.

    :param image_dir: Path to the directory containing image files.
    :param output_path: Path to save the generated video file. Defaults to "output.mp4".
    :param fps: Frames per second for the video. Defaults to 30.
    :return: None. Saves the video to the specified output path.
    """
    images = []
    for filename in sorted(glob.glob(os.path.join(image_dir, '*.jpg'))):
        images.append(filename)
    
    if len(images) == 0:
        for filename in sorted(glob.glob(os.path.join(image_dir, '*.png'))):
            images.append(filename)
    
    
    with imageio.get_writer(output_path, fps=fps) as writer:
        for i, image_file in enumerate(images):
            # Read each image
            image = imageio.imread(image_file)
            image = crop_image(image) # was only in SpotLight
            # Append the image to the video
            writer.append_data(image)

def save_camera_params(scene, path):
    # Save the current camera parameters to a file
    camera_params = scene.get_view_control().convert_to_pinhole_camera_parameters()
    with open(path, "w") as f:
        json.dump(camera_params.__dict__, f, default=lambda o: o.__dict__)

def load_camera_params(scene, path):
    # Load the camera parameters from a file
    with open(path, "r") as f:
        camera_params_dict = json.load(f)
    camera_params = o3d.camera.PinholeCameraParameters()
    camera_params.__dict__.update(camera_params_dict)
    scene.get_view_control().convert_from_pinhole_camera_parameters(camera_params)

def capture_image(scene_widget, file_path):
    image = scene_widget.scene.renderer.render_to_image()
    o3d.io.write_image(file_path, image)

def stitch_videos(video_path_1, video_path_2, new_file="combined_video.mp4"):
    video1 = VideoFileClip(video_path_1)
    video2 = VideoFileClip(video_path_2)

    # Ensure both videos have the same duration, fps, and dimensions
    # This should be already true according to your information

    # Combine the videos side by side
    final_video = clips_array([[video1, video2]])

    # Save the final video
    final_video.write_videofile(new_file)