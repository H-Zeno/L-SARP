<div align='center'>
<h2 align="center"> SpotLight: Robotic Scene Understanding through Interaction and Affordance Detection </h2>

<a href="">Tim Engelbracht</a><sup>1</sup>, <a href="https://scholar.google.com/citations?user=feJr7REAAAAJ&hl=en">René Zurbrügg</a><sup>1</sup>, <a href="https://people.inf.ethz.ch/marc.pollefeys/">Marc Pollefeys</a><sup>1,2</sup>, <a href="https://hermannblum.net/">Hermann Blum</a><sup>1,3</sup>, <a href="https://zuriabauer.com/">Zuria Bauer</a><sup>1</sup>

<sup>1</sup>ETH Zurich <sup>2</sup>Microsoft <sup>3</sup>Uni Bonn

![teaser](https://github.com/timengelbracht/SpotLightWebsite/blob/main/SpotLightLogo.png?raw=true)

</div>

[[Project Webpage](https://timengelbracht.github.io/SpotLight/)]

# Spot-Light

Spot-Light is a library and framework built on top of the [Spot-Compose repository](https://github.com/oliver-lemke/spot-compose) codebase to interact with Boston Dynamics' Spot robot. It enables processing point clouds, performing robotic tasks, and updating scene graphs.

---

# Dataset

The dataset used in the paper is available at [Roboflow](https://universe.roboflow.com/timengelbracht/spotlight-light-switch-dataset)

## Setup Instructions

Heads up: this setup is a bit involved, since we will explain not only some example code, but also the enttire setup, including acquiring the point clouds, aligning them, setting up the docker tools and scene graphs and so on and so forth. So bear with me here. In case u run into issues, don't hesitate to leave an issue or just send me an email :)

### Define SpotLight Path

Set the environment variable `LSARP` to the path of the L-SARP folder. Just to make sure we're all on the same page ... I mean path ;) Example:

```bash
export LSARP=/home/cvg-robotics/zeno_ws/L-SARP/
```

### Virtual Environment

1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install required packages:
   ```bash
   pip install -e .
   ```

---

## Point Cloud Capturing and Alignment

We need two point clouds for the scene understanding:

### Prerequisites

1. Update the configuration file (`configs/config.yaml`):
   ```yaml
   pre_scanned_graphs:
     low_res: '<low_res_name>' # Name for the low-resolution scan (acquired during autowalk with spot)
     high_res: '<high_res_name>' # Name for the high-resolution scan (acquired through 3D lidar scan with e.g. Ipad)
   ```

### Low-Resolution Point Cloud

1. Position Spot in front of the AprilTag and start the autowalk (control spot to walk around the office, point cloud gets captured in the meantime)
2. Zip the resulting data and unzip it into `$LSARP/data/autowalk/`
   - The point cloud should be at: `$LSARP/data/autowalk/<low_res_name>.walk/point_cloud.ply`

### High-Resolution Point Cloud

1. Use the 3D Scanner App (iOS) to capture the point cloud
   - Ensure the fiducial is visible during the scan
2. Export (with the 3D Scanner APP):
   - **All Data** as a zip file.
   - **Point Cloud/PLY** file with "High Density" enabled and "Z axis up" disabled.
3. Unzip the "All Data" into $LSARP/data/prescans/ and copy the PLY file into this folder as well.

### Running the Alignment Script

```bash
# Make it executable (only need to do this once)
chmod +x setup_scripts/allign_pointclouds.sh

# Run the script
./setup_scripts/allign_pointclouds.sh
```

The script will:

1. Read scan names from config.yaml
2. Create required directories if needed
3. Copies the low resolution point cloud data to $LSARP/data/point_clouds/<low_res_name>.ply
4. Copies the high resolution point cloud data to $LSARP/data/prescans/<high_res_name>/pcd.ply
5. Run point cloud alignment
6. Save aligned results to `$LSARP/data/aligned_point_clouds/`

---

## Scene Graph Setup

```bash
# Make the script executable
chmod +x setup_scripts/setup_mask3d.sh

# Run the script
./setup_scripts/setup_mask3d.sh
```

The script will:

1. Set up Mask3D repository and download the checkpoint that we want to use
2. Pull and run the Mask3D Docker container
3. Renames the high-resolution point cloud (pcd.ply) to mesh.ply (required for Mask3D)
4. Processes mesh.ply and create a semantic instance segmentated mesh (mesh_labeled.ply)
5. Copies certain files to the correct location for further processing by the repo

---

## Docker Dependencies

- **YoloDrawer**: Required for drawer interactions in the scene graph.
- **OpenMask**: Required for `search_all_drawers.py`.
- **GraspNet**: Required for `gs_grasp.py` and openmask feature extraction

Refer to the [Spot-Compose repository](https://github.com/oliver-lemke/spot-compose) documentation for Docker downlaod and setup.

**NOTE** If u plan on using the graspnet Docker, make sure to run this one first, and the other containers afterwards! Othwerwise the container won't work...No idea why

---

## Update python path

Just to make sure we don't run into pathing/ import issues

```bash
export PYTHONPATH=$LSARP:$LSARP/source:\$PYTHONPATH
```

## Extracting OpenMask Features

In case you are using the OpenMask functionalities:

```bash
python3 $LSARP/source/utils/openmask_interface.py
```

## Configuration File

Create a hidden `.environment.yaml` file to store sensitive configurations:

```yaml
spot:
  wifi-network: <password>
  spot_admin_console_username: <username>
  spot_admin_console_password: <password>
  wifi_default_address: 192.168.50.1
  wifi_password: <password>
  nuc1_user_password: <password>
api:
  openai:
    key: <api key>
```

---

### Networking

## Workstation Networking

This is an over view for workstation networking. Again, this information can also be found in the [Spot-Compose repository](https://github.com/oliver-lemke/spot-compose).

On the workstation run

```bash
   $LSARP/shells/ubuntu_routing.sh
```

(or $LSARP/shells/mac_routing.sh depending on your workstation operating system).

**Short Explanation for the curious**: This shell script has only a single line of code: sudo ip route add 192.168.50.0/24 via <local NUC IP>

In this command:

    192.168.50.0/24 represents the subnet for Spot.
    <local NUC IP> is the IP address of your local NUC device.

If you're working with multiple Spot robots, each Spot must be assigned a distinct IP address within the subnet (e.g., 192.168.50.1, 192.168.50.2, etc.). In such cases, the routing needs to be adapted for each Spot. For example:

```bash
sudo ip route add 192.168.50.2 via <local NUC IP>
```

## NUC Networking

First, ssh into the NUC, followed by running ./robot_routing.sh to configure the NUC as a network bridge. Note that this might also need to be adapted based on your robot and workstation IPs.

## Example Scripts

### Light Switch Demo

```bash
python3 -m source/scripts/my_robot_scripts/light_switch_demo.py
```

### Scene Graph Update

```bash
python3 -m source/scripts/my_robot_scripts/scene_graph_demo.py
```

### Swing Drawer Interaction

```bash
python3 -m source/scripts/my_robot_scripts/search_swing_drawer_dynamic.py
```

---

# BibTeX :pray:

```
@misc{engelbracht2024spotlightroboticsceneunderstanding,
      title={SpotLight: Robotic Scene Understanding through Interaction and Affordance Detection},
      author={Tim Engelbracht and René Zurbrügg and Marc Pollefeys and Hermann Blum and Zuria Bauer},
      year={2024},
      eprint={2409.11870},
      archivePrefix={arXiv},
      primaryClass={cs.RO},
      url={https://arxiv.org/abs/2409.11870},
}
```

## License

This project is licensed under the MIT License.
