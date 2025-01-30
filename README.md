
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
Set the environment variable `SPOTLIGHT` to the path of the Spot-Light folder. Just to make sure we're all on the same page ... I mean path ;) Example:
```bash
export SPOTLIGHT=/home/cvg-robotics/tim_ws/Spot-Light/
```

### Virtual Environment
1. Create a virtual environment:
   ```bash
   python3 -m venv venv
   source venv/bin/activate
   ```
2. Install required packages:
   ```bash
   pip install -r requirements.txt
   ```

---

## Point Cloud Capturing
We will need two pint clouds here.

### Low-Resolution Point Cloud
1. Position Spot in front of the AprilTag and start the autowalk.
2. Zip the resulting data and unzip it into $SPOTLIGHT/data/autowalk/.
3. Update the configuration file:
   - Fill in the name of the unzipped folder `<low_res_name>` under $SPOTLIGHT/pre_scanned_graphs/low_res.
4. Copy the low-resolution point cloud:
   ```bash
   cp $SPOTLIGHT/data/autowalk/<low_res_name>.walk/point_cloud.ply $SPOTLIGHT/data/point_clouds/<low_res_name>.ply
   ```

### High-Resolution Point Cloud
1. Use the 3D Scanner App (iOS) to capture the point cloud. Ensure the fiducial is visible during the scan.
2. Export:
   - **All Data** as a zip file.
   - **Point Cloud/PLY** with "High Density" enabled and "Z axis up" disabled.
3. Unzip the "All Data" into $SPOTLIGHT/data/prescans/ and rename the folder `<high_res_name>`.
4. Rename and copy the point cloud:
   ```bash
   cp <exported_point_cloud>.ply $SPOTLIGHT/data/prescans/<high_res_name>/pcd.ply
   ```

5. Update the configuration file:
   - Fill in `<high_res_name>` under `pre_scanned_graphs/high_res`.
   - Fill in `<low_res_name>` under `pre_scanned_graphs/low_res`.

---

## Aligning Point Clouds
Run the alignment script (ensure names in the config are updated beforehand):
```bash
python3 $SPOTLIGHT/source/scripts/point_cloud_scripts/full_align.py
```
The aligned point clouds will be written to:
```
$SPOTLIGHT/data/aligned_point_clouds
```

---

## Scene Graph Setup

1. Clone and set up Mask3D:
   ```bash
   cd $SPOTLIGHT/source/
   git clone https://github.com/behretj/Mask3D.git
   mkdir Mask3D/checkpoints
   cd Mask3D/checkpoints
   wget "https://zenodo.org/records/10422707/files/mask3d_scannet200_demo.ckpt"
   cd ../../..
   ```

2. Run the Mask3D Docker container:
   ```bash
   docker pull rupalsaxena/mask3d_docker:latest
   docker run --gpus all -it -v /home:/home -w $SPOTLIGHT/source/Mask3D rupalsaxena/mask3d_docker:latest
   ```

3. Inside the container, process the high-resolution point cloud:
   ```bash
   python mask3d.py --seed 42 --workspace $SPOTLIGHT/data/prescans/<high_res_name>
   chmod -R 777 $SPOTLIGHT/data/prescans/<high_res_name>
   ```

4. Update permissions and move processed files:
   ```bash
   cp $SPOTLIGHT/mask3d_label_mapping.csv $SPOTLIGHT/data/prescans/<high_res_name>/mask3d_label_mapping.csv
   cp $SPOTLIGHT/data/aligned_point_clouds/<high_res_name>/pose/icp_tform_ground.txt $SPOTLIGHT/data/prescans/<high_res_name>/icp_tform_ground.txt
   cp $SPOTLIGHT/data/prescans/<high_res_name>/pcd.ply $SPOTLIGHT/data/prescans/<high_res_name>/mesh.ply
   ```

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
export PYTHONPATH=$SPOTLIGHT:$SPOTLIGHT/source:\$PYTHONPATH
```

## Extracting OpenMask Features

In case you are using the OpenMask functionalities:
```bash
python3 $SPOTLIGHT/source/utils/openmask_interface.py
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
   $SPOTLIGHT/shells/ubuntu_routing.sh
```
(or $SPOTLIGHT/shells/mac_routing.sh depending on your workstation operating system).

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
