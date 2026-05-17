# SonarSweep Dataset Preparation

This repository converts ROS bags recorded in OceanSim/IsaacSim into a folder-based dataset that can be used to train and test the SonarSweep model. Each synchronized timestamp is exported as one datapoint folder containing stereo camera images, depth maps, sonar images, camera pose, camera intrinsics, sonar intrinsics, and the camera-to-sonar extrinsic transform.

The expected raw data layout is:

```text
raw_dataset/
└── {sonar_type}/
    ├── dataset.bag
    ├── background.bag
    └── background/
        ├── {timestamp}.png
        └── ...
```

The processed output layout is:

```text
processed_dataset/
├── dataset.txt
└── {sonar_type}/
    ├── dataset_0/
    ├── dataset_1/
    └── ...
```

`processed_dataset/dataset.txt` contains one datapoint folder name per line:

```text
dataset_0
dataset_1
dataset_2
```

## Recorded Topics

The simulator should publish the following topics:

```text
/isaacsim/camera/camera_info
/isaacsim/camera/depth/image_raw
/isaacsim/camera/image_raw
/isaacsim/camera2/depth/image_raw
/isaacsim/camera2/image_raw
/isaacsim/imu/data
/isaacsim/sonar_rect_image
/tf
/tf_static
```

In the simulator setup, the left camera is colocated with the sonar. The right camera is used as the main camera frame, and the script saves `cam_right_pose.txt` plus `T_camright2sonar.txt`.

## Step 1: Record ROS Bags

Create a folder for the target sonar setting:

```bash
mkdir -p raw_dataset/vfov12hfov60
cd raw_dataset/vfov12hfov60
```

Record the main dataset bag:

```bash
rosparam set use_sim_time false
rosbag record -O dataset.bag \
  /isaacsim/camera/camera_info \
  /isaacsim/camera/depth/image_raw \
  /isaacsim/camera/image_raw \
  /isaacsim/camera2/depth/image_raw \
  /isaacsim/camera2/image_raw \
  /isaacsim/imu/data \
  /isaacsim/sonar_rect_image \
  /tf \
  /tf_static
```

Record a background sonar bag with no foreground objects:

```bash
rosbag record -O background.bag /isaacsim/sonar_rect_image
```

## Step 2: Extract Background Images

The denoising pipeline uses an averaged empty-scene sonar background. Configure `base_dir` in `raw_dataset/retrieve_background_image.py`, then run the script from inside `raw_dataset`:

```bash
cd raw_dataset
python3 retrieve_background_image.py
```

For `base_dir = "./vfov12hfov60"`, this reads:

```text
raw_dataset/vfov12hfov60/background.bag
```

and writes:

```text
raw_dataset/vfov12hfov60/background/{timestamp}.png
```

## Step 3: Configure Sonar Parameters

Edit `config/hyperparam.py` before conversion. The important sonar parameters are:

```python
Min_range = 0.1
Max_range = 5.0
Range_res = 0.005
Img_height = 1000
Hori_fov = 60.0
Vert_fov = 12.0
Angular_res = 0.4
Img_width = 150
```

These values are also written into each datapoint as `sonar_intrinsic.txt`.

## Step 4: Convert Bags to Datapoints

Run the main converter from the repository root:

```bash
python3 rosbag2folders.py --sonar_type vfov12hfov60
```

By default, this reads:

```text
raw_dataset/vfov12hfov60/*.bag
```

excluding `background.bag`, and writes:

```text
processed_dataset/vfov12hfov60/{bag_name}_{index}/
```

For example:

```text
processed_dataset/vfov12hfov60/dataset_0/
processed_dataset/vfov12hfov60/dataset_1/
```

It also writes:

```text
processed_dataset/dataset.txt
```

with one datapoint folder name per line.

You can override the input and output directories:

```bash
python3 rosbag2folders.py \
  --input_dir raw_dataset/vfov12hfov60 \
  --output_dir processed_dataset/vfov12hfov60
```

## Datapoint Contents

Each datapoint folder contains:

```text
cam_intrinsic.txt
cam_left.png
cam_right.png
depth_left.npy
depth_left_visualize.png
depth_right.npy
depth_right_visualize.png
cam_right_pose.txt
T_camright2sonar.txt
sonar_intrinsic.txt
sonar_rect.png
sonar.png
sonar_rect_denoise.png
sonar_denoise.png
```

### Camera Data

`cam_intrinsic.txt` stores the original camera intrinsic matrix.

`cam_right_pose.txt` stores the right camera pose in the world frame.

`T_camright2sonar.txt` stores the transform from the right camera frame to the sonar frame. The transform is expressed in the camera coordinate convention, where the z-axis points forward and the x-axis points right.

### Sonar Data

The raw simulator sonar image covers `Min_range` to `Max_range`. The converter pads the image so `sonar_rect.png` represents `0` to `Max_range`.

`sonar.png` is a Cartesian visualization generated from the padded rectangular sonar image.

`sonar_rect_denoise.png` is the denoised rectangular sonar image. `sonar_denoise.png` is its Cartesian visualization.

The denoiser uses the averaged background image from `raw_dataset/{sonar_type}/background` and the SCUNet model at:

```text
denoise/model/scunet_gray_25.pth
```

## Step 5: Crop and Enhance Camera Images

After conversion, crop the camera and depth images to the sonar FOV and generate grayscale CLAHE-enhanced camera images:

```bash
python3 crop_and_enhance_images.py --sonar_type vfov12hfov60
```

This adds files such as:

```text
cropped_cam_intrinsic.txt
cropped_cam_left.png
cropped_cam_right.png
cropped_depth_left.npy
cropped_depth_right.npy
enhanced_gray_cam_left.png
enhanced_gray_cam_right.png
```

To regenerate existing cropped outputs:

```bash
python3 crop_and_enhance_images.py --sonar_type vfov12hfov60 --overwrite
```

You can also specify the processed dataset directory directly:

```bash
python3 crop_and_enhance_images.py --root_dir processed_dataset/vfov12hfov60
```

## Visualization

To inspect a recorded bag:

```bash
rosparam set use_sim_time true
rviz -d rviz/visualizer.rviz
rosbag play -l -r 10 raw_dataset/vfov12hfov60/dataset.bag
```

Optional helper scripts:

```bash
python3 publish_pcd.py
python3 publish_converted_sonar.py
```

## Useful Commands

Filter a short bag segment:

```bash
rosbag filter dataset.bag 1.bag "t.to_sec() <= 1752220844.72 + 65.0"
```

Play a bag faster:

```bash
rosbag play -r 10 dataset.bag
```

Copy processed data to another machine:

```bash
scp -r processed_dataset/vfov12hfov60 user@host:/path/to/data/
scp processed_dataset/dataset.txt user@host:/path/to/data/
```
