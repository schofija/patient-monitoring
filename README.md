
# patient-monitoring
###### Jack Schofield, Null Atwood, Travis Hudson, Rohan B Ballapragada, Zachary Taylor, Jet Ittihrit, Aditya Raj

[![ubuntu20][ubuntu20-badge]][ubuntu20]
[![foxy][foxy-badge]][foxy]
[![Official Website](https://img.shields.io/badge/Official%20Website-pmr--osu-blue?style=flat&logo=world&logoColor=white)](https://pmr-osu.github.io/)
[![License](https://img.shields.io/github/license/schofija/patient-monitoring?style=flat-square)](LICENSE)

ROS2 packages for the patient-monitoring robot. For the patient-monitoring Android application, go to [**pmr-app-test**]( https://github.com/JetLiTheQT/pmrtest).
### Example GIF
![Alt Text: Demonstration of fall_detection ROS2 node](https://github.com/schofija/patient-monitoring/blob/master/fall.gif)
## Table of Contents

 - [System Diagram](#system-diagram)
 - [Requirements](#requirements)
 - [Installation and Setup](#installation)
 - [Usage](#usage) 
	 - [Starting the TurtleBot3 and RealSense™ nodes](#tb_rs_nodes)
	 - [Starting the patient-monitoring nodes](#pmr_nodes)
- [Research](#research)
- [Legacy](#legacy)

## System Diagram
![Alt Text: Flow chart diagram of patient-monitoring system](https://github.com/schofija/patient-monitoring/blob/master/docs/system-diagram-pmr-dark.png)

## Requirements
#### Hardware:
+ TurtleBot3 Waffle
+ Intel® RealSense™ Depth Camera

*Note: The patient-monitoring nodes were tested using an Intel RealSense™ D405, but should function with any camera compatible with the Intel® RealSense™ ROS2 wrapper. Please inform us of any issues that arise using different camera models.*

#### Software:
+ Ubuntu 20.04 ([link](https://releases.ubuntu.com/focal/))
+ ROS2 Foxy ([link](https://docs.ros.org/en/foxy/Installation.html))
+ Intel® RealSense™ SDK 2.0 ([link](https://github.com/IntelRealSense/librealsense/blob/master/doc/installation.md))
+ Intel® RealSense™ ROS2 wrapper ([link](https://github.com/IntelRealSense/realsense-ros))

## Installation
Before continuing, please install [Ubuntu 20.04](https://releases.ubuntu.com/focal/) on the Turtlebot3's onboard computer.
If you plan on using a remote PC, it will require Ubuntu 20.04 (and a ROS2 foxy install) as well. This guide will not cover setting up a remote PC on ROS2.

#### Step 1: Install ROS2 Foxy and TurtleBot3 Packages

The <a href="https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/">TurtleBot3 Quick Start Guide (Section 3)</a> is a good resource for setting up our ROS2 environment. Please refer to it (and make sure you are following the steps for "Foxy").  

In our usage, we replaced the onboard Raspberry Pi that ships with the Turtlebot3 with an Intel® NUC to allow for real-time image processing onboard the robot. If you are planning to do something similar, please see the annotated notes below:

 - [Section 3. 1. PC Setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#pc-setup) can be applied to the onboard PC, as we will require a working ROS2 installation to run nodes directly on the robot. 
	 - [Section 3. 1. 6. Network Configuration](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/#network-configuration) can be skipped, although we recommend completing it. Even if you want your image processing nodes to run directly on the robot, it is still useful to have a remote PC to see system diagnostics.
- [Section 3. 2. SBC Setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/) can be entirely skipped if you are not using a Raspberry Pi, except for:
	- [Section 3. 2. 6. Configure the Raspberry Pi](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#configure-the-raspberry-pi) for those using a remote PC setup (the Raspberry Pi correspondsROS_DISTRO=<YOUR_SYSTEM_ROS_DISTRO>  # set your ROS_DISTRO: humble, galactic, foxy
source /opt/ros/$ROS_DISTRO/setup.bash
cd ~/ros2_ws
. install/local_setup.bash to the onboard PC, in our case the NUC).
	- [Section 3. 2. 7. NEW LDS-02 Configuration](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#new-lds-02-configuration-4) if you are using an LDS-02 (2022+ TurtleBot3 models). ROS2 packages default to the older LDS-01.
- [Section 3. 3. OpenCR Setup](https://emanual.robotis.com/docs/en/platform/turtlebot3/opencr_setup/#opencr-setup) can be followed, but we recommend attempting to bring-up the robot first and seeing if the current firmware on the OpenCR board is functional.
- [Section 3. 5. 1. Bringup TurtleBot3](https://emanual.robotis.com/docs/en/platform/turtlebot3/bringup/) demonstrates run basic TurtleBot3 programs and test basic operation. 
	- You do not need to follow Step 1 of *3. 5. 1.* if you are running onboard the robot. 
	- Step 2 of *3. 5. 1.* recommends running ``export TURTLEBOT3_MODEL=burger``. You will have to run this command every terminal emulator instance. Instead, we recommend ``export TURTLEBOT3_MODEL=burger >> ~/.bashrc `` to save this environmental variable.
- [Section 3. 6. 1. Teleoperation](https://emanual.robotis.com/docs/en/platform/turtlebot3/basic_operation/#teleoperation) is a good way to test functionality of your robot past simply running the ``turtlebot3_bringup`` node. If your robot does not respond to teleoperation commands, we've documented some common issues we ran into below:
	- Ensure the 12V power and the USB to the OpenCR board is plugged in. If only the USB is connected, ROS2 can successfully communicate with the board, but the board cannot send movements to the wheels. ROS2 will not display any error messages if this occurs.
	- **Ensure the USB port the OpenCR is plugged into has appropriate privileges (/dev/ttyACM0 by default). ``chmod 777 /dev/ttyACM0``. This is also a common issue with the LDS-01/02 (/dev/ttyUSB0 by default)**
</details>

#### Step 2: Install the Intel® RealSense™ SDK 2.0 and ROS2 Wrapper for for Intel® RealSense™ Devices
The [realsense-ros]([https://github.com/IntelRealSense/realsense-ros](https://github.com/IntelRealSense/realsense-ros#installation)) repository's README contains installation instructions for the Intel® RealSense™ SDK 2.0 and ROS2 Wrapper.

#### Step 3: Install the patient-monitoring ROS2 package
 - Create a ROS2 workspace (this should have been completed in Step 2).
 ```bash
 mkdir -p ~/ros2_ws/
 cd ~/ros2_ws/
```
 - Clone the patient-monitoring package and copy the ``/src`` directory to your ROS2 workspace.
 ```bash
 git clone https://github.com/schofija/patient-monitoring
 cp -r patient-monitoring/src/ ~/ros2_ws/
```
- Install dependencies
```bash
sudo apt-get install python3-rosdep -y
sudo rosdep init # "sudo rosdep init --include-eol-distros" for Eloquent and earlier
rosdep update # "sudo rosdep update --include-eol-distros" for Eloquent and earlier
rosdep install -i --from-path src --rosdistro $ROS_DISTRO --skip-keys=librealsense2 -y
```
- Build patient_monitoring and my_robot_interfaces package
```bash
colcon build --packages-select patient_monitoring && colcon build --packages-select my_robot_interfaces
```
- Source environment (do this in a different terminal than the one you ran ``colcon build`` in to avoid weird errors)
```bash
cd ~/ros2_was
. install/setup.bash
```

## Usage

<h3 id="tb_rs_nodes">Turtlebot3 and RealSense™ Nodes</h3>

The ``turtlebot3_bringup`` and ``realsense2_camera_node`` nodes must be running for the ``patient_monitoring`` nodes to function.
Option 1: Run the ``robot_launch`` launchfile
```bash
cd ~/ros2_ws/src/launch
ros2 launch robot_launch.py
```
Option 2: Manually launch nodes
```bash
ros2 launch turtlebot3_bringup robot.launch.py
ros2 launch realsense2_camera rs_launch.py depth_module.profile:=1280x720x30 align_depth.enable:=true
```
Please ensure your environment is sourced to avoid 'package not found' errors.
<h3 id="pmr_nodes">patient_monitoring Nodes</h3>

- **patient_watcher**:
```bash
ros2 run patient_monitoring patient_watcher
```
Reads camera feed and detects humans on screen. Publishes position and landmark data (as well as other extrinsics) to the ``Person`` message, which can be used by other ROS2 nodes.

- **patient_watcher_yolo**:
```bash
ros2 run patient_monitoring patient_watcher_yolo
```
Alternative version of ``patient_watcher`` that utilizes yolov7-tiny. ***Currently incompatible with fall_detection***

- **patient_follower**:
```bash
ros2 run patient_monitoring patient_follower
```
Uses positioning from ``Person`` message. Attempts to keep human at the center of screen aswell and an optimal distance away from the robot.

- **fall_detection**: 
```bash
ros2 run patient_monitoring fall_detection
```
Uses landmark data from ``Person`` message to detect falling movements. Detected falls are pushed to a realtime database to display push notifications on the patient-monitoring android application ([pmr-app-test](https://github.com/JetLiTheQT/pmrtest))

## Research
<details>
  <summary>User Research</summary>
  
- [Persona LifeCycle Research](https://github.com/schofija/patient-monitoring/blob/master/docs/Persona_LifeCycle_Research.pdf)
</details>

<details>
  <summary>Abnormal Motion Research</summary>
  
- [Research article collection with annotations](https://github.com/schofija/patient-monitoring/blob/master/docs/Anormal_Motion_Research_Annotations.pdf)
</details>

## Legacy
- [**FortyFive-Robot_ws**](https://github.com/villanub2/FortyFive-Robot_ws) (2019-2020 Capstone) 

[ubuntu20-badge]: https://img.shields.io/badge/-UBUNTU%2020%2E04-blue?style=flat-square&logo=ubuntu&logoColor=white
[ubuntu20]: https://releases.ubuntu.com/focal/
[foxy-badge]: https://img.shields.io/badge/-FOXY-orange?style=flat-square&logo=ros
[foxy]: https://docs.ros.org/en/foxy/index.html
