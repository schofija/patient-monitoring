
# patient-monitoring
###### Jack Schofield, Null Atwood, Travis Hudson, Rohan B Ballapragada, Zachary Taylor, Jet Ittihrit, Aditya Raj
[![Official Website](https://img.shields.io/badge/Official%20Website-pmr--osu-blue?style=flat&logo=world&logoColor=white)](https://pmr-osu.github.io/)
[![License](https://img.shields.io/github/license/schofija/patient-monitoring?style=flat-square)](LICENSE)
[![ubuntu20][ubuntu20-badge]][ubuntu20]
[![foxy][foxy-badge]][foxy]

ROS2 packages for the patient-monitoring robot. For the patient-monitoring Android application, go to [**pmr-app-test**]( https://github.com/JetLiTheQT/pmrtest).
### Example GIF
![Alt Text: Demonstration of fall_detection ROS2 node](https://github.com/schofija/patient-monitoring/blob/master/fall.gif)
## Table of Contents

 - [Requirements](#requirements)
 - [Installation and Setup](#installation)
 - [Usage](#usage) 
	 - [Starting the TurtleBot3 node](#start-turtlebot-node)
	 - [Starting the RealSense™ node](#start-realsense-camera-node)
	 - [Starting the patient-monitoring nodes](#start-patient--monitoring-nodes)

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
	- [Section 3. 2. 6. Configure the Raspberry Pi](https://emanual.robotis.com/docs/en/platform/turtlebot3/sbc_setup/#configure-the-raspberry-pi) for those using a remote PC setup (the Raspberry Pi corresponds to the onboard PC, in our case the NUC).
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
The [realsense-ros](https://github.com/IntelRealSense/realsense-ros) repository's README contains installation instructions for the Intel® RealSense™ SDK 2.0 and ROS2 Wrapper. 

## Usage

[ubuntu20-badge]: https://img.shields.io/badge/-UBUNTU%2020%2E04-blue?style=flat-square&logo=ubuntu&logoColor=white
[ubuntu20]: https://releases.ubuntu.com/focal/
[foxy-badge]: https://img.shields.io/badge/-FOXY-orange?style=flat-square&logo=ros
[foxy]: https://docs.ros.org/en/foxy/index.html
