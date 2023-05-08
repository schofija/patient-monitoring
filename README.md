# patient-monitoring
###### Jack Schofield, Null Atwood, Travis Hudson, Rohan B Ballapragada, Zachary Taylor, Jet Ittihrit, Aditya Raj
[![ubuntu20][ubuntu20-badge]][ubuntu20]
[![foxy][foxy-badge]][foxy]

[![License](https://img.shields.io/github/license/schofija/patient-monitoring?style=flat-square)](LICENSE)

ROS2 packages for the patient-monitoring robot. For the patient-monitoring Android application, go to [**pmr-app-test**]( https://github.com/JetLiTheQT/pmrtest).

### Requirements
**Hardware:**
+ TurtleBot 3 Waffle
+ Remote PC
+ Intel® RealSense™ D405 Depth Camera

**Software:**
+ Ubuntu 20.04.5 LTS x86_64
+ ROS 2 Foxy

Setup

1. Follow the [TurtleBot3 Quick Start Guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/). Please ensure you are following the steps for "Foxy".
2. Install the [ROS2 Wrapper for Intel® RealSense™ Devices](https://github.com/IntelRealSense/realsense-ros) on the TurtleBot's onboard PC.

After doing these steps, you should have the `turtlebot3_bringup` and `realsense2_camera` packages installed on your onboard PC. Next, we will bring up the robot (in order to communicate with the DYNAMIXEL actuators), and start the RealSense™ camera node.

+ To bring up the robot, run: `ros2 launch turtlebot3_bringup robot.launch.py`.
+ To bring up the camera node, run `ros2 launch realsense2_camera rs_launch.py`. 
  + If you get a "package not found" error, you need to set your terminal environment. See Step 5 of the ROS2 Wrapper for Intel® RealSense™ Devices.
  
From here, you can either run the Patient Monitoring Robot (PMR) package on a remote PC, or the TurtleBot3 PC. There are several factors to consider when making this decision:
+ Processing power and resources on each device
+ Bandwidth and latency of the network the devices communicate on
+ Requirements of the application
  + As the PMR nodes require real-time processing of camera data, it is beneficial to run on the robot PC, where data can be processed without needing to transmit it over a network. This minimizes latency and removes the risk of dropped packets.
  + However, as this program requires heavy processing power, you may see better performance by running it on a remote PC.



[ubuntu20-badge]: https://img.shields.io/badge/-UBUNTU%2020%2E04-blue?style=flat-square&logo=ubuntu&logoColor=white
[ubuntu20]: https://releases.ubuntu.com/focal/
[foxy-badge]: https://img.shields.io/badge/-FOXY-orange?style=flat-square&logo=ros
[foxy]: https://docs.ros.org/en/foxy/index.html
