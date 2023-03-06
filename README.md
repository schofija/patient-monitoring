# patient-monitoring
###### Jack Schofield, Null Atwood, Travis Hudson, Rohan B Ballapragada, Zachary Taylor, Jet Ittihrit, Aditya Raj

The repository contains ROS 2 packages for patient detection. These packages are intended to be run on the remote PC.

### Requirements
**Hardware:**
+ TurtleBot 3 Waffle
+ Remote PC
+ Intel® RealSense™ D405 Depth Camera

**Software:**
+ Ubuntu 20.04.5 LTS x86_64
+ ROS 2 Foxy

### Robot Setup
This section is a guide for setting up the robot's onboard PC.

1. Follow the [TurtleBot3 Quick Start Guide](https://emanual.robotis.com/docs/en/platform/turtlebot3/quick-start/). Please ensure you are following the steps for "Foxy".
2. Install the [ROS2 Wrapper for Intel® RealSense™ Devices](https://github.com/IntelRealSense/realsense-ros).

After doing these steps, you should have the `turtlebot3_bringup` and `realsense2_camera` packages installed on your onboard PC. Next, we will bring up the robot (in order to communicate with the DYNAMIXEL actuators), and start the RealSense™ camera node.

+ To bring up the robot, run: `ros2 launch turtlebot3_bringup robot.launch.py`.
+ To bring up the camera node, run `ros2 launch realsense2_camera rs_launch.py`. 
  + If you get a "package not found" error, you need to set your terminal environment. See Step 5 of the ROS2 Wrapper for Intel® RealSense™ Devices.


