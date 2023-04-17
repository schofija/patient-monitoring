"""
robot_launch.py
authors: null
"""

import os

from ament_index_python import get_package_share_directory

from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.actions import IncludeLaunchDescription
from launch.actions import GroupAction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration
from launch.substitutions import TextSubstitution
from launch_ros.actions import Node
from launch_ros.actions import PushRosNamespace
from launch_xml.launch_description_sources import XMLLaunchDescriptionSource
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare
from launch.actions import SetEnvironmentVariable

def generate_launch_description():
    # export TURTLEBOT3_MODEL=waffle
    turtlebot_model = SetEnvironmentVariable(name="TURTLEBOT3_MODEL", value=["waffle"])

    # ros2 launch turtlebot3_bringup robot.launch.py
    robot_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("turtlebot3_bringup"),
                    "launch",
                    "robot.launch.py",
                ]
            )
        ),
    )

    # ros2 launch realsense2_camera rs_launch.py align_depth.enable:=true
    rs2_launch = IncludeLaunchDescription(
        launch_description_source=PythonLaunchDescriptionSource(
            PathJoinSubstitution(
                [
                    FindPackageShare("realsense2_camera"),
                    "launch",
                    "rs_launch.py",
                ]
            )
        ),
        launch_arguments={
            "align_depth.enable": "True"
        }.items()
    )

    ld = LaunchDescription()
    ld.add_action(turtlebot_model)
    ld.add_action(robot_launch)
    ld.add_action(rs2_launch)
    return ld
