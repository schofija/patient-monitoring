from setuptools import setup

package_name = 'patient_monitoring'

setup(
    name=package_name,
    version='0.0.0',
    packages=[package_name],
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='pmr',
    maintainer_email='pmr@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'my_node = patient_monitoring.my_node:main',
            'patient_follower = patient_monitoring.patient_follower:main',
            'patient_watcher = patient_monitoring.patient_watcher:main',
            'pose_detection = patient_monitoring.pose_detection:main',
            'fall_detection = patient_monitoring.fall_detection:main'
        ],
    },
)
