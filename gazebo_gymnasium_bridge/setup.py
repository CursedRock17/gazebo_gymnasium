from setuptools import find_packages, setup

package_name = 'gazebo_gymnasium_bridge'

setup(
    name=package_name,
    version='1.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lucas Wendland',
    maintainer_email='lwendlan@umd.edu',
    description='ROS 2 package defining the Openai Gymnasium API mixed with ROS code so we can work in simulation',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'gazebo_environment= gazebo_gymnasium_bridge.gazebo_environment:main',
        ],
    },
)
