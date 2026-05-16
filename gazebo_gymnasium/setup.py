from setuptools import find_packages, setup

package_name = 'gazebo_gymnasium'

setup(
    name=package_name,
    version='0.1.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages', ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Lucas Wendland',
    maintainer_email='mtglucas1@gmail.com',
    description='Gymnasium interface for Gazebo Sim',
    license='Apache-2.0',
    entry_points={
        'console_scripts': [],
    },
)
