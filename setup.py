from setuptools import find_packages, setup

package_name = 'bgs_fit'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='root',
    maintainer_email='root@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'vision_save = bgs_fit.vision_save:main',
            'shape_fit = bgs_fit.shape_fit:main',
            'plan_grasp = bgs_fit.plan_grasp:main',
            'get_gazebo_poses = bgs_fit.get_gazebo_poses:main',
        ],
    },
)
