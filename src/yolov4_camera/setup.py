from setuptools import find_packages, setup

package_name = 'yolov4_camera'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),

        #启动脚本
        ('share/' + package_name + '/launch',
         ['launch/face_detector.launch.py',
          
          ]),
        #配置文件
        ('share/' + package_name + '/config/yolov4',
          ['config/yolov4/yolov4-tiny.weights', 
           'config/yolov4/yolov4-tiny.cfg', 
           'config/yolov4/coco.names',
           'config/yolov4/haarcascade_frontalface_default.xml',
           ]),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='jiguang',
    maintainer_email='jiguang@todo.todo',
    description='TODO: Package description',
    license='Apache2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'yolov4_re = yolov4_camera.yolo_detector_node:main',
            'edge_detector = yolov4_camera.edge_detector_node:main',
            'face_detection = yolov4_camera.face_detection_node:main',
            
        ],
    },
)
