import launch
import launch_ros
from ament_index_python.packages import get_package_share_directory
import os
from launch.actions import ExecuteProcess


def generate_launch_description():
    # 获取默认路径
    # base_path = get_package_share_directory('yolov4_camera')
    # default_model_path = os.path.join(base_path,'config/yolov4','haarcascade_frontalface_default.xml')

    # 启动 usb_cam_node_exe
    usb_cam_node = launch_ros.actions.Node(
        package='usb_cam',
        executable='usb_cam_node_exe',
        name='usb_cam_node',
        parameters=[{
            'video_device': '/dev/video0',
            'image_width': 640,
            'image_height': 480,
            'framerate': 30.0,
            'pixel_format': 'yuyv',
            'camera_frame_id': 'camera_link'
        }]
    )

    # 启动 face_detection_node
    face_detection_node = launch_ros.actions.Node(
        package='yolov4_camera',
        executable='face_detection',
    )

    # 启动 rqt
    rqt_process = ExecuteProcess(
        cmd=['rqt'],
        output='screen'
    )

    return launch.LaunchDescription([
        usb_cam_node,
        face_detection_node,
        rqt_process
    ])