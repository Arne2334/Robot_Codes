import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2 as cv
import os
from ament_index_python.packages import get_package_share_directory

class FaceDetectionNode(Node):
    def __init__(self):
        super().__init__('face_detection_node')
        
        # 订阅图像话题
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10
        )
        
        # 发布绘制后的图像
        self.publisher = self.create_publisher(
            Image,
            '/face_detection/faces',
            10
        )
        
        # 加载 Haar Cascade 分类器
        config_yolov4_path = get_package_share_directory('yolov4_camera')
        model_path = os.path.join(config_yolov4_path,'config/yolov4','haarcascade_frontalface_default.xml')

        self.face_cascade = cv.CascadeClassifier(
            model_path
        )
        
        self.bridge = CvBridge()
        self.get_logger().info('Face Detection Node 已启动')

    def face_filter(self, faces):
        '''
        过滤人脸，保留画面中面积最大的人脸
        '''
        if len(faces) == 0: 
            return None
        
        # 找到面积最大的人脸
        max_face = max(faces, key=lambda face: face[2] * face[3])
        (x, y, w, h) = max_face
        
        # 设置最小检测阈值
        if w < 10 or h < 10: 
            return None
        
        return max_face

    def follow_function(self, img, faces):
        '''
        绘制人脸检测结果
        '''
        img = cv.resize(img, (640, 480))
        
        if len(faces) != 0:
            face = self.face_filter(faces)
            if face is not None:
                (x, y, w, h) = face
                # 绘制矩形框
                cv.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 4)
                cv.putText(img, 'Person', (280, 30), cv.FONT_HERSHEY_SIMPLEX, 0.8, (105, 105, 105), 2)
        return img

    def image_callback(self, msg):
        try:
            # 转换 ROS 图像为 OpenCV 格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, 'bgr8')
            
            # 检测人脸
            faces = self.face_cascade.detectMultiScale(
                cv.cvtColor(cv_image, cv.COLOR_BGR2GRAY),
                scaleFactor=1.3,
                minNeighbors=5
            )
            
            # 绘制检测结果
            result_img = self.follow_function(cv_image, faces)
            
            # 发布绘制后的图像
            result_img_msg = self.bridge.cv2_to_imgmsg(result_img, 'bgr8')
            self.publisher.publish(result_img_msg)
            
        except Exception as e:
            self.get_logger().error(f'处理图像时出错: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = FaceDetectionNode()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


