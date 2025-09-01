import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np

class EdgeDetectorNode(Node):
    def __init__(self):
        super().__init__('edge_detector_node')
        
        # 初始化CV桥接器
        self.bridge = CvBridge()
        
        # 声明参数
        self.declare_parameters(
            namespace='',
            parameters=[
                ('blur_kernel_size', 5),
                ('canny_threshold1', 50),
                ('canny_threshold2', 150),
                ('input_image_topic', '/image_raw'),
                ('output_image_topic', '/image_edge')
            ]
        )
        
        # 获取参数值
        self.blur_kernel_size = self.get_parameter('blur_kernel_size').value
        self.canny_threshold1 = self.get_parameter('canny_threshold1').value
        self.canny_threshold2 = self.get_parameter('canny_threshold2').value
        input_topic = self.get_parameter('input_image_topic').value
        output_topic = self.get_parameter('output_image_topic').value
        
        # 确保模糊核大小为奇数
        if self.blur_kernel_size % 2 == 0:
            self.blur_kernel_size += 1
            self.get_logger().info(f"Adjusted blur kernel size to {self.blur_kernel_size} (must be odd)")
        
        # 创建订阅者，订阅原始图像话题
        self.subscription = self.create_subscription(
            Image,
            input_topic,
            self.image_callback,
            10  # 队列大小
        )
        self.subscription  # 防止未使用变量警告
        
        # 创建发布者，发布边缘检测后的图像
        self.publisher = self.create_publisher(
            Image,
            output_topic,
            10
        )
        
        self.get_logger().info(f"Edge detector node initialized. Subscribing to {input_topic}, publishing to {output_topic}")
    
    def image_callback(self, msg):
        try:
            # 将ROS图像消息转换为OpenCV格式
            cv_image = self.bridge.imgmsg_to_cv2(msg, desired_encoding='bgr8')
        except Exception as e:
            self.get_logger().error(f"Error converting image: {e}")
            return
        
        # 转换为灰度图
        gray = cv2.cvtColor(cv_image, cv2.COLOR_BGR2GRAY)
        
        # 应用高斯模糊以减少噪声
        blurred = cv2.GaussianBlur(
            gray, 
            (self.blur_kernel_size, self.blur_kernel_size), 
            0
        )
        
        # 应用Canny边缘检测
        edges = cv2.Canny(
            blurred, 
            self.canny_threshold1, 
            self.canny_threshold2
        )
        
        # 将二值边缘图像转换为BGR格式以便可视化
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        try:
            # 将处理后的图像转换回ROS消息格式
            edges_msg = self.bridge.cv2_to_imgmsg(edges_bgr, encoding="bgr8")
            # 保留原始消息的时间戳
            edges_msg.header = msg.header
            # 发布边缘检测图像
            self.publisher.publish(edges_msg)
        except Exception as e:
            self.get_logger().error(f"Error publishing edges image: {e}")

def main(args=None):
    rclpy.init(args=args)
    
    edge_detector_node = EdgeDetectorNode()
    
    try:
        rclpy.spin(edge_detector_node)
    except KeyboardInterrupt:
        pass
    finally:
        edge_detector_node.destroy_node()
        rclpy.shutdown()

