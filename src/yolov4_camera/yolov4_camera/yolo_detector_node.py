import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from cv_bridge import CvBridge
import cv2
import numpy as np
import os
from ament_index_python.packages import get_package_share_directory


class YOLODetector(Node):
    def __init__(self):
        super().__init__('yolo_detector')
        self.bridge = CvBridge()
        
        # 订阅摄像头数据
        self.subscription = self.create_subscription(
            Image,
            '/image_raw',
            self.image_callback,
            10)
        
        # 加载 YOLOv4-tiny 模型
        config_yolov4_path = get_package_share_directory('yolov4_camera')
        name_path = os.path.join(config_yolov4_path,'config/yolov4','coco.names')
        cfg_path = os.path.join(config_yolov4_path,'config/yolov4','yolov4-tiny.cfg')
        weights_path = os.path.join(config_yolov4_path,'config/yolov4','yolov4-tiny.weights')
        # self.get_logger().info(f"YOLO 配置文件路径: {config_yolov4_path}")  # 输出路径
        # self.get_logger().info(f"YOLO 类别名称文件路径: {name_path}")  # 输出路径
        # self.get_logger().info(f"YOLO 网络配置文件路径: {cfg_path}")  # 输出路径
        # self.get_logger().info(f"YOLO 权重文件路径: {weights_path}")  # 输出路径
        self.net = cv2.dnn.readNet(weights_path, cfg_path)
        self.layer_names = self.net.getLayerNames()
        self.output_layers = [self.layer_names[i - 1] for i in self.net.getUnconnectedOutLayers()]
        
        # 加载类别名称
        with open(name_path, "r") as f:
            self.classes = [line.strip() for line in f.readlines()]
        
        # 设置阈值
        self.confidence_threshold = 0.5 #置信度最低值
        self.nms_threshold = 0.4
        
        # 发布检测结果
        self.detection_pub = self.create_publisher(Image, '/detection_results', 10)
    
    def image_callback(self, msg):
        try:
            # 转换 ROS 图像消息为 OpenCV 格式
            frame = self.bridge.imgmsg_to_cv2(msg, "bgr8")
            
            # 执行 YOLO 检测
            height, width, channels = frame.shape
            blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
            self.net.setInput(blob)
            outs = self.net.forward(self.output_layers)
            
            # 解析检测结果
            class_ids = [] # 类别 ID,从classes变量中提取
            confidences = [] # 存储检测到的目标的置信度
            boxes = [] # 存储检测到的目标的边界框（bounding box）坐标。
            
            for out in outs:
                # 遍历当前输出层中的每个检测框信息
                for detection in out:
                    # detection是一个包含检测信息的数组，前5个元素是边界框坐标和置信度，
                    # 从第6个元素开始是各个类别的预测分数
                    scores = detection[5:]

                    # 找到分数最高的类别索引（即预测的目标类别）
                    class_id = np.argmax(scores)

                    # 获取该类别对应的置信度（预测正确的概率）
                    confidence = scores[class_id]
                    
                    if confidence > self.confidence_threshold:
                        # 坐标转换
                        center_x = int(detection[0] * width)
                        center_y = int(detection[1] * height)
                        w = int(detection[2] * width)
                        h = int(detection[3] * height)
                        x = int(center_x - w / 2)
                        y = int(center_y - h / 2)
                        
                        boxes.append([x, y, w, h])
                        confidences.append(float(confidence))
                        class_ids.append(class_id)
            
            # 应用非最大抑制
            # 只保留一个边框
            indices = cv2.dnn.NMSBoxes(boxes, confidences, self.confidence_threshold, self.nms_threshold)
            
            # 绘制检测结果
            colors = np.random.uniform(0, 255, size=(len(self.classes), 3))
            font = cv2.FONT_HERSHEY_PLAIN
            
            for i in indices:
                box = boxes[i]
                x, y, w, h = box
                label = str(self.classes[class_ids[i]])
                confidence = confidences[i]
                color = colors[class_ids[i]]
                
                cv2.rectangle(frame, (x, y), (x + w, y + h), color, 2)
                cv2.putText(frame, f"{label} {confidence:.2f}", (x, y - 5), font, 1, color, 2)
            
            # 发布检测结果
            detection_msg = self.bridge.cv2_to_imgmsg(frame, "bgr8")
            self.detection_pub.publish(detection_msg)
            
        except Exception as e:
            self.get_logger().error(f"Error processing image: {e}")

def main(args=None):
    rclpy.init(args=args)
    node = YOLODetector()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()
