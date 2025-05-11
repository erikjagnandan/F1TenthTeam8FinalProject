#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
from sensor_msgs.msg import LaserScan
from ackermann_msgs.msg import AckermannDriveStamped
from nav_msgs.msg import Odometry
from visualization_msgs.msg import MarkerArray, Marker
from ament_index_python.packages import get_package_share_directory
import os
import cv2
import time


class NeRFImitationNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = models.resnet18(pretrained=False)
        self.encoder.fc = nn.Sequential(
            nn.Linear(self.encoder.fc.in_features, 128),
            nn.ReLU(),
            nn.Linear(128, 2),
            nn.Tanh()
        )

    def forward(self, x):
        out = self.encoder(x)  # shape: (batch_size, 2)
        # First: steering angle in [-pi/4, pi/4], Second: speed in [0, 4]
        steering = out[:, 0] * (np.pi / 4)
        speed = (out[:, 1] + 1) * 2
        return steering, speed


class PurePursuit(Node):
    def __init__(self):
        super().__init__('vision_node')
        
        print("Setting Up Camera")
        self.cap = cv2.VideoCapture('/dev/video4', cv2.CAP_V4L2)
        if not self.cap.isOpened():
            print("Cannot open camera")
            exit()

        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 960)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 540)

        target_fps = 60.0
        frame_interval = 1.0 / target_fps
        print("Finished Setting Up Camera")

        # ROS subscriptions and publications
        self.drive_pub = self.create_publisher(AckermannDriveStamped, '/drive', 10)
        self.timer = self.create_timer(1.0/30.0, self.publish_drive)  # Publish every second

        # Load model
        pkg_share_dir = get_package_share_directory('final_project')
        self.model_path = os.path.join(pkg_share_dir, 'nerf_imitation_model_lam_1.pth')
        #self.model_path = "/home/nvidia/f1tenth_ws/src/pure_pursuit_final_project/share/nerf_imitation_model_lam_1.pth"
        self.model = NeRFImitationNet()
        self.model.load_state_dict(torch.load(self.model_path, map_location=torch.device('cpu')))
        print("Loaded Model at " + self.model_path)
        self.model.cuda()
        self.model.eval()
        
        self.image_transform = transforms.Compose([
            transforms.Resize((224, 224)),  # ResNet input size
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ResNet normalization
                                 std=[0.229, 0.224, 0.225])
        ])
                
        
    def publish_drive(self):
        
        try:
            ret, frame = self.cap.read()
            if not ret:
                print("Failed to open frame")
                return

            try:
                frame_bgr = cv2.cvtColor(frame, cv2.COLOR_YUV2BGR_UYVY)
            except:
                frame_bgr = frame
                
            #cv2.imshow('frame', frame_bgr)
                
            image = Image.fromarray(frame_bgr)
            
            input_tensor = self.image_transform(image).unsqueeze(0).cuda()  # Add batch dimension

            # Inference
            with torch.no_grad():
                steering_angle, speed = self.model(input_tensor)
                steering_angle = steering_angle.squeeze(0).cpu().numpy()
                speed = speed.squeeze(0).cpu().numpy()
                
            max_speed_expert = 2.0
            max_speed_agent = 2.0
            min_speed_agent = 0.5
            speed *= max_speed_agent/max_speed_expert
            if speed < min_speed_agent:
                speed = min_speed_agent
            self.get_logger().info(f'Predicted: steering_angle={steering_angle:.3f}, speed={speed:.3f}')

            # Publish drive command
            drive_msg = AckermannDriveStamped()
            drive_msg.header.stamp = self.get_clock().now().to_msg()
            drive_msg.header.frame_id = 'map'
            drive_msg.drive.steering_angle = float(steering_angle)
            drive_msg.drive.speed = float(speed)
            self.drive_pub.publish(drive_msg)

        except KeyboardInterrupt:
            print("Received CTRL-C, releasing cap and closing cv2 windows before exiting")
            self.cap.release()
            cv2.destroyAllWindows()
            

def main(args=None):
    rclpy.init(args=args)
    print("Vision-Based Navigation Initialized")
    node = PurePursuit()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()


if __name__ == '__main__':
    main()
