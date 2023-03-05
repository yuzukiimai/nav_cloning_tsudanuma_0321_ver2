#!/usr/bin/env python3
from __future__ import print_function
from numpy import dtype
import roslib
roslib.load_manifest('nav_cloning')
import rospy
import cv2
from sensor_msgs.msg import Image
from cv_bridge import CvBridge, CvBridgeError
from nav_cloning_pytorch import *
from skimage.transform import resize
from geometry_msgs.msg import Twist
import copy
from std_msgs.msg import Bool
from waypoint_manager_msgs.msg import Waypoint

class nav_cloning_node:
    def __init__(self):
        rospy.init_node('nav_cloning_node', anonymous=True)
        self.mode = rospy.get_param("/nav_cloning_node/mode", "change_dataset_balance")
        self.action_num = 1
        self.dl = deep_learning(n_action = self.action_num)
        self.bridge = CvBridge()
        self.image_sub = rospy.Subscriber("/camera/rgb/image_raw", Image, self.callback)
        self.vel_sub = rospy.Subscriber("/nav_vel", Twist, self.callback_vel)
        self.nav_pub = rospy.Publisher('/cmd_vel', Twist, queue_size=10)
        self.action = 0.0
        self.episode = 0
        self.vel = Twist()
        self.cv_image = np.zeros((480,640,3), np.uint8)
        self.navigation = True
        self.is_started = False
        self.waypoint_flg = rospy.Subscriber("/waypoint_manager/waypoint/is_reached", Bool, self.callback_dl_output)
        self.waypoint_sub = rospy.Subscriber("/waypoint_manager/waypoint", Waypoint, self.callback_waypoint)
        self.goal = False
        self.count = 0
        self.way_1 = str()
        self.way_2 = str()
        self.way_3 = str()
        self.way_4 = str()
        self.way_5 = str()
        self.way_6 = str()
        self.way_7 = str()
        self.flg = 0

    def callback(self, data):
        try:
            self.cv_image = self.bridge.imgmsg_to_cv2(data, "bgr8")
        except CvBridgeError as e:
            print(e)

    def callback_vel(self, data):
        self.vel = data
        self.action = self.vel

    def callback_dl_output(self, data):
        self.goal = data.data
        if self.goal == True and self.flg == 0:
            self.count += 1
    
    def callback_waypoint(self, data):
        self.way_1 = data.properties[0].data
        self.way_2 = data.properties[1].data
        self.way_3 = data.properties[2].data
        self.way_4 = data.properties[3].data
        self.way_5 = data.properties[4].data
        self.way_6 = data.properties[5].data

        if self.count <= 4:
            self.way_7 = data.properties[6].data

        if self.way_1 == "true" or\
           self.way_2 == "true" or\
           self.way_3 == "true" or\
           self.way_4 == "true" or\
           self.way_5 == "true" or\
           self.way_6 == "true":
            self.flg = 1
        
        elif self.count <= 4 and self.way_7 == "true":
            self.flg = 1

        else:
            self.flg = 0

    def loop(self):
        if self.cv_image.size != 640 * 480 * 3:
            return
        if self.vel.linear.x != 0:
            self.is_started = True
        if self.is_started == False:
            return
        
        img = resize(self.cv_image, (48, 64), mode='constant')

        if self.flg == 1:
            self.count = 0

        if self.count == 5:
            self.navigation = False
            self.dl.load("/home/yuzuki/model_gpu.pt")

        if self.episode > 700:
            self.navigation = True
           
        if self.count == 14:
            self.navigation = False
            self.dl.load("/home/yuzuki/model_gpu.pt")

        if self.episode > 1700:
            self.navigation = True

        if self.navigation == True:
            target_action = self.action
            self.vel = target_action
            self.nav_pub.publish(self.vel)
            print("navigation")

        else:
            target_action = self.dl.act(img)
            print(str(self.episode) + ", dl_output")
            self.episode += 1
            self.vel.linear.x = 0.2
            self.vel.angular.z = target_action
            self.nav_pub.publish(self.vel)

        print(self.count)
        temp = copy.deepcopy(img)
        cv2.imshow("Resized Image", temp)
        cv2.waitKey(1)

if __name__ == '__main__':
    rg = nav_cloning_node()
    DURATION = 0.2
    r = rospy.Rate(1 / DURATION)
    while not rospy.is_shutdown():
        rg.loop()
        r.sleep()