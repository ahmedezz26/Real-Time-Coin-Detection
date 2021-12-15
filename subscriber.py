#!/usr/bin/env python3
import rospy
from cv_bridge import CvBridge 
from sensor_msgs.msg import Image



def callback(data):
    
    bridge = CvBridge()
    coins= bridge.imgmsg_to_cv2(data)
    
    if(coins[0][0] == 0 and coins[0][1] == 0 and coins[0][2] == 0):
        rospy.loginfo("I have recieved 0 coins:")
    else:
        rospy.loginfo("I have recieved "+ str(coins.shape[0] )+ " coins:")
    rospy.loginfo(coins)
    
def subscriber():

    # In ROS, nodes are uniquely named. If two nodes with the same
    # name are launched, the previous one is kicked off. The
    # anonymous=True flag means that rospy will choose a unique
    # name for our 'listener' node so that multiple listeners can
    # run simultaneously.
    rospy.init_node('subscriber', anonymous=True)

    rospy.Subscriber("mrcamera", Image, callback)

    # spin() simply keeps python from exiting until this node is stopped
    rospy.spin()

if __name__ == '__main__':
    subscriber()
