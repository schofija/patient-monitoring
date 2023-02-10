import rclpy
from geometry_msgs.msg import Twist
from my_robot_interfaces.msg import Person

class PatientSubscriber:
    def __init__(self):
        self.ang_thresh = 36
        self.max_ang = .2
        self.opt_dist = 1500
        self.dist_thresh = 500
        self.max_vel = 3.
        self.last_spotted_x = 0
        self.node = rclpy.create_node('patient_subscriber')
        self.subscription = self.node.create_subscription(
            Person,
            'person',
            self.person_callback,
            1)

        self.publisher = self.node.create_publisher(Twist, 'cmd_vel', 1) # If the robot falls behind, we'd rather have it follow the newest command than old ones
        
    def person_callback(self,msg_in):
        msg_out = Twist()

        #If we can't see anyone, just start spinning
        if msg_in.x == -1 :
            if self.last_spotted_x == 1:
                print("rotate LEFT ***PATIENT OUT OF FRAME***")
                msg_out.angular.z = self.max_ang
                msg_out.linear.x = 0.

            if self.last_spotted_x == 2:
                print("rotate RIGHT ***PATIENT OUT OF FRAME***")
                msg_out.angular.z = (-1.) * self.max_ang
                msg_out.linear.x = 0.
        else:
            # Assuming left-handed coordinate system
            if msg_in.x < 212 - self.ang_thresh :
                print("rotate LEFT, msg_in.x==", msg_in.x)
                msg_out.angular.z = self.max_ang #self.max_ang * (0.5 - msg_in.x + self.ang_thresh)
                self.last_spotted_x = 1
            elif msg_in.x > 212 + self.ang_thresh :
                print("rotate RIGHT, msg_in.x==", msg_in.x)
                msg_out.angular.z = (-1.) * self.max_ang #- self.max_ang * (0.5 - msg_in.x + self.ang_thresh)
                self.last_spotted_x = 2
#            if msg_in.h > .7 :
#                msg_out.linear.x = -.1 # Move backwards with speed proportional to how close the person is
#            elif msg_in.h < 0.65 :
#                msg_out.linear.x = .1 # Move forwards with speed proportional to how distant the person is
            # Have another case for depth == 0 where velocity is determined based on height and width of identified blob

        self.publisher.publish(msg_out)

def main():
    rclpy.init()
    patient_sub = PatientSubscriber()
    rclpy.spin(patient_sub.node)

if __name__ == '__main__':
    main()
