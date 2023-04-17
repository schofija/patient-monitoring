import rclpy
from geometry_msgs.msg import Twist
from my_robot_interfaces.msg import Person

class PatientSubscriber:
    def __init__(self):
        self.ang_thresh = .15
        self.max_ang = .2
        self.opt_dist = 1500
        self.dist_thresh = 500
        self.max_vel = .26
        self.last_spotted_x = 0
        
        self.node = rclpy.create_node('patient_subscriber')
        self.subscription = self.node.create_subscription(
            Person,
            'person',
            self.person_callback,
            1)
        self.x_sign = 0
        self.depth = [0,0,0,0,0]
        self.depth_count = 0

        self.publisher = self.node.create_publisher(Twist, 'cmd_vel', 1) # If the robot falls behind, we'd rather have it follow the newest command than old ones
        
    def person_callback(self,msg_in):
        msg_out = Twist()
        
        if msg_in.depth > 0:
            self.depth[self.depth_count] = msg_in.depth
            self.depth_count = (self.depth_count + 1) % 5
            
        image_width = msg_in.image_width
        image_height = msg_in.image_height

        follow_threshold = int(float(image_width) * self.ang_thresh)

	##############################
        # PATIENT NOT IN FRAME
        ##############################
        if msg_in.x == -1 :
            pass
            # TODO: IMPLEMENT PATIENT VISION LOSS

	##############################
        # PATIENT DETECTED IN FRAME
        ##############################
        else:
            # Assuming left-handed coordinate system
            if msg_in.x < image_width//2 - follow_threshold:
                print("rotate LEFT, msg_in.x==", msg_in.x)
                msg_out.angular.z = self.max_ang #self.max_ang * (0.5 - msg_in.x + self.ang_thresh)
                self.last_spotted_x = 1
                self.x_sign = 0
            elif msg_in.x > image_width//2 + follow_threshold :
                print("rotate RIGHT, msg_in.x==", msg_in.x)
                msg_out.angular.z = (-1.) * self.max_ang #- self.max_ang * (0.5 - msg_in.x + self.ang_thresh)
                self.last_spotted_x = 2
                self.x_sign = 0

            human_depth = sum(self.depth)/len(self.depth)
            if human_depth < 1500:
                # Set magnitude of velocity to max speed (fine tune later)
                msg_out.linear.x = - self.max_vel
                # Set direction of velocity 
                self.x_sign = 1
                print("BACKWARDS", human_depth)
            elif human_depth  > 2500:
                # Set magnitude of velocity to max speed (fine tune later)
                msg_out.linear.x = self.max_vel
                # Set direction of velocity 
                self.x_sign = 1
                print("FORWARDS", human_depth)
            # If no movement needs to be taken, just log that for future reference.
            else:
                self.x_sign = 0

        self.publisher.publish(msg_out)

def main():
    rclpy.init()
    patient_sub = PatientSubscriber()
    rclpy.spin(patient_sub.node)

if __name__ == '__main__':
    main()
