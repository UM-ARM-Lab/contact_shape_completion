import abc

from arm_robots.victor import Victor
from sensor_msgs.msg import PointCloud2
from sensor_msgs import point_cloud2
import numpy as np
import rospy
from visualization_msgs.msg import Marker
from std_msgs.msg import ColorRGBA, Header
from geometry_msgs.msg import Pose
from copy import deepcopy
from gpu_voxel_planning_msgs.msg import TSR


class GoalGenerator(abc.ABC):
    def __init__(self):
        self.goal_pt_pub = rospy.Publisher("goal_pt", Marker, queue_size=10)

    @abc.abstractmethod
    def generate_goal(self, state):
        pass


class CheezeitGoalGenerator(GoalGenerator):
    def __init__(self):
        super().__init__()
        self.victor = Victor()
        self.victor.connect()

        pose = Pose()
        pose.orientation.x = -0.05594805960241513
        pose.orientation.y = -0.7682472566147173
        pose.orientation.z = -0.6317937464624142
        pose.orientation.w = 0.08661771909760922
        self.base_pose = pose

    def generate_goal(self, state: PointCloud2):
        goal_pt = self.generate_goal_point(state)
        goal_config = self.generate_goal_config(goal_pt)
        return goal_config

    def generate_goal_config(self, goal_pt):
        # goal_config = [0, 0, 0, 0, 0, 0]
        # move_group = self.victor.get_move_group_commander('right_arm')
        pose = deepcopy(self.base_pose)

        pose.position.x = goal_pt[0]
        pose.position.y = goal_pt[1]
        pose.position.z = goal_pt[2]

        goal_configs = self.victor.jacobian_follower.compute_IK_solutions(pose, "right_arm")

        if len(goal_configs) == 0:
            return None

        def norm(lst):
            return sum([elem ** 2 for elem in lst])

        goal_config = min(goal_configs, key=norm)

        return goal_config

    def generate_goal_point(self, state: PointCloud2):
        pts = point_cloud2.read_points(state, field_names=('x', 'y', 'z'))
        pts = np.array([p for p in pts])
        if len(pts) == 0:
            raise RuntimeError("Cannot generate a goal from no points")

        centroid = np.mean(pts, axis=0)
        back = np.max(pts, axis=0)

        goal_pt = [back[0] + 0.1, centroid[1], centroid[2]]

        m = Marker(color=ColorRGBA(a=1.0, r=0.0, g=1.0, b=0.0),
                   header=Header(stamp=rospy.Time().now(), frame_id=state.header.frame_id),
                   type=Marker.SPHERE)
        m.pose.orientation.w = 1.0
        m.pose.position.x = goal_pt[0]
        m.pose.position.y = goal_pt[1]
        m.pose.position.z = goal_pt[2]
        m.scale.x = 0.1
        m.scale.y = 0.1
        m.scale.z = 0.1

        self.goal_pt_pub.publish(m)
        return goal_pt

    def generate_goal_tsr(self, pts):
        pt = self.generate_goal_point(pts)
        tsr = TSR(x_lower=pt[0] - 0.1, x_upper=pt[0] + 0.1,
                  y_lower=pt[1] - 0.1, y_upper=pt[1] + 0.1,
                  z_lower=pt[2] - 0.05, z_upper=pt[2] + 0.05
                  )
        tsr.header.frame_id="notsureyet"

        m = Marker(color=ColorRGBA(a=0.3, r=0.0, g=1.0, b=0.0),
                   header=Header(stamp=rospy.Time().now(), frame_id=pts.header.frame_id),
                   type=Marker.CUBE)
        m.ns = "tsr"
        m.pose.orientation.w = 1.0
        m.pose.position.x = pt[0]
        m.pose.position.y = pt[1]
        m.pose.position.z = pt[2]
        m.scale.x = tsr.x_upper - tsr.x_lower
        m.scale.y = tsr.y_upper - tsr.y_lower
        m.scale.z = tsr.z_upper - tsr.z_lower

        self.goal_pt_pub.publish(m)

        return tsr
