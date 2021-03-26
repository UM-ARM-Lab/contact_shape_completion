# Running
## Running with simulated data
1. `roslaunch shape_completion_visualization live_shape_completion.launch`
2. `roslaunch gpu_voxel_planning fake_victor_setup.launch`
3. `rosrun contact_shape_completion rosbag_kinect.py --play camera --repeat`
4. (In pycharm run) `contact_completion_live_kinect.py`
5. (In clion run) `wip_shape_completion`

## Running with live kinect data (simulated robot)
Same as above, except instead of running the rosbag do:
1. `ssh loki; rc blizzard; roslaunch mps_launch_files kinect_vicon_real_robot.launch pov:="victor_head"`
2. `ssh armada; rc blizzard; roslaunch object_segmentation ycb_segmentation.launch`
