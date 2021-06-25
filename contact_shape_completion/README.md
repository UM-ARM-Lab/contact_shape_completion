# Running
## Running with simulated data
1. `roslaunch shape_completion_visualization live_shape_completion.launch`
2. `roslaunch gpu_voxel_planning fake_victor_setup.launch`
3. `rosrun contact_shape_completion rosbag_kinect.py --play camera --repeat`
4. (In pycharm run) `store_simulation_examples.py` with various args, e.g. `--trial YCB --scene pitcher --store`
5. (In clion run) `shape_completion_simulation`

## Running with live kinect data (simulated robot)
Same as above, except instead of running the rosbag do:
1. `ssh [kinect computer]; roslaunch mps_launch_files kinect_vicon_real_robot.launch pov:="victor_head"`
2. `ssh [segmentation computer]; roslaunch object_segmentation ycb_segmentation.launch`

## Running with live robot
1. `roslaunch shape_completion_visualization live_shape_completion.launch`
2. `roslaunch gpu_voxel_planning real_victor_setup.launch`
3. `ssh [kinect computer]; roslaunch mps_launch_files kinect_vicon_real_robot.launch pov:="victor_head"`
4. `ssh [segmentation computer]; roslaunch object_segmentation ycb_segmentation.launch`
5. `ssh [robot_computer]; (launch the dual arm bridge)
6.  `rosrun gpu_voxel_planning execute_path_with_collision_detection.py`
8. (In pycharm run) `store_simulation_examples.py` with various args, e.g. `--trial YCB --scene pitcher --store`
9. (In clion run) `shape_completion_live`
10. If storing the data for an experiment, copy the `debugging/files/latest_segmented_pts.msg` to the appropriate `saved_requests` folder.


## Evaluating data
1. `roslaunch shape_completion_visualization live_shape_completion.launch`
2. `roslaunch gpu_voxel_planning fake_victor_setup.launch` (Yes, run fake_victor_setup even when evaluating on live data. All messages have already been converted to not rely on live data)
3. (In pycharm run) `contact_completion_evaluation.py --plot`
