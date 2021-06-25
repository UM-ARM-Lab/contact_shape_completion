# CLASP

THE CODE FOR THE INITIAL SUBMISSION TO CoRL: A note to the reviewers
This repo will likely not run as is. Due to anonymity, some dependencies are not listed, and therefore it is unlikely you will be able to run all parts of the code without errors.
The code exists on a github repo, which will be published after paper acceptance. The github repo includes these readmes, contact information, dependencies and install instructions for the lab-specific repositories.





Constrained LAtent Shape Projection (CLASP) combines a shape completion neural network with contact measurements from a robot.


# Quick Start
1. Set up ROS
2. Clone this repo in your ROS path. Rebuild (e.g. catkin build), re-source
3. Install dependencies
4. Download datasets and pretrained models by running `shape_completion_training/scripts/download_pretrained.py`

### Data Analysis
Trial results are in `./evaluations`
To recreate the plots from the paper using the pre-run trials, `contact_completion_evaluation.py --plot`


### Rerun shape completion experiments and visualize performance
To rerun shape completion experiments using the robot motion and contacts as recorded from the paper, in separate terminals run:
1. `roslaunch shape_completion_visualization live_shape_completion.launch` (Rviz will start)
2. `roslaunch shape_completion_visualization transforms.launch` (Sets transforms between robot and shape objects)
3. `contact_completion_evaluation.py --plot --regenerate`  (You will see many shape being generated and updates. Will take over an hour to complete)


# Full Stack
The full experimental setup requires running a simulated, or real robot, which moves and contacts objects.
To build the software stack used in the experiments, set up the dependencies

[[[ Dependencies omitted in CoRL submission due to double-blind review requirements ]]]

Then run
1. `roslaunch shape_completion_visualization live_shape_completion.launch` (Rviz will start)
2. [[[ Robot launch function omitted due to double-blind review requirements ]]]
3. `store_simulation_examples --trial [PRETRAINED_NETWORK_NAME] --scene [SCENE_NAME] --store`
