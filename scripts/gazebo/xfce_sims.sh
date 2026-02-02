#!/bin/bash

# 1. Kill any existing instances
echo "Cleaning up old processes..."
pkill -f arduplane
pkill -f arducopter
pkill -f sim_vehicle.py

# 2. Launch XFCE Terminal with 4 Tabs
# We source conda.sh to give the subshell access to the 'conda' command.
CONDA_PATH=$(conda info --base)

xfce4-terminal --maximize \
  --tab --title="UAV1 Target" --command="bash -c 'source $CONDA_PATH/etc/profile.d/conda.sh; conda activate vis; sim_vehicle.py -v ArduPlane -f JSON -I0 --add-param-file=$HOME/ardupilot_gazebo/config/mini_talon_vtail.param --console --map; exec bash'" \
  --tab --title="UAV2 Right"  --command="bash -c 'source $CONDA_PATH/etc/profile.d/conda.sh; conda activate vis; sim_vehicle.py -v ArduCopter -f JSON -I1 --console --map; exec bash'" \
  --tab --title="UAV3 Left"   --command="bash -c 'source $CONDA_PATH/etc/profile.d/conda.sh; conda activate vis; sim_vehicle.py -v ArduCopter -f JSON -I2 --console --map; exec bash'" \
  --tab --title="UAV4 Bottom" --command="bash -c 'source $CONDA_PATH/etc/profile.d/conda.sh; conda activate vis; sim_vehicle.py -v ArduCopter -f JSON -I3 --console --map; exec bash'"