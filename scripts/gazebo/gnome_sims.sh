#!/bin/bash

# Kill any existing instances of ArduPilot or SITL to start fresh
pkill -f arduplane
pkill -f arducopter
pkill -f sim_vehicle.py

# --- Instance 0: TARGET (Mini Talon - ArduPlane) ---
gnome-terminal --tab --title="UAV1 Target" -- bash -c "
sim_vehicle.py -v ArduPlane -f JSON \
-I0 \
--add-param-file=$HOME/ardupilot_gazebo/config/mini_talon_vtail.param \
--console --map \
exec bash"

# --- Instance 1: RIGHT DRONE (Iris - ArduCopter) ---
gnome-terminal --tab --title="UAV2 Right" -- bash -c "
sim_vehicle.py -v ArduCopter -f JSON \
-I1 \
--console --map \
exec bash"

# --- Instance 2: LEFT DRONE (Iris - ArduCopter) ---
gnome-terminal --tab --title="UAV3 Left" -- bash -c "
sim_vehicle.py -v ArduCopter -f JSON \
-I2 \
--console --map \
exec bash"

# --- Instance 3: BOTTOM DRONE (Iris - ArduCopter) ---
gnome-terminal --tab --title="UAV4 Bottom" -- bash -c "
sim_vehicle.py -v ArduCopter -f JSON \
-I3 \
--console --map \
exec bash"