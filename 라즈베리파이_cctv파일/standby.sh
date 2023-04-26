#!/bin/bash
sleep 0.5
echo "standby.sh is executed."

sleep 0.5
echo -n "standby.py loading."
sleep 0.5
echo -n "."
sleep 0.5
echo -n "."
sleep 0.5
echo ""
sleep 0.5

python /home/kiki/cctv/cctv.py & exit 0