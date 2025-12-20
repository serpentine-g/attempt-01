#!/bin/bash
source /opt/conda/etc/profile.d/conda.sh
conda activate trellis2
cd /app
exec python mining_server.py

