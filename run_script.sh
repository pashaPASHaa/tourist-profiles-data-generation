#!/bin/bash
python3 -u src/run.py --N 10000 --J 200 2>&1 | tee log/traininglog_condensed.log
