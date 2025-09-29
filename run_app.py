#!/usr/bin/env python3
import os
import subprocess
import sys

# Set the directory to the current working directory
script_dir = os.getcwd()
os.chdir(script_dir)

# Set environment variables
env = os.environ.copy()
env.setdefault("PORT", "8080")

# Adjust the script name below if your main file is different
subprocess.run([sys.executable, "hr_agent_tbz_aws.py"], env=env)
