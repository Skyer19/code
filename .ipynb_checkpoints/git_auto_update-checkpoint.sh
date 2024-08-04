#!/bin/bash

# Navigate to the repository directory (optional)
# cd /path/to/your/repository

cd /data/mr423/project/code/

echo 'Current time: $(date)'

# Add all changes to the staging area
git add .

# Commit the changes with a message
git commit -m "auto update"

# Push the changes to the remote repository
git push

