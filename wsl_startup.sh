#!/bin/bash

# ==============================================================================
# Script to start the SSH server and explicitly mount the Windows G: drive in WSL.
# ==============================================================================

# --- Step 1: Start the SSH Server ---
echo "INFO: Attempting to start the SSH server..."

# Use the 'service' command to start the SSH service with sudo.
sudo service ssh start

# Check the status of the SSH service to confirm it's running.
SSH_STATUS=$(sudo service ssh status)
if [[ $SSH_STATUS == *"is running"* ]]; then
  echo "SUCCESS: SSH server is running."
else
  echo "WARNING: SSH server may not have started correctly. Status: $SSH_STATUS"
fi

echo "" # Add a blank line for readability

# --- Step 2: Mount Windows G: Drive ---
echo "INFO: Proceeding to mount the Windows G: drive..."

# Define the mount point directory.
MOUNT_POINT="/mnt/g"

# The mount command requires the destination directory to exist.
# This command ensures the directory is there, creating it if necessary.
echo "INFO: Ensuring mount point directory '$MOUNT_POINT' exists."
sudo mkdir -p "$MOUNT_POINT"

# Execute the specific mount command as requested.
# Note: The correct filesystem type for WSL is 'drvfs'.
echo "INFO: Executing mount command..."
sudo mount -t drvfs G: "$MOUNT_POINT"

# Verify if the mount was successful by checking the mount point.
if mountpoint -q "$MOUNT_POINT"; then
    echo "SUCCESS: The command seems to have completed successfully."
    echo "INFO: G: drive is now mounted at $MOUNT_POINT."
else
    echo "ERROR: The mount command may have failed. Please check for any error messages above."
    echo "TROUBLESHOOTING:"
    echo "1. Ensure the G: drive is available in Windows."
    echo "2. Run the script with 'bash -x ./your_script_name.sh' to debug."
    exit 1
fi

exit 0