#!/bin/bash

# ==============================================================================
# Script to start the SSH server and explicitly mount the Windows G: drive in WSL.
# ==============================================================================

# --- Step 1: Start the SSH Server ---
echo "INFO: Attempting to start the SSH server..."

# Use the 'service' command to start the SSH service with sudo.
sudo service ssh start &> /dev/null

# Check the status of the SSH service to confirm it's running.
if pgrep -x "sshd" &> /dev/null; then
  echo "SUCCESS: SSH server process is running."
else
  echo "WARNING: SSH server does not appear to be running."
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

# Before mounting, let's try to unmount it first to clear any broken states.
echo "INFO: Attempting to unmount '$MOUNT_POINT' to ensure a clean state."
sudo umount "$MOUNT_POINT" &> /dev/null

# Execute the specific mount command.
# The '-o metadata' option helps with file permissions.
echo "INFO: Executing mount command..."
sudo mount -t drvfs G: "$MOUNT_POINT" -o metadata

# Verify if the mount was successful.
if mountpoint -q "$MOUNT_POINT"; then
    echo "SUCCESS: The G: drive has been mounted to $MOUNT_POINT."
    echo "INFO: Listing contents of /mnt/g:"
    ls -l "$MOUNT_POINT"
else
    echo "ERROR: The mount command failed. The drive is not mounted."
    echo "INFO: Please check for any error messages above this line."
    exit 1
fi

exit 0