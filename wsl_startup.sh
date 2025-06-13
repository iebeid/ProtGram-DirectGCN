#!/bin/bash

# ==============================================================================
# Script to start the SSH server and verify the Windows G: drive mount in WSL.
# ==============================================================================

# --- Step 1: Start the SSH Server ---
echo "INFO: Attempting to start the SSH server..."

# We use the 'service' command which is standard on WSL/Ubuntu
# The 'sudo' is necessary to have the permissions to start a system service.
sudo service ssh start

# Check the status of the SSH service to confirm it's running
SSH_STATUS=$(sudo service ssh status)

# Provide feedback to the user based on the status
if [[ $SSH_STATUS == *"is running"* ]]; then
  echo "SUCCESS: SSH server is running."
else
  echo "WARNING: SSH server may not have started correctly. Status: $SSH_STATUS"
  echo "INFO: Make sure you have installed it with 'sudo apt install openssh-server'"
fi

echo "" # Add a blank line for readability

# --- Step 2: Mount Windows G: Drive ---
echo "INFO: Checking for the Windows G: drive mount point..."

# In WSL, Windows drives are automatically mounted under /mnt/
# The G: drive corresponds to the /mnt/g directory.
WINDOWS_MOUNT_POINT="/mnt/g"

# Check if the directory /mnt/g exists. The '-d' flag checks for a directory.
if [ -d "$WINDOWS_MOUNT_POINT" ]; then
  echo "SUCCESS: Windows drive G: is accessible at $WINDOWS_MOUNT_POINT."
  echo "INFO: You can now access its contents, for example:"
  echo "ls -l $WINDOWS_MOUNT_POINT"
else
  echo "ERROR: The directory $WINDOWS_MOUNT_POINT was not found."
  echo "TROUBLESHOOTING:"
  echo "1. Ensure your G: drive is connected and visible in Windows File Explorer."
  echo "2. Ensure that auto-mounting is enabled in your WSL configuration."
  echo "3. Try restarting WSL with 'wsl --shutdown' in PowerShell, then reopening Ubuntu."
  exit 1
fi

exit 0