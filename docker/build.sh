#!/bin/bash
set -e
set -u
SCRIPTROOT="$( cd "$(dirname "$0")" ; pwd -P )"
cd "${SCRIPTROOT}/.."

# Download Isaac Gym Preview 4 if it doesn't exist
ISAAC_GYM_FILE="IsaacGym_Preview_4_Package.tar.gz"
ISAAC_GYM_DIR="isaacgym"

if [ ! -f "${ISAAC_GYM_FILE}" ]; then
    echo "Downloading Isaac Gym Preview 4..."
    wget -O "${ISAAC_GYM_FILE}" "https://developer.nvidia.com/isaac-gym-preview-4"
    echo "Downloaded Isaac Gym Preview 4 as ${ISAAC_GYM_FILE}"
fi

# Extract Isaac Gym if directory doesn't exist
if [ ! -d "${ISAAC_GYM_DIR}" ]; then
    echo "Extracting Isaac Gym..."
    mkdir -p "${ISAAC_GYM_DIR}"
    tar -xzf "${ISAAC_GYM_FILE}"
    rm -f "${ISAAC_GYM_FILE}"
fi

docker build --network host -t phc -f docker/Dockerfile .
