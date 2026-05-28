#!/usr/bin/env bash
#
# Save the last-built Docker image as a .tar.gz for Grand Challenge upload.
#
# Usage:
#   ./do_save.sh                              # uses last-built image
#   ./do_save.sh restormer3d restormer3d_finetune  # builds first, then saves
#
# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# If algo/config args were given, rebuild first.
if [ -n "$1" ] || [ -n "$2" ]; then
    echo "=+= (Re)build for $1 / $2"
    source "${SCRIPT_DIR}/do_build.sh" "$1" "$2"
else
    if [ ! -f "${SCRIPT_DIR}/.last_built_tag" ]; then
        echo "=+= No prior build; building with defaults"
        source "${SCRIPT_DIR}/do_build.sh"
    else
        echo "=+= Using last-built image"
    fi
fi

DOCKER_IMAGE_TAG=$(cat "${SCRIPT_DIR}/.last_built_tag")
SUBMISSION_ALGO=$(cat "${SCRIPT_DIR}/.last_built_algo")
SUBMISSION_CONFIG=$(cat "${SCRIPT_DIR}/.last_built_config")

# Get build timestamp
build_timestamp=$( docker inspect --format='{{ .Created }}' "$DOCKER_IMAGE_TAG")
if [ -z "$build_timestamp" ]; then
    echo "Error: cannot find Docker image $DOCKER_IMAGE_TAG"
    exit 1
fi

formatted_build_info=$(echo $build_timestamp | sed -E 's/(.*)T(.*)\..*Z/\1_\2/' | sed 's/[-,:]/-/g')
output_filename="${SCRIPT_DIR}/${DOCKER_IMAGE_TAG}_${formatted_build_info}.tar.gz"

echo "==+=="
echo "Saving image as ${output_filename}. This can take a while."
echo "  ALGO:   $SUBMISSION_ALGO"
echo "  CONFIG: $SUBMISSION_CONFIG"
echo ""

docker save "$DOCKER_IMAGE_TAG" | gzip -c > "$output_filename"
echo "Container image saved as ${output_filename}"
echo "==+=="
echo "Upload this .tar.gz to Grand Challenge."
