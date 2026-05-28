#!/usr/bin/env bash
#
# Run the submission Docker image on local test data.
#
# Usage:
#   ./do_test_run.sh                          # uses last-built image
#   ./do_test_run.sh restormer3d restormer3d_finetune  # builds + runs that algo
#
# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Pick algo/config from args, or fall back to what was last built.
if [ -n "$1" ] || [ -n "$2" ]; then
    DEFAULT_ALGO="${1:-restormer3d}"
    DEFAULT_CONFIG="${2:-restormer3d_finetune}"
    echo "=+= (Re)build the container for $DEFAULT_ALGO / $DEFAULT_CONFIG"
    source "${SCRIPT_DIR}/do_build.sh" "$DEFAULT_ALGO" "$DEFAULT_CONFIG"
else
    if [ -f "${SCRIPT_DIR}/.last_built_tag" ]; then
        echo "=+= Using last-built image (no algo args provided)"
    else
        echo "=+= No prior build found; building with defaults"
        source "${SCRIPT_DIR}/do_build.sh"
    fi
fi

DOCKER_IMAGE_TAG=$(cat "${SCRIPT_DIR}/.last_built_tag")
SUBMISSION_ALGO=$(cat "${SCRIPT_DIR}/.last_built_algo")
SUBMISSION_CONFIG=$(cat "${SCRIPT_DIR}/.last_built_config")
DOCKER_NOOP_VOLUME="${DOCKER_IMAGE_TAG}-volume"

INPUT_DIR="${SCRIPT_DIR}/test/input"
OUTPUT_DIR="${SCRIPT_DIR}/test/output"

cleanup() {
    echo "=+= Cleaning permissions…"
    # Ensure permissions are set correctly on the output
    docker run --rm \
        --platform=linux/amd64 \
        --quiet \
        --volume "$OUTPUT_DIR":/output \
        --entrypoint /bin/sh \
        $DOCKER_IMAGE_TAG \
        -c "chmod -R -f o+rwX /output/* || true"

    # Ensure volume is removed
    docker volume rm "$DOCKER_NOOP_VOLUME" > /dev/null 2>&1 || true
}

# Make sure paths exist
mkdir -p "${SCRIPT_DIR}/model"  # for optional pretrained weights
if [ ! -d "$INPUT_DIR" ]; then
    echo "ERROR: $INPUT_DIR does not exist. Create it and place test"
    echo "       data under: ${INPUT_DIR}/interf0/images/stacked-neuron-images-with-noise/"
    exit 1
fi

# Allow Docker user to read inputs and model
chmod -R -f o+rX "$INPUT_DIR" "${SCRIPT_DIR}/model" 2>/dev/null || true

if [ -d "${OUTPUT_DIR}/interf0" ]; then
    chmod -f o+rwX "${OUTPUT_DIR}/interf0"
    echo "=+= Cleaning up earlier output"
    docker run --rm \
        --platform=linux/amd64 \
        --quiet \
        --volume "${OUTPUT_DIR}/interf0":/output \
        --entrypoint /bin/sh \
        $DOCKER_IMAGE_TAG \
        -c "rm -rf /output/* || true"
else
    mkdir -p -m o+rwX "${OUTPUT_DIR}/interf0"
fi

docker volume create "$DOCKER_NOOP_VOLUME" > /dev/null

trap cleanup EXIT

run_docker_forward_pass() {
    local interface_dir="$1"
    echo "=+= Doing a forward pass on ${interface_dir}"
    echo "=+= Algo: ${SUBMISSION_ALGO}  Config: ${SUBMISSION_CONFIG}"
    # Args on Grand Challenge:
    #   --network none       — offline
    #   --gpus all           — GPU access
    #   --volume /tmp        — /tmp not for permanent files
    #   --volume model:ro    — optional pretrained weights
    docker run --rm \
        --platform=linux/amd64 \
        --network none \
        --gpus all \
        --volume "${INPUT_DIR}/${interface_dir}":/input:ro \
        --volume "${OUTPUT_DIR}/${interface_dir}":/output \
        --volume "$DOCKER_NOOP_VOLUME":/tmp \
        --volume "${SCRIPT_DIR}/model":/opt/ml/model:ro \
        --env SUBMISSION_ALGO="$SUBMISSION_ALGO" \
        --env SUBMISSION_CONFIG="$SUBMISSION_CONFIG" \
        "$DOCKER_IMAGE_TAG"

    echo "=+= Wrote results to ${OUTPUT_DIR}/${interface_dir}"
}

run_docker_forward_pass "interf0"

echo "=+= Save this image for upload via ./do_save.sh"
