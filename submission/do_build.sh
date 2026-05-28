#!/usr/bin/env bash
#
# Build the submission Docker image for a chosen algorithm + config.
#
# Usage:
#   ./do_build.sh                              # uses defaults below
#   ./do_build.sh restormer3d restormer3d_finetune  # algo and config name
#   ./do_build.sh restormer3d restormer3d_eval_only
#
# The image tag is derived from algo name so different algos can
# coexist in your local Docker (one image each, easy to swap).
#
# Stop at first error
set -e

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

# Defaults — change these or pass as args.
DEFAULT_ALGO="restormer3d"
DEFAULT_CONFIG="restormer3d_finetune"

SUBMISSION_ALGO="${1:-$DEFAULT_ALGO}"
SUBMISSION_CONFIG="${2:-$DEFAULT_CONFIG}"

# Tag = algo-config, lowercased and sanitized for Docker
TAG_SUFFIX=$(echo "${SUBMISSION_ALGO}-${SUBMISSION_CONFIG}" \
             | tr '[:upper:]' '[:lower:]' | tr -c 'a-z0-9-' '-')
DOCKER_IMAGE_TAG="cidc25-submission-${TAG_SUFFIX}"

echo "=+= Build args:"
echo "    SUBMISSION_ALGO   = $SUBMISSION_ALGO"
echo "    SUBMISSION_CONFIG = $SUBMISSION_CONFIG"
echo "    DOCKER_IMAGE_TAG  = $DOCKER_IMAGE_TAG"

# Bring the framework's algos/, configs/, runner/ into the build context.
# Submission lives in dlproj/submission; framework is in dlproj/.
# We copy them in before build, then clean up after.
echo "=+= Staging framework into build context…"
PROJ_ROOT="$( cd "$SCRIPT_DIR/.." && pwd )"
for dir in algos configs runner; do
    if [ -d "${SCRIPT_DIR}/${dir}" ]; then
        rm -rf "${SCRIPT_DIR}/${dir}"
    fi
    cp -r "${PROJ_ROOT}/${dir}" "${SCRIPT_DIR}/${dir}"
done

# Ensure no leftover __pycache__ folders pollute the image.
find "${SCRIPT_DIR}/algos" "${SCRIPT_DIR}/configs" "${SCRIPT_DIR}/runner" \
     -name __pycache__ -type d -exec rm -rf {} + 2>/dev/null || true

cleanup_staging() {
    # Remove the staged copies; keep the original framework intact.
    echo "=+= Removing staged copies…"
    for dir in algos configs runner; do
        rm -rf "${SCRIPT_DIR}/${dir}"
    done
}
trap cleanup_staging EXIT

echo "=+= Building Docker image…"
docker build \
    --platform=linux/amd64 \
    --build-arg SUBMISSION_ALGO="$SUBMISSION_ALGO" \
    --build-arg SUBMISSION_CONFIG="$SUBMISSION_CONFIG" \
    --tag "$DOCKER_IMAGE_TAG" \
    "$SCRIPT_DIR" 2>&1

echo "=+= Built: $DOCKER_IMAGE_TAG"

# Write the tag to a file so do_test_run.sh and do_save.sh can find it.
echo "$DOCKER_IMAGE_TAG" > "${SCRIPT_DIR}/.last_built_tag"
echo "$SUBMISSION_ALGO"  > "${SCRIPT_DIR}/.last_built_algo"
echo "$SUBMISSION_CONFIG" > "${SCRIPT_DIR}/.last_built_config"
