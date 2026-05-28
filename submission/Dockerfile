FROM --platform=linux/amd64 pytorch/pytorch AS algorithm-cidc25-amd64
# Base: pytorch/pytorch (CUDA-enabled). Grand Challenge runs amd64.

# Build-time selection of which algo+config to bake into this image.
# Override via:  docker build --build-arg SUBMISSION_ALGO=restormer3d ...
ARG SUBMISSION_ALGO=restormer3d
ARG SUBMISSION_CONFIG=restormer3d_finetune

# Promote to runtime env vars so inference.py picks them up by default.
# (Can still be overridden at `docker run` with `-e SUBMISSION_ALGO=...`.)
ENV SUBMISSION_ALGO=${SUBMISSION_ALGO}
ENV SUBMISSION_CONFIG=${SUBMISSION_CONFIG}

# Ensures Python output to stdout/stderr is not buffered.
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

# Python deps
COPY --chown=user:user requirements.txt /opt/app/
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Bundle the whole framework — algos, configs, runner package.
# This makes EVERY algo available, the env var picks which one runs.
COPY --chown=user:user algos /opt/app/algos
COPY --chown=user:user configs /opt/app/configs
COPY --chown=user:user runner /opt/app/runner

# Inference entrypoint
COPY --chown=user:user inference.py /opt/app/

# Print which algo+config the image was built for, so the build log is
# self-documenting.
RUN echo "=== Built for SUBMISSION_ALGO=${SUBMISSION_ALGO} SUBMISSION_CONFIG=${SUBMISSION_CONFIG} ==="

ENTRYPOINT ["python", "inference.py"]
