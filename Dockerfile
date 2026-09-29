FROM ghcr.io/astral-sh/uv:0.12.17-python3.12-trixie-slim AS host
RUN apt-get update && apt-get install -y --no-install-recommends git openssh-client ca-certificates \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /opt/embodirun
COPY pyproject.toml uv.lock README.md LICENSE NOTICE THIRD_PARTY_NOTICES.md ./
COPY src/ src/
COPY examples/ examples/
COPY agents/ agents/
COPY integrations/ integrations/
# Checkout file modes can be owner-only; the documented Compose flow runs as
# the host UID so generated recipe files stay editable outside the container.
RUN chmod -R a+rX /opt/embodirun
ENV UV_LINK_MODE=copy UV_PROJECT_ENVIRONMENT=/opt/venv \
    PATH="/opt/venv/bin:$PATH" PYTHONPATH=/opt/embodirun \
    EMBODIRUN_SOURCE_ROOT=/opt/embodirun EMBODIRUN_SCENE_PYTHON=/opt/venv/bin/python \
    PYTHONUNBUFFERED=1
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    python examples/setup_environment.py host --environment /opt/venv
ARG EMBODIRUN_REVISION=unknown
ENV EMBODIRUN_REVISION=$EMBODIRUN_REVISION
ENTRYPOINT ["embodirun", "example"]
CMD ["--help"]

# Local Linux robot host. Device access and verified calibration are supplied at run time.
FROM host AS xlerobot
RUN apt-get update && apt-get install -y --no-install-recommends libglib2.0-0 libgl1 ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    python examples/setup_environment.py snack --environment /opt/venv
# Compose can run as an arbitrary host UID without a passwd entry. Torch asks
# getpass for a username and uses the home directory for runtime caches.
ENV LOGNAME=embodirun USER=embodirun HOME=/tmp

# GPU simulation and inference stay separate from the LeRobot/robot-owner profile.
FROM host AS microduck
RUN apt-get update && apt-get install -y --no-install-recommends libegl1 libgl1 libglfw3 libopengl0 ffmpeg build-essential \
    && rm -rf /var/lib/apt/lists/*
COPY third_party/embodiinfer/ third_party/embodiinfer/
RUN chmod -R a+rX third_party/embodiinfer
RUN --mount=type=cache,target=/root/.cache/uv,sharing=locked \
    test -f third_party/embodiinfer/embodiinfer/__init__.py \
    && python examples/setup_environment.py microduck --environment /opt/venv
ENV LOGNAME=embodirun USER=embodirun HOME=/tmp
ENV MUJOCO_GL=egl PYOPENGL_PLATFORM=egl NVIDIA_DRIVER_CAPABILITIES=compute,utility,graphics
