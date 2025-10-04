
FROM nvcr.io/nvidia/pytorch:23.02-py3
ENV DEBIAN_FRONTEND=noninteractive 

# Install system dependencies in a single layer and clean apt cache to reduce image size.
RUN apt-get update \
 && apt-get install -y --no-install-recommends \
    libxcursor-dev \
    libxrandr-dev \
    libxinerama-dev \
    libxi-dev \
    mesa-common-dev \
    zip \
    unzip \
    make \
    gcc-9 \
    g++-9 \
    vulkan-utils \
    mesa-vulkan-drivers \
    pigz \
    git \
    libegl1 \
    git-lfs \
    libosmesa6-dev \
    xserver-xephyr \
    && apt-get clean \
 && update-alternatives --install /usr/bin/gcc gcc /usr/bin/gcc-9 9 \
 && update-alternatives --install /usr/bin/g++ g++ /usr/bin/g++-9 9 \
 # remove apt caches
 && rm -rf /var/lib/apt/lists/*

# WAR for eglReleaseThread shutdown crash in libEGL_mesa.so.0 (ensure it's never detected/loaded)
# Can't remove package libegl-mesa0 directly (because of libegl1 which we need)
RUN rm /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0 /usr/lib/x86_64-linux-gnu/libEGL_mesa.so.0.0.0 /usr/share/glvnd/egl_vendor.d/50_mesa.json

COPY docker/nvidia_icd.json /usr/share/vulkan/icd.d/nvidia_icd.json
COPY docker/10_nvidia.json /usr/share/glvnd/egl_vendor.d/10_nvidia.json

WORKDIR /workspace

# copy gym repo to docker
COPY . .

# Install Python packages. Combine installs to reduce layers and use --no-cache-dir to reduce image size.
# RUN pip install --no-cache-dir \
#     "torchtext==0.18.0" \
#     "torch==2.3.1" \
#     "torchvision==0.18.1" \
#     "torchaudio==2.3.1"
RUN pip install --upgrade pip
RUN cd isaacgym/python && pip install -q -e .

WORKDIR /workspace
# rsl_rl and unitree SDK; pin rsl_rl to known tag. Combining into one pip invocation keeps layers smaller.
RUN pip install -r requirement.txt

# Expose all GPUs inside the container by default
ENV NVIDIA_VISIBLE_DEVICES=all NVIDIA_DRIVER_CAPABILITIES=all
