# syntax=docker/dockerfile:1.7
FROM nvidia/cuda:12.8.1-cudnn-devel-ubuntu22.04

ARG DEBIAN_FRONTEND=noninteractive
ARG NGC_CLI_VERSION=3.64.2
ARG NGC_CLI_SHA256=ceb2abbf473130b39008d12a3fa022afe0a03dd507b013ebc119e8c85935acc6
ARG WFM_SOURCE_REVISION=unknown
ARG WFM_RELEASE_VERSION=0.2.0
LABEL org.opencontainers.image.title="world-from-motion" \
      org.opencontainers.image.version="${WFM_RELEASE_VERSION}" \
      org.opencontainers.image.revision="${WFM_SOURCE_REVISION}"
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    TORCH_CUDA_ARCH_LIST="8.0;9.0" \
    FORCE_CUDA=1 \
    CUDA_HOME=/usr/local/cuda \
    WFM_REPO_ROOT=/opt/world-from-motion \
    WFM_VENDOR_ROOT=/opt/world-from-motion/vendor \
    WFM_CACHE_DIR=/cache \
    WFM_SOURCE_REVISION=${WFM_SOURCE_REVISION} \
    TOKENIZERS_PARALLELISM=false \
    WANDB_MODE=disabled \
    WANDB_DISABLED=true \
    HF_HUB_OFFLINE=1 \
    TRANSFORMERS_OFFLINE=1

RUN apt-get update && apt-get install -y --no-install-recommends \
      build-essential ca-certificates curl ffmpeg fonts-dejavu-core git libgl1 libglib2.0-0 \
      libgomp1 ninja-build python3.10 python3.10-dev python3-pip unzip \
    && rm -rf /var/lib/apt/lists/* \
    && python3.10 -m pip install --upgrade pip==25.2 setuptools==75.8.2 wheel==0.45.1

RUN curl -fsSL \
      "https://api.ngc.nvidia.com/v2/resources/nvidia/ngc-apps/ngc_cli/versions/${NGC_CLI_VERSION}/files/ngccli_linux.zip" \
      -o /tmp/ngccli.zip \
    && echo "${NGC_CLI_SHA256}  /tmp/ngccli.zip" | sha256sum -c - \
    && unzip -q /tmp/ngccli.zip -d /opt \
    && ln -s /opt/ngc-cli/ngc /usr/local/bin/ngc \
    && rm /tmp/ngccli.zip

WORKDIR /opt/world-from-motion
COPY requirements.lock pyproject.toml README.md LICENSE ./
RUN python3.10 -m pip install \
      torch==2.9.1 torchvision==0.24.1 torchaudio==2.9.1 \
      --index-url https://download.pytorch.org/whl/cu128 \
    && python3.10 -m pip install xformers==0.0.33.post2 --no-deps \
    && python3.10 -m pip install -r requirements.lock

COPY vendor/pytorch3d ./vendor/pytorch3d
RUN python3.10 -m pip install --no-build-isolation ./vendor/pytorch3d

RUN python3.10 -m pip install torch-geometric==2.6.1 \
    && python3.10 -m pip install \
      pyg_lib torch_scatter torch_sparse torch_cluster torch_spline_conv \
      -f https://data.pyg.org/whl/torch-2.9.0+cu128.html

COPY vendor/mosca/lib_render ./vendor/mosca/lib_render
RUN python3.10 -m pip install --no-build-isolation ./vendor/mosca/lib_render/simple-knn \
    && python3.10 -m pip install --no-build-isolation ./vendor/mosca/lib_render/diff-gaussian-rasterization-alphadep-add3 \
    && python3.10 -m pip install --no-build-isolation ./vendor/mosca/lib_render/diff-gaussian-rasterization-alphadep \
    && python3.10 -m pip install --no-build-isolation ./vendor/mosca/lib_render/gof-diff-gaussian-rasterization

COPY . .
RUN python3.10 -m pip install --no-deps -e . \
    && python3.10 -m compileall -q src vendor/mosca vendor/diffsynth \
    && wfm --help >/dev/null

ENTRYPOINT []
CMD ["wfm", "--help"]
