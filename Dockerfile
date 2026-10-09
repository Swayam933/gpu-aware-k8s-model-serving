# CUDA runtime base — required for torch to actually use the GPU inside the
# container. On a machine/CI without a GPU, this image still builds and runs
# fine on CPU; the driver/toolkit is only exercised when scheduled onto a
# GPU node with the NVIDIA device plugin installed (see k8s/nvidia-device-plugin.yaml).
FROM nvidia/cuda:12.4.1-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
      python3 python3-pip \
    && rm -rf /var/lib/apt/lists/*

RUN groupadd -r appuser && useradd -r -g appuser appuser

WORKDIR /app
COPY app/requirements.txt .
RUN pip3 install --no-cache-dir -r requirements.txt

COPY app/ .
USER appuser

EXPOSE 8080
HEALTHCHECK --interval=30s --timeout=5s CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8080/healthz')" || exit 1

CMD ["uvicorn", "server:app", "--host", "0.0.0.0", "--port", "8080"]
