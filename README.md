# Build vLLM on NVIDIA Jetson Thor (via Docker)

> Minimal steps to build and run vLLM on Jetson AGX Thor.

---

## 1) Host setup

```bash
# Docker + NVIDIA CTK
sudo apt-get update
sudo apt install -y nvidia-container curl jq
curl -fsSL https://get.docker.com | sh
sudo systemctl --now enable docker
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl daemon-reload && sudo systemctl restart docker

# Make nvidia the default runtime
sudo jq '. + {"default-runtime": "nvidia"}' /etc/docker/daemon.json |   sudo tee /etc/docker/daemon.json.tmp >/dev/null &&   sudo mv /etc/docker/daemon.json.tmp /etc/docker/daemon.json

# Optional: run docker without sudo
sudo usermod -aG docker $USER && newgrp docker
```

## 2) CUDA Toolkit 13.0 (only if `ptxas` is missing)

```bash
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2404/sbsa/cuda-keyring_1.1-1_all.deb
sudo dpkg -i cuda-keyring_1.1-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda-toolkit-13-0 python3-dev

# verify
ptxas --version
```

## 3) Start container (NGC PyTorch)

```bash
docker run -d -it \ 
    -v $(PWD):/workspace \
    --runtime nvidia \
    --ipc host \
    --privileged \ 
    --restart=unless-stopped \
    --name vllm-container 
    nvcr.io/nvidia/pytorch:25.09-py3 /bin/bash
```

## 4) GPU check (in-container)

```bash
python3 <<'EOF'
import torch
print("PyTorch version:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU name:", torch.cuda.get_device_name(0))
    x = torch.rand(10000, 10000, device="cuda")
    print("Tensor sum:", x.sum().item())
EOF
```

## 5) Build vLLM (in-container)

```bash
git clone --recursive https://github.com/vllm-project/vllm.git
cd vllm
python3 use_existing_torch.py
pip install -r requirements/build.txt
pip install --prerelease=allow triton flashinfer-python xgrammar

export TORCH_CUDA_ARCH_LIST=11.0a
export TRITON_PTXAS_PATH=$(command -v ptxas)
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

pip install --no-build-isolation -e .
```

## Notes
- If `flashinfer-python`/`apache-tvm-ffi` resolve fails, keep `--prerelease=allow`.
- If no GPU in container, re-check Docker default runtime = `nvidia` and restart Docker.
- If `ptxas` missing, (re)install CUDA Toolkit 13.0 (Section 2).

## References
- Jetson AGX Thor Dev Kit — Docker Setup
- NVIDIA Dev Forum: Run vLLM on Thor
