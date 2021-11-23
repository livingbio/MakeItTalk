CURRENT=$(pwd)

# Check CUDA_VERSION
export CUDA_VERSION=$(nvcc --version| grep -Po "(\d+\.)+\d+" | head -1)
export TORCH_CUDA_ARCH_LIST="5.2;5.3;6.0;6.1;6.2;7.0;7.2;7.5;8.0;8.6+PTX"

apt update -y &&  DEBIAN_FRONTEND=noninteractive apt install -y --allow-downgrades --allow-change-held-packages --no-install-recommends \
    gcc \
    python3-dev \
    python3-pip \
    libxml2-dev \
    libxslt1-dev \
    zlib1g-dev \
    g++ \
    git \
    ffmpeg \
    libsm6 \
    libxext6

rm -rf /var/lib/apt/lists/*

pip install --no-cache-dir --ignore-installed -r /requirements.txt

mkdir src/examples/dump
mkdir src/examples/ckpt
pip install gdown==3.10.1
gdown -O src/examples/ckpt/ckpt_autovc.pth https://drive.google.com/uc?id=1ZiwPp_h62LtjU0DwpelLUoodKPR85K7x
gdown -O src/examples/ckpt/ckpt_content_branch.pth https://drive.google.com/uc?id=1r3bfEvTVl6pCNw5xwUhEglwDHjWtAqQp
gdown -O src/examples/ckpt/ckpt_speaker_branch.pth https://drive.google.com/uc?id=1rV0jkyDqPW-aDJcj7xSO6Zt1zSXqn1mu
gdown -O src/examples/ckpt/ckpt_116_i2i_comb.pth https://drive.google.com/uc?id=1i2LJXKp-yWKIEEgJ7C6cE3_2NirfY_0a
gdown -O src/examples/dump/emb.pickle https://drive.google.com/uc?id=18-0CYl5E6ungS3H4rRSHjfYvvm-WwjTI
