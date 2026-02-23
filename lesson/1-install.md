```bash 
# By default, PyTorch CUDA 12.8 package is installed. 
# Install PyTorch CUDA 13.0 package to align with the CUDA version used for building TensorRT LLM wheels.
pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130

sudo apt-get -y install libopenmpi-dev

# Optional step: Only required for disagg-serving
sudo apt-get -y install libzmq3-dev
```
