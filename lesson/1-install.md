기본 설치된 CUDA 12.8용 PyTorch 대신, CUDA 13.0 환경에서 빌드된 PyTorch 2.9.1 버전을 설치한다.
PyTorch와 CUDA 런타임 버전을 일치시켜야 드라이버 충돌이나 성능 저하 없이 GPU 연산을 수행할 수 있다.
고성능 병렬 계산을 위한 OpenMPI 개발 라이브러리를 설치한다.TensorRT-LLM은 대규모 모델을 여러 개의 GPU에 나누어 처리(모델 병렬화)할 때 GPU 간 통신이 필수적이다. 
```bash 
pip3 install torch==2.9.1 torchvision --index-url https://download.pytorch.org/whl/cu130

sudo apt-get -y install libopenmpi-dev

# Optional step: Only required for disagg-serving
sudo apt-get -y install libzmq3-dev
```
