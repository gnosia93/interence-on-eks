
* [1. EKS 생성하기](https://github.com/gnosia93/infer-on-eks/blob/main/lesson/1-create-eks.md)

* [2. GPU 노드풀 생성](https://github.com/gnosia93/infer-on-eks/blob/main/lesson/2-gpu-nodepool.md)

* [2. TensorRT-LLM](https://github.com/gnosia93/infer-on-eks/blob/main/lesson/2-tensorrt-llm.md)
   
* [3. NVIDIA Dyanmo](https://github.com/gnosia93/post-training/blob/main/lesson/3-dynamo.md)
  - [로컬 Docker 배포하기](https://github.com/gnosia93/interence-on-eks/blob/main/lesson/3-dynamo-docker.md) 
  - [EKS 배포하기](https://github.com/gnosia93/interence-on-eks/blob/main/lesson/3-dynamo-eks.md) 

* [4. 엔드포인트 성능 테스트하기]

* [5. Quantization](https://github.com/gnosia93/post-training/blob/main/lesson/2-quantization.md)
  - 모델 성능 테스트하기 
  - 인퍼런스 성능 테스트


```
Step 1: 단일 컨테이너 방식 (Python FastAPI + Model) → "아, 트래픽 몰리니 GPU가 노네?"
Step 2: NVIDIA Dynamo/Triton 그래프 방식 도입 → "전처리(CPU), 추론(GPU), 후처리(GPU) 로 분리하여 GPU 사용률을 최대화 한다."
Step 3: Karpenter를 이용한 GPU 노드 오토스케일링 → "Keda or Planner 기반 비교"
```






## 레퍼런스 ##

* https://github.com/NVIDIA/Model-Optimizer/tree/main 
* https://github.com/huggingface/accelerate









