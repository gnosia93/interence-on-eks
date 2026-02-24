## NVIDIA Dynamo (차세대 분산 추론 프레임워크) ##
Triton Inference Server의 후속 기술로, 대규모 분산 환경에서 생성형 AI 및 추론 모델을 효율적으로 배포하는 하이레벨 프레임워크.

### 주요 특징 ###
* 대규모 분산 환경: 수천 개의 GPU에서 LLM을 구동할 때 필요한 오케스트레이션(요청 라우팅, GPU 스케줄링)을 처리.
* Disaggregated Serving: 프리필(prefill, 프롬프트 분석)과 디코드(decode, 토큰 생성) 단계를 분리하여 GPU 자원을 최적화.
* 엔진 Agnostic: TensorRT-LLM, vLLM, SGLang 등 다양한 고속 추론 엔진을 백엔드로 유연하게 선택 가능.



## TensorRT 시리즈 (모델 최적화 및 런타임) ##
* 정의: NVIDIA GPU에서 딥러닝 모델의 추론 속도를 극대화하는 저레벨(Low-level) 최적화 SDK 및 런타임.
* 주요 특징:
  * 모델 엔진 생성: PyTorch, ONNX 등의 모델을 TensorRT 엔진 파일(.engine)로 변환.
  * 핵심 기술: 레이어 융합(Layer Fusion), 커널 오토튜닝, 양자화(INT8, FP8) 등을 통해 고속 추론.
  * TensorRT-LLM: LLM에 특화된 TensorRT 버전으로, In-flight batching 등 최적화 기술 제공.
  * 사용 목적: 단일 또는 다중 GPU 환경에서 개별 모델의 지연 시간(Latency) 최소화 및 처리량 극대화. 


## 레퍼런스 ##
* * https://github.com/ai-dynamo/dynamo
