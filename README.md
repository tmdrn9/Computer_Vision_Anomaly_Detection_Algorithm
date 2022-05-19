# Computer_Vision_Anomaly_Detection_Algorithm
[Dacon 2022 Computer Vision 이상치 탐지 알고리즘 경진대회](https://dacon.io/competitions/official/235894/overview/description)

Team 'remember' : 이승리, 서원진, 최재홍

Private 66th, Score 0.8232 (66/481, 13.7%)

----

### Development Environment

- Google Colab pro

### Train (Train.ipynb)

- data split : 0.8/0.2(train/validation, stratify=label)
- nomalization : mean=0.5, std=0.5 
  
  ↳ Dataset으로 계산한 값으로 학습한 것보다 0.5로 학습한 것이 성능 더 높게 나옴
- augmentation : ShiftScaleRotate, Rotate, VerticalFlip, HorizontalFlip (filp은 metal_nuts에서 제외)

  ↳ Mixup, CutMix, Sharpness, MedianBlur, IAAEmboss, CLAHE, RandomBrightness 등 시도
- epoch : 100
- lr : 0.001
- loss : CrossEntropyLoss
- optimizer : Adam
- scheduler : ReduceLROnPlateau(patience=4)
- model(public) :
  - swin_tiny_patch4_window7_224 (0.76627)
  - mixnet_s (0.72670)
  - efficientnetB2 (0.73043)
  - efficientnetB0 (0.75146)

   ↳ model마다 class weight 유무 실험 후 성능 높은 모델로 채택 → efficientnetB2는 class weight X
   
   ↳ 비교적 parameter 수가 작은 것에 성능이 잘 나옴
   
   ↳ pretrain model 사용
- input size : 300(mixnet_s, efficientnetB2, efficientnetB0), 224(swin_tiny_patch4_window7_224

### Inference (Inference.ipynb)

- Ensemble(soft-voting)
- Test Time Augmentation(TTA)
  - Rotate90(angles=[0, 90, 180, 270])
  - Multiply(factors=[0.9, 1, 1.1])

---


