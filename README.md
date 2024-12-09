# KTL_project_09_Pathology_ProstateCancer_Segmentation

        ./KTL_project_09_Pathology_ProstateCancer_Segmentation
        |-- Code
        |   |   |-- 01.\ data_preprocessing_example-checkpoint.ipynb
        |   |   |-- 02.\ Training-checkpoint.ipynb
        |   |   |-- 03.\ Evaluation-checkpoint.ipynb
        |   |   |-- Preprocessing-checkpoint.py
        |   |-- 01.\ data_preprocessing_example.ipynb
        |   |-- 02.\ Training.ipynb
        |   |-- 03.\ Evaluation.ipynb
        |   |-- Preprocessing.py
        |   |-- __pycache__
        |   |   |-- Preprocessing.cpython-39.pyc
        |   |   |-- _utils_torch.cpython-39.pyc
        |   |   |-- loss.cpython-39.pyc
        |   |   |-- model_torch.cpython-39.pyc
        |   |   `-- modules_torch.cpython-39.pyc
        |   |-- _utils_torch.py
        |   |-- loss.py
        |   |-- model_torch.py
        |   |-- modules_torch.py
        |   `-- output
        |       |-- model_final.pth
        `-- Data
            |-- img
            |   |-- ###.tiff
            `-- label
                |-- ###_mask.tiff


## Data Description
1. 학습용 데이터 (/Data/...)
   - 이미지와 마스크의 파일명이 100% 매칭되지 않음.
   - 파일명을 기준으로 동일하게 대응되는 이미지와 마스크 파일만 Load.
## Prerequisites
Before you begin, ensure you have met the following requirements:
- Linux or macOS
- Python 3
- CPU or NVIDIA GPU + CUDA CuDNN
- Other dependencies can be installed using `environment.yml`
  
## Getting Started
### Installation

- Clone this repo:
```bash
git clone https://github.com/ktlProject0/KTL_project_09_Pathology_ProstateCancer_Segmentation.git
cd KTL_project_09_Pathology_ProstateCancer_Segmentation
```
 - You can create a new Conda environment using `conda env create -f environment.yml`.
   
## Code Description
## Data_preprocessing_example
  - 고해상도 병리데이터 전처리 튜토리얼(예시) 코드
## Training.ipynb
  - 네트워크 학습 코드
## Evaluation.ipynb
  - 네트워크 성능 평가 및 전립선암 영역 분할 결과 가시화
  - 학습완료 된 모델 가중치 (/Code/output/model_final.pth)
## model_torch.py
  - EfficientNet B0 UNet 아키텍쳐 빌드
## Preprocessing.py
  - OpenSlide 라이브러리 이용 512*512 해상도의 crop 영상 데이터셋 확보
## _utils_torch.py
  - 네트워크를 구성에 필요한 부속 코드
  - Preprocessing 이후 데이터를 모델에 입력하기 위한 처리 (torch의 tensor로 변경)
## modules.torch.py
  - 네트워크를 구성에 필요한 부속 코드
## loss.py
  - 학습용 loss 함수 코드
  - Dice loss
  - Focal loss
