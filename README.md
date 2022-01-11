# abnormal_motion_detection_3rd
사람의 쓰러짐 여부를 판별할 수 있는 모델을 배포합니다.
경량화 모델을 사용하여 판별 속도가 향상 되었습니다.

## Requirements
numpy==1.18.5   
opencv-python==4.4.0.42   
opencv-python-headless==4.4.0.44   
glob2==0.7   
pytorch==1.10.0   
pandas==1.1.1   
albumentations==0.5.0   
pytorchcv==0.0.58   
tqdm==4.48.2   
seaborn==0.11.2

## How to install

```
git clone https://github.com/nuualab/abnormal_motion_detection_3rd
```

## How to run
```
python main.py
```
example 폴더의 연속된 이미지 파일을 읽어 추론 후 answer.json 파일을 생성 합니다.  

## Pretrained Model Download
weights/falldown_classification 디렉토리에 저장   
[efficientnetb4b_fall_detection.pth](https://drive.google.com/file/d/1aAcbP8E-g2BHUmoHCVGydUVSCy4g3vh0/view?usp=sharing, "efficientnetb4b_fall_detection.pth")   
     
weights/yolov5 디렉토리에 저장   
[yolov5x_human_detection.pt](https://drive.google.com/file/d/1x_B1vepkkI4An_7ApexVxYQdaJWgcck0/view?usp=sharing, "yolov5x_human_detection.pt")   
   
## License
이 프로젝트는 Apache 2.0 라이선스를 따릅니다. 모델 및 코드를 사용할 경우 라이선스 내용을 준수해주세요. 라이선스 전문은 LICENSE 파일에서 확인하실 수 있습니다.

*이 프로젝트는 과학기술정보통신부 인공지능산업원천기술개발사업의 지원을 통해 제작 되었습니다.*
