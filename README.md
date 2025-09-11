
<img width="600" alt="Image" src="https://github.com/user-attachments/assets/8d8920e4-aa4b-4c25-9046-c3aa9a5f6cf7" />
# 관심영역 기반 위험 예측 관제 시스템 개발

## 프로젝트 진행 프로세스

1. 타임 테이블 작성 (세부 일정 확정)
2. 모델 개량, ROI기반 탐지, 관제시스템/각 팀별 작업
3. README.md 작성 (프로젝트 개요, 목표, 팀원 역할 포함)
4. 주제 선정 (인파 밀집의 세부 주제 확정)
5. 팀장 선정 및 역할 분배
6. 데이터 수집 및 전처리 담당
7. 모델 설계 및 학습 담당
8. 시스템 운영 및 실시간 장애 감지 방안
9. 시스템 통합 및 구현 담당
10. 발표 자료 제작 및 문서화 담당
11. 1차 발표 (9월 10일)
12. 1차 발표 후 피드백 수집 및 정리
13. 최종 발표 (9월 11일)
14. 최종 모델 및 결과 발표

## 타임 테이블
- [타임테이블](https://www.notion.so/2683c414e0cc802f80ebcb913700dbee?v=2683c414e0cc802da810000c5abdc096&source)
<img width="600" alt="Image" src="https://github.com/user-attachments/assets/eb7b2e1f-28da-4528-8005-50b17e7ffdaa" />
## 프로젝트 일정 수립
| 날짜 | 내용 |
|--|--|
| 9월 3일 | 주제선정 및 역할 분담 |
| 9월 4 ~ 5일 | 데이터 수집 및 전처리<br>탐지시스템 개발 시작 <br>관제시스템 개발 시작 |
| 9월 5 ~ 6일 | 1창 모델 훈련 |
| 9월 8~9일 |데이터 경량화 및 2차 모델 훈련|
| 9월 10일 | 모델 선정 및 중간발표 |
| 9월 12일 | 최종발표 |

- 프로젝트 기간: 2025.09.03 ~ 2025.09.12
- 위 일정에 따라 프로젝트를 진행하며, 각 단계별 진행 상황을 지속적으로 공유할 예정입니다.

## 기술 스텍 & 협업 환경

- 프레임워크 : PyTorch
- 개발 도구 : Python, OpenCV, YOLOv11, RT-DETR, flask
- 데이터셋 관리 : AIHub
- 협업 도구 : Notion


## 발표준비
```

```

## 테스트 및 평가
```
프로젝트 성능 평가 지표 확정 (정확도, 속도 등)
```

## 구성원
- 팀장 : 이효찬
- 팀원 : 구슬기, 김예령, 한믿음, 김지수, 황재영, 전강호


## 프로젝트 소개

- 프로젝트 주제 : 인파밀집 경고 시스템
- 프로젝트 명 : 관심영역 기반 위험 예측 관제 시스템 개발
- 프로잭트 목표 :
   - YOLO v11 모델을 **전이학습**시켜 **광장**, **골목길** 에서의 **이상징후를 탐지**하고, **이를 알리는 관제시스템**
   - **골목길**에서의 **인파밀집 예측 및 탐지**
   - **광장** 에서의 **이상 징후 탐지**
   - 해당 사항 탐지 시 **관제시스템에 알림 표출**
   - 전광판 및 알림판에 **탐지내용 송출**
  
## 환경 셋팅

- Flask==3.1.2
- numpy==2.3.3
- opencv_python==4.12.0.88
- torch==2.8.0+cu128
- ultralytics==8.3.197


## 환경 설치
```
pip install -r requirements.txt
```

## 실행
```
cd Python_Project
python app_tensorrt_live.py
```

## 영상 접근
1. [youtube](https://www.youtube.com/live/rnXIjl_Rzy4?si=lRcDqlANkukTPBAx)
   
### 접근 방법
```
youtube 영상 접근 => 공유 => 링크 복사 => subprocess로 링크 변환 => cv2.CaptureVideo() 적용 => 영상 스트리밍
```

### 예시 링크
```
youtube_live_url = "https://www.youtube.com/live/rnXIjl_Rzy4?si=lRcDqlANkukTPBAx"
```
