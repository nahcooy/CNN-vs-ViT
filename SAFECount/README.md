# SAFECount 사용법 README

---

## 1. 소개

SAFECount는 이미지 내 객체의 개수를 세는 딥러닝 모델로, 특정 객체를 효율적으로 세는 데 활용할 수 있습니다.  본 문서는 SAFECount를 활용한 데이터 준비, 학습, 평가 및 시각화 과정을 설명합니다.

---

## 2. 필수 준비물

1. **환경 설정**
   - PyTorch 및 필요한 라이브러리가 설치된 환경
   - Python 3.x
   - `labelImg` 설치: `pip install labelImg`

2. **리포지토리 다운로드**
   ```bash
   git clone https://github.com/kehuantiantang/SAFECount.git
   cd SAFECount
