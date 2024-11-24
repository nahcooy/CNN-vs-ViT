### SAFECount 사용법 README

---

#### 1. 소개

SAFECount는 이미지 내 객체의 개수를 세는 딥러닝 모델로, 특정 객체를 효율적으로 세는 데 활용할 수 있습니다.  
본 문서는 SAFECount를 활용한 데이터 준비, 학습, 평가 및 시각화 과정을 설명합니다.

---

#### 2. 필수 준비물

1. **환경 설정**
   - PyTorch 및 필요한 라이브러리가 설치된 환경
   - Python 3.x
   - `labelImg` 설치: `pip install labelImg`

2. **리포지토리 다운로드**
   ```bash
   git clone https://github.com/kehuantiantang/SAFECount.git
   cd SAFECount
   ```

3. **데이터 다운로드**
   - 학습하고 싶은 image set을 다운로드하시오

---

#### 3. 데이터 준비

1. **Bounding Box 주석 작업**
   - 위에서 다운받은 모든 데이터에 대하여 annotaion을 수행
   - `labelImg` 실행 후 단축키 `w`를 사용하여 객체 주석 작업 수행.
   - 주석 작업이 완료되면 `.xml` 파일이 생성됩니다.

2. **exampler 데이터 변환**
   - 위의 과정에서 생성된 xml 파일 중 exampler로 사용할 데이터를 SAFECount 형식으로 변환:
     ```bash
     cd ./SAFECount/support_tools
     python safe_data_convert.py --input_dir {지원 xml 파일 폴더 경로} \
                                 --output_dir {결과 저장 경로}
     ```
   - 변환 후 생성된 `exemplar.json` 파일을 `SAFECount/data/Chicken` 폴더로 이동.

3. **train/test 데이터 변환 및 데이터 분할**
   - 위의 과정에서 생성된 xml 파일 중 query로 사용할 데이터를 SAFECount 형식으로 변환 및 분할:
   - train/test 데이터 변환 및 데이터 분할:
     ```bash
     python data_split.py --input_dir {query 이미지 폴더 경로} \
                          --output_dir {결과 저장 경로} \
                          --xml_dir {query xml 폴더 경로} \
                          --split_type random \
                          --test_suffix xml
     ```
   - `train.json` 및 `test.json` 그리고 `gt_density_map` 파일 생성.

4. **폴더 구조 준비**
   아래와 같은 폴더 구조로 데이터를 준비:
   ```
   SAFECount/
   └── data/
       └── Chicken/
           ├── frames/          # 이미지 파일
           ├── gt_density_map/  # 밀도 맵 파일
           ├── exemplar.json    # 지원 이미지 정보 파일
           ├── train.json       # 학습 데이터 정보 파일
           └── test.json        # 테스트 데이터 정보 파일
   ```

---

#### 4. 학습 설정

1. **설정 파일 수정**
   - `SAFECount/experiments/Chicken/config_exemplar.yaml` 파일 수정:
     - 데이터 경로 설정:
       ```yaml
       dataset:
         img_dir: "/home/{사용자}/SAFECount/data/Chicken/frames/"
         meta_file: "/home/{사용자}/SAFECount/data/Chicken/exemplar.json"
         ...
       ```
     - 체크포인트 저장 경로:
       ```yaml
       saver:
         save_dir: checkpoints/camera
         ...
       ```

2. **사전 학습된 가중치 사용 (선택 사항)**
   - 선택사항이지만, 원본 모델의 사전학습된 가중치를 사용하는 것이 좋다.
   - [가중치 다운로드 링크]([http://gofile.me/5RXEF/43uSWuEfs](https://drive.google.com/file/d/1mbV0xJdORIpSLlMCwlgENMB9Y1kUOhk2/view))  
   - 다운로드한 파일을 `checkpoints/camera/` 폴더에 저장.

---

#### 5. 모델 학습

1. **학습 시작**
   ```bash
   cd /SAFECount/experiments/Chicken
   bash train_torch_exemplar.sh
   ```
2. **결과 저장**
   - 학습 결과는 다음 경로에 저장:
     ```
     SAFECount/
     └── experiments/
         └── Chicken/
             ├── checkpoints/
             │   └── camera/
             │       ├── ckpt_best.pth.tar
             │       └── ckpt.pth.tar
             ├── log/
             └── vis/
     ```

---

#### 6. 평가 및 시각화

1. **모델 평가**
   ```bash
   cd /SAFECount/experiments/Chicken
   bash eval_torch_exemplar.sh
   ```

2. **시각화 결과 확인**
   - 시각화 결과는 `vis/camera` 폴더에 저장됩니다.

---

#### 7. 기타 참고 사항

1. **GPU 환경 확인**
   - PyTorch GPU 설정 확인:
     ```python
     import torch
     print(torch.cuda.is_available())  # True
     print(torch.cuda.get_device_name(0))  # GPU 이름 출력
     ```
