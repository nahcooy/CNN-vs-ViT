### SAFECount 사용법 README

---

#### 1. 소개

SAFECount는 이미지 내 객체의 개수를 세는 딥러닝 모델로, 특정 객체를 효율적으로 세는 데 활용할 수 있습니다.  
본 문서는 SAFECount를 활용한 데이터 준비, 학습, 평가 및 시각화 과정을 설명합니다.

[원본 git 주소](https://github.com/zhiyuanyou/SAFECount)

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
       ```yaml
       port: 22222
       random_seed: 131

       dataset:
         type: custom_exemplar
         exemplar:
           # the absolute path of frames folder in your computer
           img_dir: &img_train_dir /home/khtt/code/SAFECount/data/Chicken/camera/frames/
           # the absolute path of exemplar.json in your computer
           meta_file: /home/khtt/code/SAFECount/data/Chicken/camera/exemplar.json
           norm: True
           num_exemplar: 8
         input_size:  [512, 512] # [h, w]
         pixel_mean: [0.485, 0.456, 0.406]
         pixel_std: [0.229, 0.224, 0.225]
         batch_size: 1
         workers: 6

         train:
           img_dir: *img_train_dir
           # the path of gt_density_map folder in your computer
           density_dir: /home/khtt/code/SAFECount/data/Chicken/camera/gt_density_map/
           # the absolute path of train.json in your computer
           meta_file: /home/khtt/code/SAFECount/data/Chicken/camera/train.json
           hflip:
             prob: 0.5
           vflip:
             prob: 0.5
           rotate:
             degrees: 10
           colorjitter:
             brightness: 0.1
             contrast: 0.1
             saturation: 0.1
             hue: 0.1
             prob: 0.5
           gamma:
             range: [0.75, 1.5]
             prob: 0.5
           gray:
             prob: 0.5
         val:
           # the absolute path of frames folder in your computer
           img_dir: &img_test_dir /home/khtt/code/SAFECount/data/Chicken/camera/frames/
           # the absolute path of gt_density_map folder in your computer
           density_dir: /home/khtt/code/SAFECount/data/Chicken/camera/gt_density_map/
           # the absolute path of test.json folder in your computer
           meta_file: /home/khtt/code/SAFECount/data/Chicken/camera/test.json

       criterion:
         - name: _MSELoss
           type: _MSELoss
           kwargs:
             outstride: 1
             weight: 250

       trainer:
         epochs: 1000
         lr_scale_backbone: 0 # 0: frozen, 0.1: 0.1 * lr, 1: lr
         optimizer:
           type: Adam
           kwargs:
             lr: 0.0002
         lr_scheduler:
           type: StepLR
           kwargs:
             step_size: 400
             gamma: 0.25

       saver:
         auto_resume: False
         always_save: False
         # path where to load the pretrain file
         load_path: checkpoints/camera/ckpt_best.pth.tar
         # path where the save checkpoint
         save_dir: checkpoints/camera
         # path where to store the log
         log_dir: log/camera

       evaluator:
         save_dir: result_eval_temp

       visualizer:
         # path where to store visualization image, should use eval_torch_exemplar.sh to show
         vis_dir: vis/camera
         img_dir: *img_test_dir
         activation: sigmoid # [null, sigmoid]
         normalization: True
         with_image: True

       net:
         builder: models.safecount_exemplar.build_network
         kwargs:
           block: 2
           backbone:
             type: resnet18
             out_layers: [1, 2, 3]
             out_stride: 4
           pool:
             type: max
             size: [1, 1]
           embed_dim: 256
           mid_dim: 1024
           head: 8
           dropout: 0.1
           activation: leaky_relu
           exemplar_scales: []
           initializer:
             method: normal
             std: 0.001
       ```


2. **사전 학습된 가중치 사용 (선택 사항)**
   - 선택사항이지만, 원본 모델의 사전학습된 가중치를 사용하는 것이 좋다.
   - [가중치 다운로드 링크](https://drive.google.com/file/d/1mbV0xJdORIpSLlMCwlgENMB9Y1kUOhk2/view) 
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
