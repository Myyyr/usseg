## 3D Ultrasound medical image segmentation

### 0. Installation

```
python -m venv usenv
source env/bin/activate

pip install torch==1.10.2+cu113 torchvision==0.11.3+cu113 torchaudio==0.10.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html


pip install -r requirements.txt
```


### 1. Training


multi-run
```
python main.py -m training.lr=1,1e-1

```