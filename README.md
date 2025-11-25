## 🛠️ Requirements

### Environment
- **Python** 3.8+
- **PyTorch** 1.13.0+
- **CUDA** 11.6+
- **Ubuntu** 18.04 or higher / Windows 10

### Installation

```bash
# Create conda environment
conda create -n dccs python=3.8 -y
conda activate dccs

# Install PyTorch
pip install torch==1.13.0 torchvision==0.14.0 torchaudio==0.13.0

# Install dependencies
pip install packaging
pip install timm==0.4.12
pip install pytest chardet yacs termcolor
pip install submitit tensorboardX
pip install triton==2.0.0
pip install causal_conv1d==1.0.0
pip install mamba_ssm==1.0.1

# Or simply run
pip install -r requirements.txt
```

## 📁 Dataset Preparation

We evaluate our method on three public datasets: **NUDT-SIRST**, **IRSTD-1K**, and **SIRST-Aug**.

| Dataset | Link |
|---------|------|
| IRSTD-1K | [Download](https://github.com/RuiZhang97/ISNet) |
| NUAA-SIRST | [Download](https://github.com/YimianDai/sirst) |
| SIRST-Aug | [Download](https://github.com/Tianfang-Zhang/AGPCNet) |

Please organize the datasets as follows:

```
├── dataset/
│    ├── IRSTD-1K/
│    │    ├── images/
│    │    │    ├── XDU514png
│    │    │    ├── XDU646.png
│    │    │    └── ...
│    │    ├── masks/
│    │    │    ├── XDU514.png
│    │    │    ├── XDU646.png
│    │    │    └── ...
│    │    └── trainval.txt
│    │    └── test.txt
│    ├── NUAA-SIRST/
│    │    └── ...
│    └── SIRST-Aug/
│         └── ...
```

## 🚀 Training

```bash
python main.py --dataset-dir '/path/to/dataset' \
               --batch-size 4 \
               --epochs 400 \
               --lr 0.05 \
               --mode 'train'
```

**Example:**
```bash
python main.py --dataset-dir './dataset/IRSTD-1K' --batch-size 4 --epochs 400 --lr 0.05 --mode 'train'
```

## 📊 Testing

```bash
python main.py --dataset-dir '/path/to/dataset' \
               --batch-size 4 \
               --mode 'test' \
               --weight-path '/path/to/weight.tar'
```

**Example:**
```bash
python main.py --dataset-dir './dataset/IRSTD-1K' --batch-size 4 --mode 'test' --weight-path './weight/irstd1k_weight.pkl'
```

## 📈 Results

### Quantitative Results

| Dataset | IoU (×10⁻²) | Pd (×10⁻²) | Fa (×10⁻⁶) | Weights |
|:-------:|:------------:|:----------:|:----------:|:-------:|
| IRSTD-1K | 69.64 | 95.58 | 10.48 | [Download](https://drive.google.com/file/d/1KqlOVWIktfrBrntzr53z1eGnrzjWCWSe/view?usp=sharing) |
| NUAA-SIRST | 78.65 | 78.65 | 2.48 | [Download](https://drive.google.com/file/d/13JQ3V5xhXUcvy6h3opKs15gseuaoKrSQ/view?usp=sharing) |
| SIRST-Aug | 75.57 | 98.90 | 33.46 | [Download](https://drive.google.com/file/d/1lcmTgft0LStM7ABWDIMRHTkcOv95p9LO/view?usp=sharing) |


## 📂 Project Structure

```
DCCS/
├── dataset/          # Dataset loading and preprocessing
├── model/            # Network architecture
├── utils/            # Utility functions
├── weight/           # Pretrained weights
├── main.py           # Main entry point
├── requirements.txt  # Dependencies
└── README.md
```

## 🙏 Acknowledgement

We sincerely thank the following works for their contributions:

- [BasicIRSTD](https://github.com/XinyiYing/BasicIRSTD) - A comprehensive toolbox for infrared small target detection
- [MSHNet](https://github.com/ying-fu/MSHNet) - Scale and Location Sensitive Loss
