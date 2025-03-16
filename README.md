# Installation

## Requirements

- Linux
- Python
- conda
- git
- cuda >= 12.1

## Training Setup

```bash
cd {PATH TO SMART}
conda create -y --name SMART python=3.12
conda activate SMART

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

pip install -r requirements.txt

cd src/
pip install -e .

```

## Blip2 Caption Setup
```
cd {PATH TO SMART}
conda create -n lavis python=3.9
conda activate lavis
python -m pip install --upgrade pip

python -m pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121
python -m pip install accelerate==1.0.1
python -m pip install bitsandbytes==0.44.1
python -m pip install transformers==4.30.0

python -m pip install salesforce-lavis

cd output/
git clone https://hf-mirror.com/Salesforce/blip2-opt-2.7b

```