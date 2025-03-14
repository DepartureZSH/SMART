# Installation

## Requirements


## Setup

```bash
conda create -y --name SMART python=3.12
conda activate SMART

conda install pytorch==2.5.1 torchvision==0.20.1 torchaudio==2.5.1 pytorch-cuda=12.1 -c pytorch -c nvidia

# pip install ftfy regex tqdm
# pip install git+https://github.com/openai/CLIP.git
# clip @ git+https://github.com/openai/CLIP.git@dcba3cb2e2827b402d2701e7e1c7d9fed8a20ef1
pip install -r requirements.txt

cd src/
pip install -e .

# Blip Caption
conda create -n lavis python=3.9
conda activate lavis

pip install salesforce-lavis
```