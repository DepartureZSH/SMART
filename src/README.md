```python
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