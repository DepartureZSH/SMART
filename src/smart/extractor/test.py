import requests
from PIL import Image
from transformers import Blip2Processor, Blip2ForConditionalGeneration
import pathlib
pretrained_model = "{}/pretrained_models/blip2-opt-2.7b".format(pathlib.Path(__file__).parent.parent.parent.parent.resolve())

processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
model = Blip2ForConditionalGeneration.from_pretrained(pretrained_model, device_map="auto")

img_url = 'https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg' 
raw_image = Image.open(requests.get(img_url, stream=True).raw).convert('RGB')

question = "how many dogs are in the picture?"
inputs = processor(raw_image, question, return_tensors="pt").to("cuda")

out = model.generate(**inputs)
print(processor.decode(out[0], skip_special_tokens=True).strip())