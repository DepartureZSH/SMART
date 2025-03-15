import pathlib
import os
os.environ['HF_ENDPOINT'] = 'https://hf-mirror.com'
image_folder = "{}/Datasets/VisualGenome".format(pathlib.Path(__file__).parent.parent.parent.parent.parent.resolve())
data_folder = "{}/data".format(pathlib.Path(__file__).parent.parent.parent.parent.resolve())
pretrained_model = "{}/pretrained_models/blip2-opt-2.7b".format(pathlib.Path(__file__).parent.parent.parent.parent.resolve())
import pandas as pd
import json
import torch
from PIL import Image
import numpy


devices = []
if torch.cuda.is_available():
    num_gpus = torch.cuda.device_count()
    devices = [torch.device(f'cuda:{i}') for i in range(num_gpus)]
else:
    devices = [torch.device('cpu')]
device = devices[0]
splits = ["val", "test"]

def read_tsv(file):
    return pd.read_csv(file, sep='\t', header=None)

def read_json(file):
    with open(file) as f:
        return json.load(f)

def write_csv(data, dest_file):
    data.to_csv(dest_file, sep='\t', quoting=3, header=False, index=False)

def Blip2_Caption_globally(model, processor):
    for split in splits:
        data = read_tsv(f"{data_folder}/obj_feat_{split}.tsv")
        image_mapping = read_json(f"{data_folder}/Features/idx_image_id_mapping.json")
        for ind in range(len(data)):
            print(f"Processing {ind}/{len(data)}")
            objects_list = []
            indx = data.iloc[ind, 0]
            image_path = image_mapping[str(indx)]
            objects = json.loads(data.iloc[ind, 1])
            image = Image.open(f"{image_folder}/{image_path}").convert('RGB') # PIL Image
            prompt = "Question: What is the relationship between the main objects in the image? Answer:"
            inputs = processor(images=image, text=prompt, return_tensors="pt").to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=20)
            caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
            caption = caption.split("Answer:")[-1]
            objects_list.append({"G_caption": caption})
            data.iloc[ind, 1] = json.dumps({"objects": objects_list}, ensure_ascii=False)
        write_csv(data, f"{data_folder}/Features/BLIP2/Global/obj_feat_{split}.tsv")

def Blip2_Caption_locally(model, processor):
    for split in splits:
        data = read_tsv(f"{data_folder}/Features/obj_feat_{split}.tsv")
        image_mapping = read_json(f"{data_folder}/Features/idx_image_id_mapping.json")
        prompt = "Question: What is the relationship between the main objects in the image? Answer:"
        for ind in range(len(data)):
            print(f"Processing {ind}/{len(data)}")
            objects_list = []
            indx = data.iloc[ind, 0]
            image_path = image_mapping[str(indx)]
            objects = json.loads(data.iloc[ind, 1])
            image = Image.open(f"{image_folder}/{image_path}").convert('RGB') # PIL Image
            for j, obj in enumerate(objects["objects"]):
                image_patch = image.crop(obj["rect"])
                inputs = processor(images=image_patch, text=prompt, return_tensors="pt").to(device)
                generated_ids = model.generate(**inputs, max_new_tokens=20)
                caption = processor.batch_decode(generated_ids, skip_special_tokens=True)[0].strip()
                objects_list.append({"caption": caption})
            data.iloc[ind, 1] = json.dumps({"objects": objects_list}, ensure_ascii=False)
        write_csv(data, f"{data_folder}/Features/BLIP2/Local/obj_feat_{split}.tsv")

# def Blip2_Predict(data, model, device, batch_size=64):
#     model.to(device)
#     model.eval()
#     num_batches = len(data) // batch_size
#     image = vis_processors["eval"](raw_image).unsqueeze(0).to(device)

def validate(file):
    data = pd.read_csv(file, sep='\t', header=None)
    image_mapping = read_json(f"{data_folder}/Features/idx_image_id_mapping.json")
    ind = 199
    indx = data.iloc[ind, 0]
    image_path = image_mapping[str(indx)]
    objects = json.loads(data.iloc[ind, 1])
    print(objects)

if __name__ == '__main__':
    # Load the data
    from transformers import Blip2Processor, Blip2ForConditionalGeneration
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-opt-2.7b")
    model = Blip2ForConditionalGeneration.from_pretrained(
        pretrained_model
    )
    
    model.to(device)
    # Blip2_Caption_globally(model, processor)
    Blip2_Caption_locally(model, processor)
    # validate(f"{data_folder}/Features/obj_feat_val.tsv")