import torch
from PIL import Image
import requests
import argparse
from transformers import AutoProcessor, LlavaForConditionalGeneration

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate response from image and prompt')
parser.add_argument('--image_url', type=str, required=True, help='The URL of the image')
parser.add_argument('--prompt', type=str, required=True, help='The prompt for the model')

args = parser.parse_args()

model_id = "llava-hf/llava-1.5-7b-hf"
model = LlavaForConditionalGeneration.from_pretrained(model_id)
processor = AutoProcessor.from_pretrained(model_id)

# Define messages with the given prompt
prompt = "USER: <image>\n"+args.prompt+"? ASSISTANT:"
url = args.image_url
image = Image.open(url)

inputs = processor(text=prompt, images=image, return_tensors="pt")
generate_ids = model.generate(**inputs, max_new_tokens=40)
response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]

print(response)
