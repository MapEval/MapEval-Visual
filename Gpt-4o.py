import base64
import requests
import PIL.Image
import os
import argparse

parser = argparse.ArgumentParser(description='Generate content using a prompt and image URL.')
parser.add_argument('--prompt', required=True, help='The prompt to send to the generative AI model.')
parser.add_argument('--image_url', required=True, help='The URL or path of the image to process.')
args = parser.parse_args()

# OpenAI API Key
api_key = "<your_api_key>"

# Function to encode the image
def encode_image(image_path):
  with open(image_path, "rb") as image_file:
    return base64.b64encode(image_file.read()).decode('utf-8')

# Path to your image
image_path = args.image_url

# Getting the base64 string
base64_image = encode_image(image_path)

headers = {
  "Content-Type": "application/json",
  "Authorization": f"Bearer {api_key}"
}

payload = {
  "model": "gpt-4o",
  "messages": [
    {
      "role": "user",
      "content": [
        {
          "type": "text",
          "text": args.prompt
        },
        {
          "type": "image_url",
          "image_url": {
            "url": f"data:image/jpeg;base64,{base64_image}"
          }
        }
      ]
    }
  ],
  "max_tokens": 300
}
print("hello world")

response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
response = response.json()
print(response)
print(response['choices'][0]['message']['content'])