import argparse
import google.generativeai as genai
import PIL.Image
import os

# Set up argument parser
parser = argparse.ArgumentParser(description='Generate content using a prompt and image URL.')
parser.add_argument('--prompt', required=True, help='The prompt to send to the generative AI model.')
parser.add_argument('--image_url', required=True, help='The URL or path of the image to process.')

# Parse the arguments
args = parser.parse_args()

# Load the image
img = PIL.Image.open(args.image_url)

# Configure the generative AI model
genai.configure(api_key="<your_api_key>")

model = genai.GenerativeModel(model_name="gemini-1.5-flash")

# Generate content using the prompt and image
response = model.generate_content([args.prompt, img])
print(response.text)
