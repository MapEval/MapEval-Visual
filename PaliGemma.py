import pandas as pd
from transformers import AutoProcessor, PaliGemmaForConditionalGeneration
from PIL import Image
import torch
from huggingface_hub import login
import argparse

# Log in to Hugging Face
login()

# Load the model and processor
model_id = "google/paligemma-3b-mix-224"
model = PaliGemmaForConditionalGeneration.from_pretrained(model_id).eval()
processor = AutoProcessor.from_pretrained(model_id)

# Function for performing inference with image_url and prompt as inputs
def inference(prompt, image_url):
    try:
        # Load and convert the image to RGB mode
        image = Image.open(image_url).convert("RGB")

        # Prepare the prompt
        prompt_text = (
            f"You will be given one image, one question and 4 options. "
            f"You need to answer the question by choosing the correct option (give visual quotation marks around) and giving proper reasoning. "
            f"This is the question: {prompt}. Choose the correct option with the answer and give a proper reason in brief."
        )

        # Prepare model inputs
        model_inputs = processor(text=prompt_text, images=image, return_tensors="pt")
        input_len = model_inputs["input_ids"].shape[-1]
        generation = model.generate(**model_inputs, max_new_tokens=100, do_sample=False)
        generation = generation[0][input_len:]
        decoded = processor.decode(generation, skip_special_tokens=True)

        return decoded

    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Main function to parse arguments and run inference
if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Perform inference using PaliGemma with image and prompt.")

    # Add arguments for image_url and prompt
    parser.add_argument('--image_url', type=str, required=True, help="URL or path to the image file.")
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt to be used with the image.")

    # Parse the arguments
    args = parser.parse_args()

    # Run the inference with the provided arguments
    response = inference(args.prompt, args.image_url)

    # Output the response
    if response:
        print(f"Model Response: {response}")
    else:
        print("No response generated.")
