import torch
from PIL import Image
from transformers import LlavaNextProcessor, LlavaNextForConditionalGeneration
import argparse

# Load the processor and model
processor = LlavaNextProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
model = LlavaNextForConditionalGeneration.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf", torch_dtype=torch.float16, low_cpu_mem_usage=True) 
model.to("cuda:0")

# Function for performing inference with image_url and prompt as inputs
def inference(prompt, image_url):
    try:
        # Load and convert the image to RGB mode
        image = Image.open(image_url).convert("RGB")

        # Format the prompt for the model
        formatted_prompt = "[INST] <image>\n" + prompt + " [/INST]"

        # Prepare the inputs for the model
        inputs = processor(formatted_prompt, image, return_tensors="pt").to("cuda:0")

        # Generate the output using the model
        output = model.generate(**inputs, max_new_tokens=100)

        # Decode the output
        decoded_output = processor.decode(output[0], skip_special_tokens=True)

        # Extract the part after [/INST]
        response_part = decoded_output.split('[/INST]')[-1].strip()

        # Return the final response
        return response_part

    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Main function to parse arguments and run inference
if __name__ == "__main__":
    # Initialize the argument parser
    parser = argparse.ArgumentParser(description="Perform inference using LlavaNext with image and prompt.")

    # Add arguments for image_url and prompt
    parser.add_argument('--image_url', type=str, required=True, help="Path to the image file.")
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
