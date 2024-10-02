import torch
from PIL import Image
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.generation import GenerationConfig

# Set random seed for reproducibility
torch.manual_seed(1234)

# Load the tokenizer and model for Qwen-VL-Chat
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen-VL-Chat", device_map="cuda", trust_remote_code=True).eval()
model.generation_config = GenerationConfig.from_pretrained("Qwen/Qwen-VL-Chat", trust_remote_code=True)

# Function for performing inference using image_url and prompt as inputs
def inference(prompt, image_url):
    try:
        # Load and convert the image to RGB mode
        image = Image.open(image_url).convert("RGB")

        # Prepare the prompt by adding image and query in the expected format
        query = tokenizer.from_list_format([
            {'image': image_url},  # Local path or URL for the image
            {'text': prompt},
        ])

        # Generate the response using the model
        response, history = model.chat(tokenizer, query=query, history=None)

        # Return the model's response
        return response.strip()
    
    except Exception as e:
        print(f"Error during inference: {e}")
        return None

# Main function to parse arguments and run inference
if __name__ == "__main__":
    # Initialize argument parser
    parser = argparse.ArgumentParser(description="Inference using Qwen-VL-Chat with image and prompt.")
    
    # Add arguments for image_url and prompt
    parser.add_argument('--image_url', type=str, required=True, help="Path or URL to the image")
    parser.add_argument('--prompt', type=str, required=True, help="Text prompt to be used with the image")
    
    # Parse the arguments
    args = parser.parse_args()

    # Call the inference function with the provided prompt and image URL
    response = inference(args.prompt, args.image_url)
    
    # Output the result
    if response:
        print(f"Model Response: {response}")
    else:
        print("No response generated.")
