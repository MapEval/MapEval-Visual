import torch
from PIL import Image
from transformers import AutoModel, AutoTokenizer
import argparse

# Define function to load model and tokenizer
def load_model(device='cuda'):
    model_path = 'openbmb/MiniCPM-Llama3-V-2_5'
    model = AutoModel.from_pretrained(model_path, trust_remote_code=True).to(device=device)
    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    model.eval()
    return model, tokenizer

# Define function to process image and text
def process_image_and_text(image_path, text, model, tokenizer, device='cuda', params=None):
    if params is None:
        params = {
            'num_beams': 3,
            'repetition_penalty': 1.2,
            'max_new_tokens': 1024,
            'top_p': 0.8,
            'top_k': 100,
            'temperature': 0.7,
            'sampling': True,
            'stream': False
        }

    try:
        # Load and preprocess the image
        image = Image.open(image_path).convert('RGB')

        # Prepare the input context with user input text
        context = [{"role": "user", "content": text}]
        
        # Generate response from the model
        with torch.no_grad():
            answer = model.chat(
                image=image,
                msgs=context,
                tokenizer=tokenizer,
                **params
            )

        return answer
    except Exception as err:
        print(err)
        return "Error, please retry"

# Main function to execute the script
def main(image_path, text, device='cuda'):
    model, tokenizer = load_model(device)
    response = process_image_and_text(image_path, text, model, tokenizer, device)
    print("Model Response:", response)

if __name__ == "__main__":
    import argparse

    # Argument parser for command-line arguments
    parser = argparse.ArgumentParser(description="Image Description Generator")
    parser.add_argument('--prompt', type=str, required=True, help='The query prompt for the model.')
    parser.add_argument('--image_url', type=str, required=True, help='The path to the image file.')
    args = parser.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Run the main function
    main(args.image_url, args.prompt, device)
