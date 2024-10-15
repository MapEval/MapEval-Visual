import torch
from transformers import AutoModel, AutoTokenizer
import argparse

def generate_image_description(query, image_url):
    torch.set_grad_enabled(False)

    # Init model and tokenizer
    model = AutoModel.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True).eval()
    tokenizer = AutoTokenizer.from_pretrained('internlm/internlm-xcomposer2-vl-7b', trust_remote_code=True)

    query = '<ImageHere>' + query
    image = image_url

    with torch.cuda.amp.autocast():
        response, _ = model.chat(tokenizer, query=query, image=image, history=[], do_sample=False)
    
    return response

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Description Generator")
    parser.add_argument('--prompt', type=str, required=True, help='The query prompt for the model.')
    parser.add_argument('--image_url', type=str, required=True, help='The path to the image file.')
    args = parser.parse_args()

    # Generate image description using Ray
    result = (generate_image_description(args.prompt, args.image_url))
    print(result)


if __name__ == '__main__':
    main()
