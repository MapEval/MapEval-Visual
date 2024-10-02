import torch
from PIL import Image
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Description Generator")
    parser.add_argument('--prompt', type=str, required=True, help='The query prompt for the model.')
    parser.add_argument('--image_url', type=str, required=True, help='The path to the image file.')
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("THUDM/glm-4v-9b", trust_remote_code=True)

    # Load image
    image = Image.open(args.image_url).convert('RGB')
    
    # Prepare inputs
    query = args.prompt
    inputs = tokenizer.apply_chat_template(
        [{"role": "user", "image": image, "content": query}],
        add_generation_prompt=True,
        tokenize=True,
        return_tensors="pt",
        return_dict=True  # chat mode
    )

    inputs = inputs.to(device)

    # Load model
    model = AutoModelForCausalLM.from_pretrained(
        "THUDM/glm-4v-9b",
        torch_dtype=torch.bfloat16,
        low_cpu_mem_usage=True,
        trust_remote_code=True
    ).to(device).eval()

    gen_kwargs = {"max_length": 2500, "do_sample": True, "top_k": 1}
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(**inputs, **gen_kwargs)
        outputs = outputs[:, inputs['input_ids'].shape[1]:]
        print(tokenizer.decode(outputs[0], skip_special_tokens=True))

if __name__ == '__main__':
    main()
