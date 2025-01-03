import torch
from PIL import Image
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
import argparse

def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Image Description Generator")
    parser.add_argument('--prompt', type=str, required=True, help='The query prompt for the model.')
    parser.add_argument('--image_url', type=str, required=True, help='The path to the image file.')
    args = parser.parse_args()

    # Set device
    device = "cuda" if torch.cuda.is_available() else "cpu"

    # Load model and processor
    model = Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-7B-Instruct",
        torch_dtype="auto",
        device_map="auto"
    ).to(device).eval()

    processor = AutoProcessor.from_pretrained("Qwen/Qwen2-VL-7B-Instruct")

    # Load image
    image = Image.open(args.image_url).convert('RGB')

    # Prepare conversation template
    conversation = [
        {
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": args.prompt}
            ]
        }
    ]

    # Prepare inputs
    text_prompt = processor.apply_chat_template(conversation, add_generation_prompt=True)
    inputs = processor(
        text=[text_prompt],
        images=[image],
        padding=True,
        return_tensors="pt"
    ).to(device)

    # Inference: Generate response
    gen_kwargs = {"max_new_tokens": 128}
    with torch.no_grad():
        output_ids = model.generate(**inputs, **gen_kwargs)
        generated_ids = [
            output_ids[len(input_ids):]
            for input_ids, output_ids in zip(inputs.input_ids, output_ids)
        ]
        output_text = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True
        )

    # Print output
    print(output_text[0])

if __name__ == '__main__':
    main()
