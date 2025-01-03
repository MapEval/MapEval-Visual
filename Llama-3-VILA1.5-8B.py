import argparse
import json
import os

import torch
from PIL import Image

from llava.constants import IMAGE_TOKEN_INDEX
from llava.conversation import SeparatorStyle, conv_templates
from llava.mm_utils import (KeywordsStoppingCriteria, process_images,
                            tokenizer_image_token)
from llava.model.builder import load_pretrained_model

DEFAULT_IMAGE_PATCH_TOKEN = "<im_patch>"

def eval_model(args, model, tokenizer, image_processor):
    # Load image
    image = Image.open(args.image_url).convert("RGB")
    image_tensor = process_images([image], image_processor, model.config)

    # Prepare conversation template
    if args.prompt:
        query_text = args.prompt
        conv = conv_templates["vicuna_v1"].copy()
        conv.append_message(conv.roles[0], query_text)
        conv.append_message(conv.roles[1], None)
        prompt = conv.get_prompt()
    else:
        raise ValueError("Prompt is required.")

    print("%" * 10 + " " * 5 + "VILA Response" + " " * 5 + "%" * 10)

    # Tokenize input
    inputs = tokenizer_image_token(prompt, tokenizer, IMAGE_TOKEN_INDEX)
    input_ids = torch.as_tensor(inputs).cuda().unsqueeze(0)

    # Set stopping criteria
    stop_str = conv.sep if conv.sep_style != SeparatorStyle.TWO else conv.sep2
    keywords = [stop_str]
    stopping_criteria = KeywordsStoppingCriteria(keywords, tokenizer, input_ids)

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=image_tensor.to(dtype=torch.float16, device="cuda", non_blocking=True),
            do_sample=True if args.temperature > 0 else False,
            temperature=args.temperature,
            top_p=0.7,
            max_new_tokens=512,
            stopping_criteria=[stopping_criteria],
        )

    # Decode and print results
    outputs = tokenizer.batch_decode(output_ids, skip_special_tokens=True)[0]
    outputs = outputs.strip()

    print(f"Question: {query_text}")
    print(f"VILA output: {outputs}")

    return {"question": query_text, "output": outputs}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--image_url", type=str, required=True, help="Path to the input image.")
    parser.add_argument("--prompt", type=str, required=True, help="Prompt for the model.")
    parser.add_argument("--temperature", type=float, default=0.2, help="Sampling temperature.")

    args = parser.parse_args()

    # Hardcoded model name
    model_name = "Llama-3-VILA1.5-8B"

    # Load model, tokenizer, and image processor
    tokenizer, model, image_processor, context_len = load_pretrained_model(model_name, "llava_llama", None)

    # Evaluate the model
    result = eval_model(args, model, tokenizer, image_processor)

    # Save the result
    save_name = "inference-result.json"
    with open(save_name, "w") as f:
        json.dump(result, f, indent=2)

    print(f"Results saved to {save_name}")
