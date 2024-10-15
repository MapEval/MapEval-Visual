import base64
import anthropic
import argparse

def main():
    parser = argparse.ArgumentParser(description="Send an image and text prompt to Anthropic API.")
    
    parser.add_argument('--prompt', required=True, help="Text prompt describing the image.")
    parser.add_argument('--image_url', required=True, help="Path to the image file.")
    
    args = parser.parse_args()
    
    image_path = args.image_url
    prompt_text = args.prompt
    
    with open(image_path, "rb") as image_file:
        image_data = base64.b64encode(image_file.read()).decode("utf-8")
    
    if image_path.endswith('.PNG'):
        image_media_type = "image/png"
    elif image_path.endswith('.jpg') or image_path.endswith('.jpeg'):
        image_media_type = "image/jpeg"
    else:
        raise ValueError("Unsupported image format. Only PNG and JPEG are supported.")
    
    client = anthropic.Anthropic(api_key="<your api key>")
    
    message = client.messages.create(
        model="claude-3-5-sonnet-20240620",
        max_tokens=1024,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": image_media_type,
                            "data": image_data,
                        },
                    },
                    {
                        "type": "text",
                        "text": prompt_text
                    }
                ],
            }
        ],
    )
    
    print(message.content[0].text)

if __name__ == "__main__":
    main()
