#!/usr/bin/env python3
import os
import base64
import sys
from dotenv import load_dotenv
from openai import OpenAI

def encode_image(image_path):
    """Encode image to base64 string"""
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')

def analyze_image(image_path, user_prompt, model="gpt-5"):
    """
    Send image to OpenAI Vision API with user prompt

    Args:
        image_path: Path to the image file
        user_prompt: The prompt/question about the image
        model: OpenAI model to use (default: gpt-5)
    """
    load_dotenv()

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OPENAI_API_KEY not found in .env file")
        sys.exit(1)

    client = OpenAI(api_key=api_key)

    if not os.path.exists(image_path):
        print(f"Error: Image file not found: {image_path}")
        sys.exit(1)

    base64_image = encode_image(image_path)

    try:
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "text",
                            "text": user_prompt
                        },
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ],
            max_completion_tokens=1000
        )

        return response.choices[0].message.content

    except Exception as e:
        print(f"Error calling OpenAI API: {e}")
        sys.exit(1)

def main():
    if len(sys.argv) < 3:
        print("Usage: python vision_analyzer.py <image_path> <prompt>")
        sys.exit(1)

    image_path = sys.argv[1]
    user_prompt = " ".join(sys.argv[2:])

    print(f"\nAnalyzing image: {image_path}")
    print(f"Prompt: {user_prompt}\n")

    result = analyze_image(image_path, user_prompt)

    print("Response:")
    print("-" * 60)
    print(result)
    print("-" * 60)

if __name__ == "__main__":
    main()
