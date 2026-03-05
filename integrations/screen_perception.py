import base64
from pathlib import Path
from typing import Dict

from openai import OpenAI


class ScreenPerception:
    def __init__(self, client: OpenAI) -> None:
        self.client = client

    def describe_screen(self, screenshot_path: Path) -> Dict[str, str]:
        image_bytes = screenshot_path.read_bytes()
        image_b64 = base64.b64encode(image_bytes).decode("ascii")
        response = self.client.chat.completions.create(
            model="gpt-5.3-chat-latest",
            messages=[
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "Describe this screenshot with actionable UI details."},
                        {"type": "image_url", "image_url": {"url": f"data:image/png;base64,{image_b64}"}},
                    ],
                }
            ],
            max_completion_tokens=200,
        )
        description = response.choices[0].message.content or ""
        return {"description": description}
