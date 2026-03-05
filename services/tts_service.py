from pathlib import Path
from typing import Optional

import requests


class TextToSpeechService:
    def __init__(self, api_key: str, voice_id: str = "q9WrmbeoNoOQB0kmb5gD") -> None:
        self.api_key = api_key
        self.voice_id = voice_id

    def synthesize(self, text: str, output_path: Path) -> Optional[Path]:
        if not self.api_key:
            return None
        response = requests.post(
            f"https://api.elevenlabs.io/v1/text-to-speech/{self.voice_id}",
            headers={
                "Accept": "audio/mpeg",
                "Content-Type": "application/json",
                "xi-api-key": self.api_key,
            },
            json={
                "text": text,
                "model_id": "eleven_monolingual_v1",
                "voice_settings": {"stability": 0.5, "similarity_boost": 0.75},
            },
            timeout=30,
        )
        response.raise_for_status()
        output_path.write_bytes(response.content)
        return output_path
