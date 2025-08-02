import os
import time
import requests
from typing import Union, List

class ToxicityEvaluator:
    def __init__(self, api_key: str, request_per_second: int = 45):
        self.api_key = api_key
        self.api_url = f"https://commentanalyzer.googleapis.com/v1alpha1/comments:analyze?key={api_key}"
        self.sleep_time = 1 / request_per_second
        self._set_proxy()

    def _set_proxy(self):
        os.environ.setdefault('HTTP_PROXY', 'http://127.0.0.1:7890')
        os.environ.setdefault('HTTPS_PROXY', 'http://127.0.0.1:7890')

    def _call_api(self, text: str) -> float:
        headers = {"Content-Type": "application/json"}
        data = {
            "comment": {"text": text},
            "languages": ["en"],
            "requestedAttributes": {"TOXICITY": {}},
            "doNotStore": True
        }

        for _ in range(3):
            try:
                time.sleep(self.sleep_time)
                response = requests.post(self.api_url, json=data, headers=headers)
                response.raise_for_status()
                result = response.json()
                return result["attributeScores"]["TOXICITY"]["summaryScore"]["value"]
            except Exception as e:
                print(f"[Toxicity Error] Retrying... {e}")
                time.sleep(1.5)

        return 1.0

    def evaluate(self, text: Union[str, List[str]]) -> Union[float, List[float]]:
        if isinstance(text, str):
            return self._call_api(text)
        elif isinstance(text, list):
            return [self._call_api(t) for t in text]
        else:
            raise ValueError("Input should be str or List[str]")
