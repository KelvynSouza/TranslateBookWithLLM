"""
LLM Provider abstraction and implementations
"""
from abc import ABC, abstractmethod
from typing import Optional, Dict, Any
import re
import httpx
import json
import asyncio

from src.config import (
    API_ENDPOINT, DEFAULT_MODEL, REQUEST_TIMEOUT, OLLAMA_NUM_CTX,
    MAX_TRANSLATION_ATTEMPTS, RETRY_DELAY_SECONDS,
    TRANSLATE_TAG_IN, TRANSLATE_TAG_OUT, TEMPERATURE, TOP_K, TOP_P
)


class LLMProvider(ABC):
    """Abstract base class for LLM providers"""

    def __init__(self, model: str, config: Dict[str, Any] = {}):
        self.model = model
        self._compiled_regex = re.compile(
            rf"{re.escape(TRANSLATE_TAG_IN)}(.*?){re.escape(TRANSLATE_TAG_OUT)}",
            re.DOTALL
        )
        self._client = None
        self.config = config

    async def _get_client(self) -> httpx.AsyncClient:
        """Get or create a persistent HTTP client with connection pooling"""
        if self._client is None:
            self._client = httpx.AsyncClient(
                limits=httpx.Limits(max_keepalive_connections=5, max_connections=10),
                timeout=httpx.Timeout(REQUEST_TIMEOUT)
            )
        return self._client

    async def close(self):
        """Close the HTTP client"""
        if self._client:
            await self._client.aclose()
            self._client = None

    @abstractmethod
    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
        """Generate text from prompt"""
        pass

    def extract_translation(self, response: str) -> Optional[str]:
        """Extract translation from response using configured tags"""
        if not response:
            return None

        match = self._compiled_regex.search(response)
        if match:
            return match.group(1).strip()
        return None

    async def translate_text(self, prompt: str) -> Optional[str]:
        """Complete translation workflow: request + extraction"""
        response = await self.generate(prompt)
        if response:
            return self.extract_translation(response)
        return None


class OllamaProvider(LLMProvider):
    """Ollama API provider"""

    def __init__(self, api_endpoint: str = API_ENDPOINT, model: str = DEFAULT_MODEL, config: Dict[str, Any] = {}):
        super().__init__(model, config)
        self.api_endpoint = api_endpoint

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
        """Generate text using Ollama API"""
        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "think": False,
            "options": {"num_ctx": OLLAMA_NUM_CTX, "temperature": self.config['temperature'], "top_k": self.config['top_k'], "top_p": self.config['top_p']}
        }

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            try:
                # print(f"Ollama API Request to {self.api_endpoint} with model {self.model}")
                response = await client.post(
                    self.api_endpoint,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()

                response_json = response.json()
                response_text = response_json.get("response", "")
                # print(f"Ollama API Response received: {len(response_text)} characters")
                return response_text

            except httpx.TimeoutException as e:
                    print(f"Ollama API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None
            except httpx.HTTPStatusError as e:
                    print(f"Ollama API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None
            except json.JSONDecodeError as e:
                    print(f"Ollama API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None
            except Exception as e:
                    print(f"Ollama API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None

        return None


class GeminiProvider(LLMProvider):
    """Google Gemini API provider"""

    def __init__(self, api_key: str, model: str = "gemini-2.0-flash", config: Dict[str, Any] = {}):
        super().__init__(model, config)
        self.api_key = api_key
        self.api_endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    async def get_available_models(self) -> list[dict]:
        """Fetch available Gemini models from API, excluding thinking models"""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        models_endpoint = "https://generativelanguage.googleapis.com/v1beta/models"

        client = await self._get_client()
        try:
            response = await client.get(
                models_endpoint,
                headers=headers,
                timeout=10
            )
            response.raise_for_status()

            data = response.json()
            models = []

            for model in data.get("models", []):
                model_name = model.get("name", "").replace("models/", "")

                # Skip thinking, experimental, latest, and vision models
                model_name_lower = model_name.lower()
                skip_keywords = ["thinking", "experimental", "latest", "vision", "-exp-"]
                if any(keyword in model_name_lower for keyword in skip_keywords):
                    continue

                # Only include models that support generateContent
                supported_methods = model.get("supportedGenerationMethods", [])
                if "generateContent" in supported_methods:
                    models.append({
                        "name": model_name,
                        "displayName": model.get("displayName", model_name),
                        "description": model.get("description", ""),
                        "inputTokenLimit": model.get("inputTokenLimit", 0),
                        "outputTokenLimit": model.get("outputTokenLimit", 0)
                    })

            return models

        except Exception as e:
            print(f"Error fetching Gemini models: {e}")
            return []

    async def generate(self, prompt: str, timeout: int = REQUEST_TIMEOUT) -> Optional[str]:
        """Generate text using Gemini API"""
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self.api_key
        }

        payload = {
            "contents": [{
                "parts": [{
                    "text": prompt
                }]
            }],
            "generationConfig": {
                "temperature": self.config['temperature'],
                "maxOutputTokens": 2048
            }
        }

        # Debug logs removed - uncomment if needed for troubleshooting
        # print(f"[DEBUG] Gemini API URL: {self.api_endpoint}")
        # print(f"[DEBUG] Using API key: {self.api_key[:10]}...{self.api_key[-4:]}")

        client = await self._get_client()
        for attempt in range(MAX_TRANSLATION_ATTEMPTS):
            try:
                # print(f"Gemini API Request to {self.api_endpoint}")
                response = await client.post(
                    self.api_endpoint,
                    headers=headers,
                    json=payload,
                    timeout=timeout
                )
                response.raise_for_status()

                response_json = response.json()
                # Extract text from Gemini response structure
                response_text = ""
                if "candidates" in response_json and response_json["candidates"]:
                    content = response_json["candidates"][0].get("content", {})
                    parts = content.get("parts", [])
                    if parts:
                        response_text = parts[0].get("text", "")

                # print(f"Gemini API Response received: {len(response_text)} characters")
                return response_text

            except httpx.TimeoutException as e:
                    print(f"Gemini API Timeout (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None
            except httpx.HTTPStatusError as e:
                    print(f"Gemini API HTTP Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if hasattr(e, 'response') and hasattr(e.response, 'text'):
                        print(f"Response details: Status {e.response.status_code}, Body: {e.response.text[:200]}...")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None
            except json.JSONDecodeError as e:
                    print(f"Gemini API JSON Decode Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None
            except Exception as e:
                    print(f"Gemini API Unknown Error (attempt {attempt + 1}/{MAX_TRANSLATION_ATTEMPTS}): {e}")
                    if attempt < MAX_TRANSLATION_ATTEMPTS - 1:
                        await asyncio.sleep(RETRY_DELAY_SECONDS)
                        continue
                    return None

        return None


def create_llm_provider(provider_type: str = "ollama", **kwargs) -> LLMProvider:
    """Factory function to create LLM providers"""
    # Auto-detect provider from model name if not explicitly set
    model = kwargs.get("model", DEFAULT_MODEL)
    if provider_type == "ollama" and model and model.startswith("gemini"):
        # Auto-switch to Gemini provider when Gemini model is detected
        # print(f"[INFO] Auto-switching to Gemini provider for model '{model}'")
        provider_type = "gemini"

    if provider_type.lower() == "ollama":
        return OllamaProvider(
            api_endpoint=kwargs.get("api_endpoint", API_ENDPOINT),
            model=kwargs.get("model", DEFAULT_MODEL),
            config=kwargs["config"]
        )
    elif provider_type.lower() == "gemini":
        api_key = kwargs.get("api_key")
        if not api_key:
            # Try to get from environment
            import os
            api_key = os.getenv("GEMINI_API_KEY")
            if not api_key:
                raise ValueError("Gemini provider requires an API key. Set GEMINI_API_KEY environment variable or pass api_key parameter.")
        return GeminiProvider(
            api_key=api_key,
            model=kwargs.get("model", "gemini-2.0-flash"),
            config=kwargs["config"]
        )
    else:
        raise ValueError(f"Unknown provider type: {provider_type}")
