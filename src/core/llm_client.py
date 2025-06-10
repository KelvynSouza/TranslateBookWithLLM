"""
Centralized LLM client for all API communication
"""
from typing import Optional, Dict, Any

from src.config import API_ENDPOINT, DEFAULT_MODEL
from src.core.llm_providers import create_llm_provider, LLMProvider


class LLMClient:
    """Centralized client for LLM API communication"""
    
    def __init__(self, provider_type: str = "ollama", **kwargs):
        self.provider_type = provider_type
        self.provider_kwargs = kwargs
        self._provider: Optional[LLMProvider] = None
        
        # For backward compatibility
        if "api_endpoint" in kwargs and "model" in kwargs:
            self.api_endpoint = kwargs["api_endpoint"]
            self.model = kwargs["model"]
        else:
            self.api_endpoint = API_ENDPOINT
            self.model = DEFAULT_MODEL
    
    def _get_provider(self) -> LLMProvider:
        """Get or create the LLM provider"""
        if not self._provider:
            self._provider = create_llm_provider(self.provider_type, **self.provider_kwargs)
        return self._provider
    
    async def make_request(self, prompt: str, model: Optional[str] = None, 
                    timeout: int = None) -> Optional[str]:
        """
        Make a request to the LLM API with error handling and retries
        
        Args:
            prompt: The prompt to send
            model: Model to use (defaults to instance model)
            timeout: Request timeout in seconds
            
        Returns:
            Raw response text or None if failed
        """
        provider = self._get_provider()
        
        # Update model if specified
        if model:
            provider.model = model
            
        if timeout:
            return await provider.generate(prompt, timeout)
        else:
            return await provider.generate(prompt)
    
    def extract_translation(self, response: str) -> Optional[str]:
        """
        Extract translation from response using configured tags
        
        Args:
            response: Raw LLM response
            
        Returns:
            Extracted translation or None if not found
        """
        provider = self._get_provider()
        return provider.extract_translation(response)
    
    async def translate_text(self, prompt: str, model: Optional[str] = None) -> Optional[str]:
        """
        Complete translation workflow: request + extraction
        
        Args:
            prompt: Translation prompt
            model: Model to use
            
        Returns:
            Extracted translation or None if failed
        """
        provider = self._get_provider()
        
        # Update model if specified
        if model:
            provider.model = model
            
        return await provider.translate_text(prompt)


# Global instance for backward compatibility
default_client = LLMClient(provider_type="ollama", api_endpoint=API_ENDPOINT, model=DEFAULT_MODEL)