import os
import time
from typing import Optional, Dict, Any
from dotenv import load_dotenv

class LLMInference:
    """
    Unified interface for local and cloud LLM inference.
    Supports: Ollama, HuggingFace Transformers, Anthropic Claude, OpenAI, Gemini
    """
    
    def __init__(self, provider: str = "ollama", 
                 model: str = None,
                 config: dict = None):
        """
        Initialize LLM client.
        
        Args:
            provider (str): "ollama", "huggingface", "anthropic", "openai"
            model (str): Model identifier
            config (dict): Provider-specific configuration
        """
        self.provider = provider.lower()
        self.config = config or {}
        self.model = model or self._get_default_model()
        self.client = None
        self.tokenizer = None # For HuggingFace
        self._initialize_client()
    
    def _get_default_model(self) -> str:
        """Return default model for each provider."""
        defaults = {
            "ollama": "llama2",
            "huggingface": "mistralai/Mistral-7B-Instruct-v0.2",
            "anthropic": "claude-3-sonnet-20240229",
            "openai": "gpt-4-turbo",
            "gemini": "gemini-2.5-flash"
        }
        return defaults.get(self.provider, "llama2")
    
    def _initialize_client(self):
        """Initialize the appropriate client based on provider."""
        if self.provider == "ollama":
            self._init_ollama()
        elif self.provider == "huggingface":
            self._init_huggingface()
        elif self.provider == "anthropic":
            self._init_anthropic()
        elif self.provider == "openai":
            self._init_openai()
        elif self.provider == "gemini":
            self._init_gemini()
        else:
            raise ValueError(f"Unsupported provider: {self.provider}")
    
    def _init_ollama(self):
        """Initialize Ollama client."""
        try:
            import ollama
            self.client = ollama.Client(
                host=self.config.get('base_url', 'http://localhost:11434')
            )
            # We don't test connection here to avoid blocking during init
        except ImportError:
            raise RuntimeError("Install with: pip install ollama")
    
    def _init_huggingface(self):
        """Initialize HuggingFace Transformers."""
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer
            import torch
            
            device = self.config.get('device', 'cpu')
            load_in_8bit = self.config.get('load_in_8bit', False)
            
            self.tokenizer = AutoTokenizer.from_pretrained(self.model)
            
            if load_in_8bit:
                self.client = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    load_in_8bit=True,
                    device_map="auto"
                )
            else:
                dtype = torch.float16 if device in ["cuda", "mps"] else torch.float32
                self.client = AutoModelForCausalLM.from_pretrained(
                    self.model,
                    torch_dtype=dtype
                ).to(device)
            
        except ImportError:
            raise RuntimeError("Install with: pip install transformers torch accelerate")
        except Exception as e:
            raise RuntimeError(f"Failed to load HuggingFace model: {e}")
    
    def _init_anthropic(self):
        """Initialize Anthropic Claude API."""
        try:
            from anthropic import Anthropic
            api_key = self.config.get('api_key') or os.getenv('ANTHROPIC_API_KEY')
            if not api_key:
                # We don't raise here, allow deferred key requirement
                pass
            self.client = Anthropic(api_key=api_key)
        except ImportError:
            raise RuntimeError("Install with: pip install anthropic")
    
    def _init_openai(self):
        """Initialize OpenAI API."""
        try:
            from openai import OpenAI
            api_key = self.config.get('api_key') or os.getenv('OPENAI_API_KEY')
            if not api_key:
                pass
            self.client = OpenAI(api_key=api_key)
        except ImportError:
            raise RuntimeError("Install with: pip install openai")
            
    def _init_gemini(self):
        """Initialize Google Gemini API."""
        try:
            import google.generativeai as genai
            
            
            # Load environment variables from .env file
            load_dotenv()
           
            
            api_key = os.getenv('GEMINI_API_KEY') or os.getenv('GOOGLE_API_KEY')
            if not api_key:
                raise ValueError(
                    "Gemini API key not found. Please set 'api_key' in config.yaml "
                    "or set the GEMINI_API_KEY or GOOGLE_API_KEY environment variable in your .env file."
                )
            genai.configure(api_key=api_key)
            self.client = genai.GenerativeModel(self.model)
        except ImportError:
            raise RuntimeError("Install with: pip install google-generativeai python-dotenv")
    
    def generate(self, prompt: str, max_tokens: int = 1000, 
                 temperature: float = 0.7) -> Dict[str, Any]:
        """
        Generate response from LLM.
        """
        start_time = time.time()
        
        if self.provider == "ollama":
            result = self._generate_ollama(prompt, max_tokens, temperature)
        elif self.provider == "huggingface":
            result = self._generate_huggingface(prompt, max_tokens, temperature)
        elif self.provider == "anthropic":
            result = self._generate_anthropic(prompt, max_tokens, temperature)
        elif self.provider == "openai":
            result = self._generate_openai(prompt, max_tokens, temperature)
        elif self.provider == "gemini":
            result = self._generate_gemini(prompt, max_tokens, temperature)
        else:
            raise ValueError(f"Unknown provider: {self.provider}")
        
        latency = (time.time() - start_time) * 1000
        result['latency_ms'] = latency
        result['provider'] = self.provider
        
        return result
    
    def _generate_ollama(self, prompt: str, max_tokens: int, 
                         temperature: float) -> dict:
        """Generate using Ollama."""
        response = self.client.generate(
            model=self.model,
            prompt=prompt,
            options={
                'temperature': temperature,
                'num_predict': max_tokens
            }
        )
        
        return {
            'response': response['response'],
            'model': self.model,
            'tokens_used': {
                'prompt': response.get('prompt_eval_count', 0),
                'completion': response.get('eval_count', 0),
                'total': response.get('prompt_eval_count', 0) + response.get('eval_count', 0)
            }
        }
    
    def _generate_huggingface(self, prompt: str, max_tokens: int, 
                             temperature: float) -> dict:
        """Generate using HuggingFace Transformers."""
        import torch
        
        inputs = self.tokenizer(prompt, return_tensors="pt").to(self.client.device)
        
        with torch.no_grad():
            outputs = self.client.generate(
                **inputs,
                max_new_tokens=max_tokens,
                temperature=temperature,
                do_sample=temperature > 0,
                pad_token_id=self.tokenizer.eos_token_id
            )
        
        response_text = self.tokenizer.decode(
            outputs[0][inputs['input_ids'].shape[1]:],
            skip_special_tokens=True
        )
        
        return {
            'response': response_text,
            'model': self.model,
            'tokens_used': {
                'prompt': inputs['input_ids'].shape[1],
                'completion': outputs.shape[1] - inputs['input_ids'].shape[1],
                'total': outputs.shape[1]
            }
        }
    
    def _generate_anthropic(self, prompt: str, max_tokens: int, 
                           temperature: float) -> dict:
        """Generate using Anthropic Claude API."""
        response = self.client.messages.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            'response': response.content[0].text,
            'model': self.model,
            'tokens_used': {
                'prompt': response.usage.input_tokens,
                'completion': response.usage.output_tokens,
                'total': response.usage.input_tokens + response.usage.output_tokens
            }
        }
    
    def _generate_openai(self, prompt: str, max_tokens: int, 
                        temperature: float) -> dict:
        """Generate using OpenAI API."""
        response = self.client.chat.completions.create(
            model=self.model,
            max_tokens=max_tokens,
            temperature=temperature,
            messages=[{"role": "user", "content": prompt}]
        )
        
        return {
            'response': response.choices[0].message.content,
            'model': self.model,
            'tokens_used': {
                'prompt': response.usage.prompt_tokens,
                'completion': response.usage.completion_tokens,
                'total': response.usage.total_tokens
            }
        }
        
    def _generate_gemini(self, prompt: str, max_tokens: int, 
                         temperature: float) -> dict:
        """Generate using Google Gemini API."""
        import google.generativeai as genai
        
        generation_config = genai.types.GenerationConfig(
            max_output_tokens=max_tokens,
            temperature=temperature
        )
        
        try:
            response = self.client.generate_content(
                prompt,
                generation_config=generation_config
            )
            
            # Check if response has text (might be blocked by safety filters)
            try:
                text = response.text
            except ValueError:
                # If the response doesn't contain text, check if it's because it was blocked
                if response.candidates:
                    text = "[Blocked by safety filter or other candidate issue]"
                else:
                    text = "[No response generated]"
            
            # Extract usage info if available
            usage = getattr(response, 'usage_metadata', None)
            tokens = {
                'prompt': usage.prompt_token_count if usage else 0,
                'completion': usage.candidates_token_count if usage else 0,
                'total': usage.total_token_count if usage else 0
            }
            
            return {
                'response': text,
                'model': self.model,
                'tokens_used': tokens
            }
        except Exception as e:
            raise RuntimeError(f"Gemini generation failed: {e}")
    
    def generate_with_retry(self, prompt: str, max_retries: int = 3,
                           backoff: float = 1.0, **kwargs) -> dict:
        """
        Generate with exponential backoff retry logic.
        """
        last_error = None
        
        for attempt in range(max_retries):
            try:
                return self.generate(prompt, **kwargs)
            except Exception as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = backoff * (2 ** attempt)
                    time.sleep(wait_time)
        
        raise RuntimeError(f"Failed after {max_retries} attempts: {last_error}")

def get_available_llm(prefer_local: bool = True) -> LLMInference:
    """
    Automatically detect and return the best available LLM.
    """
    if prefer_local:
        # Try Ollama
        try:
            import ollama
            client = ollama.Client()
            client.list()
            return LLMInference(provider="ollama")
        except:
            pass
            
    # Check for API keys
    if os.getenv('ANTHROPIC_API_KEY'):
        return LLMInference(provider="anthropic")
    if os.getenv('OPENAI_API_KEY'):
        return LLMInference(provider="openai")
        
    # Default to Ollama (it will fail later if not installed, but it's our first choice)
    return LLMInference(provider="ollama")
