import time
import openai
import logging
from typing import Union, List, Any, Optional
import numpy as np
import tiktoken
from .llm_output_parser_interface import LLMOutputParserInterface
from dataclasses import dataclass
from enum import Enum

# Set up logger for this module
logger = logging.getLogger(__name__)

class ProviderType(Enum):
    """Supported LLM providers"""
    OPENAI = "openai"
    MISTRAL = "mistral"
    CLAUDE = "claude"
    UNKNOWN = "unknown"

@dataclass
class ProviderConfig:
    """Configuration for different LLM providers"""
    name: str
    max_elements_per_batch: int
    max_tokens_per_batch: int
    max_context_window: int
    max_pending_requests: Optional[int]
    warning_threshold_ratio: float = 0.8  # Warn at 80% of context window
    sleep_between_batches: float = 0.0
    
    @property
    def warning_threshold(self) -> int:
        """Calculate warning threshold for single prompts"""
        return int(self.max_context_window * self.warning_threshold_ratio)

# Provider-specific configurations
PROVIDER_CONFIGS = {
    ProviderType.OPENAI: ProviderConfig(
        name="OpenAI",
        max_elements_per_batch=40,   # Conservative batch size
        max_tokens_per_batch=8000,   # Conservative token limit
        max_context_window=128000,   # context window
        max_pending_requests=None,   # OpenAI doesn't have explicit pending request limits
        sleep_between_batches=2.0,   # Short delay + let OpenAI's built-in rate limiting handle the rest
    ),
    ProviderType.MISTRAL: ProviderConfig(
        name="Mistral",
        max_elements_per_batch=1,    # Sequential processing (respects 6 RPS limit)
        max_tokens_per_batch=10000,   # Conservative per-request token limit
        max_context_window=128000,   # Mistral Large context window
        max_pending_requests=1000000,  # 1M pending requests per workspace
        sleep_between_batches=0.2,   # 200ms delay = 5 RPS (under 6 RPS limit)
    ),
    ProviderType.CLAUDE: ProviderConfig(
        name="Claude",
        max_elements_per_batch=50,    # 50 RPM limit (conservative: 1 request per 1.2s)
        max_tokens_per_batch=8000,   # 8K input tokens per minute limit
        max_context_window=200000,   # Claude Sonnet 4 standard context window
        max_pending_requests=1000,   # Conservative pending request limit
        sleep_between_batches=1.2,   # 1.2s delay = 50 RPM (respects rate limit)
    ),
    ProviderType.UNKNOWN: ProviderConfig(
        name="Unknown",
        max_elements_per_batch=5,    # Even more conservative (was 50)
        max_tokens_per_batch=4000,   # Even lower (was 16000)
        max_context_window=32000,
        max_pending_requests=10000,
        sleep_between_batches=10.0,  # Very long delays (was 2.0)
    )
}

class LangchainOutputParser(LLMOutputParserInterface):
    """
    A provider-agnostic parser class for extracting and embedding information using Langchain.
    Automatically detects the LLM provider (OpenAI, Mistral, Claude, etc.) and applies appropriate 
    rate limiting and batch processing constraints.
    """
    
    def __init__(self, llm_model, embeddings_model, sleep_time: int = 5, provider_type: Optional[ProviderType] = None) -> None:
        """
        Initialize the LangchainOutputParser with specified models and operational parameters.
        
        Args:
            llm_model: The language model to use
            embeddings_model: The embeddings model to use  
            sleep_time: Sleep time between retries (default: 5 seconds)
            provider_type: Explicitly specify provider type (auto-detected if None)
        """
        self.model = llm_model
        self.embeddings_model = embeddings_model
        self.sleep_time = sleep_time
        
        # Auto-detect provider or use explicitly provided type
        self.provider_type = provider_type if provider_type else self._detect_provider()
        self.config = PROVIDER_CONFIGS[self.provider_type]
        
        logger.info(f"🔍 Detected LLM Provider: {self.config.name}")
        logger.info(f"📊 Rate Limiting Config: {self.config.max_elements_per_batch} requests/batch, "
              f"{self.config.max_tokens_per_batch} tokens/batch")
        
        # Add specific info for Mistral
        if self.provider_type == ProviderType.MISTRAL:
            rps = 1 / self.config.sleep_between_batches if self.config.sleep_between_batches > 0 else float('inf')
            logger.info(f"🎯 Mistral Optimization: Sequential processing at {rps:.1f} RPS (limit: 6 RPS)")
            logger.info("⏱️  Token limits: 2M/minute, 10B/month per model")
        
        # Add specific info for Claude
        if self.provider_type == ProviderType.CLAUDE:
            rps = 1 / self.config.sleep_between_batches if self.config.sleep_between_batches > 0 else float('inf')
            logger.info(f"🎯 Claude Optimization: Sequential processing at {rps:.1f} RPS (limit: 50 RPM)")
            logger.info("⏱️  Token limits: 30K input/minute, 8K output/minute, 200K context window")

    def _detect_provider(self) -> ProviderType:
        """
        Auto-detect the LLM provider based on the model class/attributes.
        
        Returns:
            ProviderType: The detected provider type
        """
        model_class_name = self.model.__class__.__name__.lower()
        model_module = self.model.__class__.__module__.lower()
        
        # Check for OpenAI indicators
        if any(indicator in model_class_name for indicator in ['openai', 'chatgpt', 'gpt']):
            return ProviderType.OPENAI
        if 'openai' in model_module:
            return ProviderType.OPENAI
            
        # Check for Mistral indicators
        if any(indicator in model_class_name for indicator in ['mistral', 'chatmistral']):
            return ProviderType.MISTRAL
        if 'mistral' in model_module:
            return ProviderType.MISTRAL
            
        # Check for Claude indicators
        if any(indicator in model_class_name for indicator in ['claude', 'anthropic']):
            return ProviderType.CLAUDE
        if 'anthropic' in model_module or 'claude' in model_module:
            return ProviderType.CLAUDE
            
        # Check model attributes for additional clues
        if hasattr(self.model, 'model_name'):
            model_name = str(self.model.model_name).lower()
            if 'gpt' in model_name or 'openai' in model_name:
                return ProviderType.OPENAI
            if 'mistral' in model_name:
                return ProviderType.MISTRAL
            if 'claude' in model_name or 'anthropic' in model_name:
                return ProviderType.CLAUDE
                
        logger.warning(f"⚠️  Could not auto-detect provider from model: {model_class_name}")
        logger.warning(f"   Module: {model_module}")
        logger.warning("   Using conservative defaults for unknown provider")
        return ProviderType.UNKNOWN

    def count_tokens(self, text: str, encoding_name: str = "cl100k_base") -> int:
        """
        Count the number of tokens in a given text using tiktoken.
        
        Note: Using cl100k_base encoding as approximation for most models.
        For more accuracy, consider using provider-specific tokenizers when available.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)

    def split_prompts_into_batches(self, prompts: List[str], max_elements: Optional[int] = None, max_tokens: Optional[int] = None, encoding_name: str = "cl100k_base") -> List[List[str]]:
        """
        Split the list of prompts into sub-batches that meet provider-specific API constraints.
        
        Uses provider-specific configurations:
        - OpenAI: 500 requests/batch, 80K tokens/batch (original GPT-4 optimized)
        - Mistral: 50 requests/batch, 10K tokens/batch (conservative for rate limits)
        - Claude: 50 requests/batch, 30K tokens/batch (respects 50 RPM, 30K input tokens/minute)
        - Unknown: 5 requests/batch, 4K tokens/batch (very conservative)
        
        Args:
            prompts: List of prompt strings to batch
            max_elements: Override default max elements per batch
            max_tokens: Override default max tokens per batch
            encoding_name: Token encoding to use (default: "cl100k_base")
        """
        # Use provider-specific defaults or user overrides
        max_elements = max_elements or self.config.max_elements_per_batch
        max_tokens = max_tokens or self.config.max_tokens_per_batch
        
        # Validate against provider limits
        if self.config.max_pending_requests and len(prompts) > self.config.max_pending_requests:
            raise ValueError(
                f"Total prompts ({len(prompts)}) exceeds {self.config.name}'s "
                f"{self.config.max_pending_requests:,} request limit"
            )
            
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_token_sum: int = 0

        for prompt in prompts:
            token_count = self.count_tokens(prompt, encoding_name)
            
            # Check if single prompt exceeds context window
            if token_count > self.config.warning_threshold:
                logger.warning(f"⚠️  Warning: Single prompt has {token_count:,} tokens, "
                      f"approaching {self.config.name}'s {self.config.max_context_window:,} token context window")
            
            # If adding this prompt would exceed either limit, start a new batch
            if (len(current_batch) + 1 > max_elements) or (current_token_sum + token_count > max_tokens):
                if current_batch:  # Only append non-empty batches
                    batches.append(current_batch)
                current_batch = [prompt]
                current_token_sum = token_count
            else:
                current_batch.append(prompt)
                current_token_sum += token_count
                
        if current_batch:
            batches.append(current_batch)
            
        logger.info(f"📦 Split {len(prompts):,} prompts into {len(batches)} batches for {self.config.name} API")
        return batches

    async def calculate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate embeddings for the given text using the initialized embeddings model.
        """
        if isinstance(text, list):
            embeddings = await self.embeddings_model.aembed_documents(text)
        elif isinstance(text, str):
            embeddings = await self.embeddings_model.aembed_query(text)
        else:
            raise TypeError("Invalid text type, please provide a string or a list of strings.")
        return np.array(embeddings)
    
    def _render_chat_messages_to_string(self, messages: list) -> str:
        """Render a list of chat messages to a single string for token counting."""
        parts = []
        for msg in messages:
            if hasattr(msg, 'content'):
                parts.append(msg.content)
            elif isinstance(msg, tuple):
                parts.append(str(msg[1]))
            else:
                parts.append(str(msg))
        return "\n".join(parts)

    async def batch_structured_chat_calls(
        self,
        prompt_template,
        variables_list: List[dict],
        output_data_structure,
    ) -> List[Any]:
        """
        Batch-process a ChatPromptTemplate across multiple variable sets with
        provider-aware rate limiting, retries, and parallel execution within batches.

        Unlike extract_information_as_json_for_context (which builds flat string prompts),
        this preserves the chat message format (system + human) for better model performance.

        Args:
            prompt_template: A ChatPromptTemplate with placeholders matching variables_list keys.
            variables_list: One dict per call (e.g. one per cluster), each filling the template.
            output_data_structure: Pydantic model for structured output (e.g. AssessResult).

        Returns:
            List of parsed output_data_structure instances, one per variables_list entry, in order.
        """
        if not variables_list:
            return []

        if self.config.max_pending_requests and len(variables_list) > self.config.max_pending_requests:
            raise ValueError(
                f"Number of calls ({len(variables_list):,}) exceeds {self.config.name}'s "
                f"{self.config.max_pending_requests:,} request limit"
            )

        structured_llm = self.model.with_structured_output(output_data_structure)
        chain = prompt_template | structured_llm

        # Render each prompt for token counting, keep the variables for chain.abatch
        rendered_strings = []
        for variables in variables_list:
            messages = prompt_template.format_messages(**variables)
            rendered_strings.append(self._render_chat_messages_to_string(messages))

        # Use existing batching logic on the rendered strings to determine batch boundaries
        batches_of_strings = self.split_prompts_into_batches(rendered_strings)
        logger.info(f"Processing {len(variables_list):,} chat calls in {len(batches_of_strings)} batches for {self.config.name} API")

        # Map string batches back to variable batches (same sizes, same order)
        batches_of_variables: List[List[dict]] = []
        offset = 0
        for batch_strings in batches_of_strings:
            batch_size = len(batch_strings)
            batches_of_variables.append(variables_list[offset:offset + batch_size])
            offset += batch_size

        outputs: List[Any] = []
        for i, batch_vars in enumerate(batches_of_variables):
            logger.info(f"Processing batch {i+1}/{len(batches_of_variables)} with {len(batch_vars)} requests ({self.config.name})")

            max_retries = 3 if self.provider_type in [ProviderType.MISTRAL, ProviderType.CLAUDE] else 2
            base_sleep = self.sleep_time

            for attempt in range(max_retries + 1):
                try:
                    batch_outputs = await chain.abatch(batch_vars)
                    outputs.extend(batch_outputs)
                    break
                except openai.RateLimitError as e:
                    if attempt == max_retries:
                        logger.error(f"Rate limit retry failed for batch {i+1} after {max_retries} attempts: {e}")
                        raise
                    sleep_time = base_sleep * (2 ** attempt)
                    if self.provider_type == ProviderType.MISTRAL:
                        sleep_time *= 2
                    logger.warning(f"RateLimitError in batch {i+1}, attempt {attempt+1}/{max_retries+1}: {e}")
                    time.sleep(sleep_time)
                except openai.BadRequestError as e:
                    if attempt == max_retries:
                        logger.error(f"BadRequest retry failed for batch {i+1} after {max_retries} attempts: {e}")
                        raise
                    logger.error(f"BadRequestError in batch {i+1}, attempt {attempt+1}/{max_retries+1}: {e}")
                    time.sleep(base_sleep)
                except Exception as e:
                    if attempt == max_retries:
                        logger.error(f"Final retry failed for batch {i+1} after {max_retries} attempts: {e}")
                        raise
                    sleep_time = base_sleep * (2 ** attempt)
                    logger.warning(f"Error in batch {i+1}, attempt {attempt+1}/{max_retries+1}: {e}")
                    time.sleep(sleep_time)

            if i < len(batches_of_variables) - 1 and self.config.sleep_between_batches > 0:
                logger.info(f"Sleeping {self.config.sleep_between_batches}s between batches for {self.config.name}")
                time.sleep(self.config.sleep_between_batches)

        logger.info(f"Successfully processed all {len(outputs):,} chat calls for {self.config.name}")
        return outputs

    async def extract_information_as_json_for_context(self,
                                                      output_data_structure,
                                                      contexts: List[str],
                                                      system_query: str = '''
                                                    # DIRECTIVES :
                                                    - Act like an experienced information extractor.
                                                    - If you do not find the right information, keep its place empty.
                                                    ''',
                                                      return_exceptions: bool = False) -> List[Any]:
        """
        Prepares prompts for each context, calculates token counts, splits the prompts into batches
        that respect provider-specific API constraints, and processes them with appropriate error handling.
        
        Provider-Specific Optimizations:
        - OpenAI: Larger batches (500 req, 80K tokens), minimal delays
        - Mistral: Sequential processing (50 req/batch, 200ms delays) respects 6 RPS limit
        - Claude: Sequential processing (50 req/batch, 1.2s delays) respects 50 RPM limit
        - Unknown: Ultra conservative batches (5 req, 4K tokens), 10s delays
        
        Args:
            output_data_structure: The expected output structure
            contexts: List of context strings to process
            system_query: The system query/instruction
            return_exceptions: If True, per-item failures are returned as Exception objects
                in the output list (forwarded to ``abatch``) instead of raising. Default False
                preserves existing callers' behavior.
        """
        # Validate against provider limits
        if self.config.max_pending_requests and len(contexts) > self.config.max_pending_requests:
            raise ValueError(
                f"Number of contexts ({len(contexts):,}) exceeds {self.config.name}'s "
                f"{self.config.max_pending_requests:,} request limit"
            )
            
        structured_llm = self.model.with_structured_output(output_data_structure)
        
        # Create prompts for each context
        all_prompts = [
            f"# Context: {context}\n\n# Question: {system_query}\n\nAnswer: "
            for context in contexts
        ]
        
        # Split the prompts into batches according to provider-specific limits
        batches = self.split_prompts_into_batches(all_prompts)
        logger.info(f"🚀 Processing {len(contexts):,} contexts in {len(batches)} batches for {self.config.name} API")
        
        # Add time estimation for Mistral
        if self.provider_type == ProviderType.MISTRAL and len(batches) > 1:
            estimated_time = len(batches) * self.config.sleep_between_batches
            if estimated_time > 60:
                logger.info(f"⏰ Estimated processing time: {estimated_time/60:.1f} minutes (sequential processing)")
            else:
                logger.info(f"⏰ Estimated processing time: {estimated_time:.1f} seconds")

        outputs = []
        # Process each batch sequentially to respect rate limits
        for i, batch in enumerate(batches):
            logger.info(f"📋 Processing batch {i+1}/{len(batches)} with {len(batch)} requests ({self.config.name})")
            
            # For Mistral and Claude, use exponential backoff and extra careful processing
            max_retries = 3 if self.provider_type in [ProviderType.MISTRAL, ProviderType.CLAUDE] else 2
            base_sleep = self.sleep_time
            
            for attempt in range(max_retries + 1):
                try:
                    batch_outputs = await structured_llm.abatch(
                        batch, return_exceptions=return_exceptions
                    )
                    outputs.extend(batch_outputs)
                    break  # Success, exit retry loop
                    
                except openai.RateLimitError as e:
                    if attempt == max_retries:
                        logger.error(f"💥 Rate limit retry failed for batch {i+1} after {max_retries} attempts: {e}")
                        raise
                    
                    # Exponential backoff for rate limits
                    sleep_time = base_sleep * (2 ** attempt)
                    if self.provider_type == ProviderType.MISTRAL:
                        sleep_time *= 2  # Extra conservative for Mistral
                    
                    logger.warning(f"⏱️  RateLimitError in batch {i+1}, attempt {attempt+1}/{max_retries+1}: {e}")
                    logger.warning(f"   Sleeping for {sleep_time} seconds (exponential backoff)")
                    time.sleep(sleep_time)
                    
                except openai.BadRequestError as e:
                    if attempt == max_retries:
                        logger.error(f"💥 BadRequest retry failed for batch {i+1} after {max_retries} attempts: {e}")
                        raise
                    
                    sleep_time = base_sleep
                    logger.error(f"❌ BadRequestError in batch {i+1}, attempt {attempt+1}/{max_retries+1}: {e}")
                    logger.error(f"   Sleeping for {sleep_time} seconds before retry")
                    time.sleep(sleep_time)
                    
                except Exception as e:
                    error_message = str(e).lower()
                    
                    # Check for Mistral-specific rate limit errors
                    is_mistral_rate_limit = (
                        "rate limit" in error_message or 
                        "1300" in error_message or 
                        "429" in error_message or
                        "too many requests" in error_message
                    )
                    
                    # Check for Claude-specific rate limit errors
                    is_claude_rate_limit = (
                        "rate limit" in error_message or 
                        "429" in error_message or
                        "too many requests" in error_message or
                        "anthropic" in error_message.lower()
                    )
                    
                    if is_mistral_rate_limit and self.provider_type == ProviderType.MISTRAL:
                        if attempt == max_retries:
                            logger.error(f"💥 Mistral rate limit exceeded for batch {i+1} after {max_retries} attempts: {e}")
                            raise
                        
                        # Very aggressive backoff for Mistral rate limits
                        sleep_time = base_sleep * (3 ** attempt)  # 3^attempt instead of 2^attempt
                        if attempt >= 1:
                            sleep_time += 10  # Extra 10 seconds from second attempt onwards
                            
                        logger.warning(f"🚨 Mistral rate limit (code 1300) in batch {i+1}, attempt {attempt+1}/{max_retries+1}")
                        logger.warning(f"   Aggressive backoff: sleeping {sleep_time} seconds")
                        time.sleep(sleep_time)
                        continue
                    
                    if is_claude_rate_limit and self.provider_type == ProviderType.CLAUDE:
                        if attempt == max_retries:
                            logger.error(f"💥 Claude rate limit exceeded for batch {i+1} after {max_retries} attempts: {e}")
                            raise
                        
                        # Conservative backoff for Claude rate limits (50 RPM)
                        sleep_time = base_sleep * (2 ** attempt)  # Exponential backoff
                        if attempt >= 1:
                            sleep_time += 5  # Extra 5 seconds from second attempt onwards
                            
                        logger.warning(f"🚨 Claude rate limit in batch {i+1}, attempt {attempt+1}/{max_retries+1}")
                        logger.warning(f"   Conservative backoff: sleeping {sleep_time} seconds")
                        time.sleep(sleep_time)
                        continue
                    
                    # Handle other API errors
                    if attempt == max_retries:
                        logger.error(f"💥 Final retry failed for batch {i+1} after {max_retries} attempts: {e}")
                        raise
                    
                    sleep_time = base_sleep
                    logger.warning(f"⚠️  Unexpected error in batch {i+1}, attempt {attempt+1}/{max_retries+1}: {e}")
                    logger.warning(f"   Sleeping for {sleep_time} seconds before retry")
                    time.sleep(sleep_time)
                    
            # Add provider-specific delay between batches
            if i < len(batches) - 1 and self.config.sleep_between_batches > 0:
                logger.info(f"😴 Sleeping {self.config.sleep_between_batches}s between batches for {self.config.name}")
                time.sleep(self.config.sleep_between_batches)
                
        logger.info(f"✅ Successfully processed all {len(outputs):,} requests for {self.config.name}")
        return outputs