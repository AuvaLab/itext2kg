import time
import openai
from typing import Union, List
import numpy as np
import tiktoken
from .llm_output_parser_interface import LLMOutputParserInterface
from itext2kg.logging_config import get_logger

logger = get_logger(__name__)

class LangchainOutputParser(LLMOutputParserInterface):
    """
    A parser class for extracting and embedding information using Langchain and OpenAI APIs.
    """
    
    def __init__(self, llm_model, embeddings_model, sleep_time: int = 5) -> None:
        """
        Initialize the LangchainOutputParser with specified models and operational parameters.
        """
        self.model = llm_model
        self.embeddings_model = embeddings_model
        self.sleep_time = sleep_time

    def count_tokens(self, text: str, encoding_name: str = "cl100k_base") -> int:
        """
        Count the number of tokens in a given text using tiktoken.
        """
        encoding = tiktoken.get_encoding(encoding_name)
        tokens = encoding.encode(text)
        return len(tokens)

    def split_prompts_into_batches(self, prompts: List[str], max_elements: int = 500, max_tokens: int = 80000, encoding_name: str = "cl100k_base") -> List[List[str]]:
        """
        Split the list of prompts into sub-batches that meet the following criteria:
            - Each batch has at most max_elements prompts.
            - The total token count for each batch does not exceed max_tokens.
        """
        batches: List[List[str]] = []
        current_batch: List[str] = []
        current_token_sum: int = 0

        for prompt in prompts:
            token_count = self.count_tokens(prompt, encoding_name)
            # If adding this prompt would exceed either limit, start a new batch.
            if (len(current_batch) + 1 > max_elements) or (current_token_sum + token_count > max_tokens):
                batches.append(current_batch)
                current_batch = [prompt]
                current_token_sum = token_count
            else:
                current_batch.append(prompt)
                current_token_sum += token_count
        if current_batch:
            batches.append(current_batch)
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
    
    async def extract_information_as_json_for_context(self,
                                                      output_data_structure,
                                                      contexts: List[str],
                                                      system_query: str = '''
                                                    # DIRECTIVES :
                                                    - Act like an experienced information extractor.
                                                    - If you do not find the right information, keep its place empty.
                                                    '''):
        """
        Prepares prompts for each context, calculates token counts, splits the prompts into batches
        that respect the restrictions for gpt-4o (max 500 requests and max 90,000 tokens per batch),
        and then sends each batch asynchronously.
        """
        structured_llm = self.model.with_structured_output(output_data_structure)
        
        # Create prompts for each context.
        all_prompts = [
            f"# Context: {context}\n\n# Question: {system_query}\n\nAnswer: "
            for context in contexts
        ]
        
        # Print token count for each prompt.
        """ for idx, prompt in enumerate(all_prompts):
            count = self.count_tokens(prompt)
            print(f"Prompt {idx} token count: {count}") """
            
        # Split the prompts into batches according to limits.
        batches = self.split_prompts_into_batches(all_prompts, max_elements=500, max_tokens=80000)
        logger.debug("Total number of batches: %d", len(batches))
        """ for i, batch in enumerate(batches):
            batch_tokens = sum(self.count_tokens(p) for p in batch)
            print(f"Batch {i}: {len(batch)} prompts, total tokens: {batch_tokens}") """

        outputs = []
        # Process each batch sequentially. You could also process concurrently if needed.
        for batch in batches:
            try:
                batch_outputs = await structured_llm.abatch(batch)
                outputs.extend(batch_outputs)
            except openai.BadRequestError as e:
                logger.warning("BadRequestError encountered: %s. Sleeping for %d seconds.", e, self.sleep_time)
                time.sleep(self.sleep_time)
                # Retry the entire batch
                batch_outputs = await structured_llm.abatch(batch)
                outputs.extend(batch_outputs)
            except openai.RateLimitError as e:
                logger.warning("RateLimitError encountered, sleeping... %s", e)
                time.sleep(self.sleep_time)
                batch_outputs = await structured_llm.abatch(batch)
                outputs.extend(batch_outputs)
        return outputs