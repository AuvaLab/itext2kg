from typing import Protocol, Union, List, Any, Optional
import numpy as np

class LLMOutputParserInterface(Protocol):
    """
    Interface defining the contract for LLM output parsers.
    This interface ensures consistent behavior across different parser implementations.
    """
    
    def count_tokens(self, text: str, encoding_name: str = "cl100k_base") -> int:
        """
        Count the number of tokens in a given text.
        
        Args:
            text: The text to count tokens for
            encoding_name: The encoding to use for token counting
            
        Returns:
            The number of tokens in the text
        """
        ...
    
    def split_prompts_into_batches(self, 
                                 prompts: List[str], 
                                 max_elements: Optional[int] = None, 
                                 max_tokens: Optional[int] = None, 
                                 encoding_name: str = "cl100k_base") -> List[List[str]]:
        """
        Split a list of prompts into batches that respect token and element limits.
        
        Args:
            prompts: List of prompts to split
            max_elements: Maximum number of elements per batch (None = use provider defaults)
            max_tokens: Maximum number of tokens per batch (None = use provider defaults)
            encoding_name: The encoding to use for token counting
            
        Returns:
            List of prompt batches
        """
        ...
    
    async def calculate_embeddings(self, text: Union[str, List[str]]) -> np.ndarray:
        """
        Calculate embeddings for the given text.
        
        Args:
            text: Text or list of texts to calculate embeddings for
            
        Returns:
            Numpy array containing the embeddings
        """
        ...
    
    async def extract_information_as_json_for_context(self,
                                                    output_data_structure: Any,
                                                    contexts: List[str],
                                                    system_query: str = '''
                                                    # DIRECTIVES :
                                                    - Act like an experienced information extractor.
                                                    - If you do not find the right information, keep its place empty.
                                                    ''') -> List[Any]:
        """
        Extract structured information from contexts using the LLM.
        
        Args:
            output_data_structure: The expected output structure/schema
            contexts: List of context strings to process
            system_query: The system instruction/query
            
        Returns:
            List of extracted structured data matching the output structure
        """
        ... 