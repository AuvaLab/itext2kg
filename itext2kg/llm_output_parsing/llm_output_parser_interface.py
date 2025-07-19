from typing import Protocol, Union, List, Any
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
                                 max_elements: int = 500, 
                                 max_tokens: int = 80000, 
                                 encoding_name: str = "cl100k_base") -> List[List[str]]:
        """
        Split a list of prompts into batches that respect token and element limits.
        
        Args:
            prompts: List of prompts to split
            max_elements: Maximum number of elements per batch
            max_tokens: Maximum number of tokens per batch
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
        Extract structured information from contexts using the specified output structure.
        
        Args:
            output_data_structure: The structure to use for the output
            contexts: List of contexts to extract information from
            system_query: The system query to use for extraction
            
        Returns:
            List of extracted information matching the output structure
        """
        ... 