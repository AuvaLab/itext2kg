from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import time
import openai
from openai import OpenAI
from .llm_output_parser_interface import LLMOutputParser
from typing import Union, List
from langchain_openai import OpenAIEmbeddings
from langchain_openai import ChatOpenAI
import numpy as np

class LangchainOutputParser(LLMOutputParser):
    def __init__(self, openai_api_key:str, embeddings_model_name :str = "text-embedding-3-large", model_name:str = "gpt-4-turbo", temperature:float = 0, sleep_time:int=5) -> None:
        self.model = ChatOpenAI(api_key=openai_api_key, model_name=model_name, temperature=temperature)
        self.sleep_time = sleep_time
        self.embeddings_model = OpenAIEmbeddings(model = embeddings_model_name, api_key=openai_api_key)
        
        
    def calculate_embeddings(self, text:Union[str, List[str]]):
        if isinstance(text, list):
            return np.array(self.embeddings_model.embed_documents(text))
        elif isinstance(text, str):
            return np.array(self.embeddings_model.embed_query(text))
        else:
            raise TypeError("Invalid text type, please provide a string or a list of strings.")

            
    def extract_information_as_json_for_context(
        self,
        output_data_structure,
        context: str,
        IE_query: str = '''
        # DIRECTIVES : 
        - Act like an experienced information extractor. 
        - You have a chunk of a company website.
        - If you do not find the right information, keep its place empty.
        '''    
        ):
        # The role of IE_query is to prompt a language model to populate the data structure.

        # Set up a parser + inject instructions into the prompt template.
        parser = JsonOutputParser(pydantic_object=output_data_structure)

        template = f"""
        Context: {context}

        Question: {{query}}
        Format_instructions : {{format_instructions}}
        Answer: """

        prompt = PromptTemplate(
            template=template,
            input_variables=["query"],
            partial_variables={"format_instructions": parser.get_format_instructions()},
        )

        chain = prompt | self.model | parser
        try:
            return chain.invoke({"query": IE_query})
        except openai.BadRequestError as e:
            print(
                f"Too much requests, we are sleeping! \n the error is {e}"
            )
            time.sleep(self.sleep_time)
            self.extract_information_as_json_for_context(context=context)

        except openai.RateLimitError:
            print(
                "Too much requests exceeding rate limit, we are sleeping!"
            )
            time.sleep(self.sleep_time)
            self.extract_information_as_json_for_context(context=context)
            
        except OutputParserException:
            print(f"Error in parsing the instance {context}")
            pass