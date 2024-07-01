from langchain_openai import ChatOpenAI
from langchain_core.exceptions import OutputParserException
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import JsonOutputParser
import time
import openai
from .llm_output_parser_interface import LLMOutputParser



class LangchainOutputParser(LLMOutputParser):
    def __init__(self, model: ChatOpenAI, sleep_time=5) -> None:
        self.model = model
        self.sleep_time = sleep_time

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