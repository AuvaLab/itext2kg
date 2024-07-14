from typing import List
from ..utils import LangchainOutputParser


class DocumentsDisiller:
    def __init__(self, openai_api_key:str, model_name:str = "gpt-4-0125-preview", temperature:str=0) -> None:
        self.temperature = temperature
        self.langchain_output_parser = LangchainOutputParser(openai_api_key=openai_api_key, model_name=model_name, temperature=temperature)
    
    @staticmethod
    def __combine_dicts(dict_list:List[dict]):
        combined_dict = {}
        
        for d in dict_list:
            for key, value in d.items():
                if key in combined_dict:
                    if isinstance(value, list) and isinstance(combined_dict[key], list):
                        combined_dict[key].extend(value)
                    elif isinstance(value, str) and isinstance(combined_dict[key], str):
                        if value and combined_dict[key]:
                            combined_dict[key] += f' {value}'
                        elif value:
                            combined_dict[key] = value
                    elif isinstance(value, dict) and isinstance(combined_dict[key], dict):
                        combined_dict[key].update(value)
                    else:
                        combined_dict[key] = value
                else:
                    combined_dict[key] = value
        
        return combined_dict


    def distill(self, documents: List[str], output_data_structure, IE_query:str) -> dict:
        output_jsons = list(
            map(
                lambda context: self.langchain_output_parser.extract_information_as_json_for_context(
                    context = context, 
                    IE_query=IE_query, 
                    output_data_structure= output_data_structure
                    ), 
                documents))
        
        return DocumentsDisiller.__combine_dicts(output_jsons)


