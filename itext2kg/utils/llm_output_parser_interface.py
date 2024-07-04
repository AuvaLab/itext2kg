from abc import ABC, abstractmethod


class LLMOutputParser(ABC):
    @abstractmethod
    def extract_information_as_json_for_context(self, context:str, IE_query:str):
        pass