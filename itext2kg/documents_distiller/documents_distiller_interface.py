from abc import ABC, abstractmethod
from typing import List

class DocumentsDistillerInterface(ABC):
    
    @abstractmethod
    def distill(self, documents:List[str]) -> dict:
        pass