from typing import List, Union, Any
from pydantic import BaseModel
from itext2kg.llm_output_parsing.langchain_output_parser import LangchainOutputParser


class DocumentsDistiller:
    """
    A class designed to distill essential information from multiple documents into a combined
    structure, using natural language processing tools to extract and consolidate information.
    """
    def __init__(self, llm_model) -> None:
        """
        Initializes the DocumentsDistiller with specified language model
        
        Args:
        llm_model: The language model instance to be used for generating semantic blocks.
        """
        self.langchain_output_parser = LangchainOutputParser(llm_model=llm_model, embeddings_model=None)
    
    @staticmethod
    def __combine_objects(object_list: List[Union[dict, BaseModel]]) -> Union[dict, BaseModel]:
        """
        Combine a list of dictionaries or Pydantic objects, returning the same type as the input.
        
        Args:
        object_list (List[Union[dict, BaseModel]]): A list of dictionaries or Pydantic objects to combine.
        
        Returns:
        Union[dict, BaseModel]: A combined object with merged values. Returns the same type as the input objects.
        """
        if not object_list:
            return {}
        
        # Check if all objects are Pydantic models of the same type
        pydantic_objects = [obj for obj in object_list if isinstance(obj, BaseModel)]
        dict_objects = [obj for obj in object_list if isinstance(obj, dict)]
        
        if pydantic_objects and all(type(obj) is type(pydantic_objects[0]) for obj in pydantic_objects):
            # All Pydantic objects are of the same type - work directly with objects
            return DocumentsDistiller.__combine_pydantic_objects(pydantic_objects, dict_objects)
        else:
            # Mixed types or only dicts - fall back to dict-based approach
            return DocumentsDistiller.__combine_via_dicts(object_list)
    
    @staticmethod
    def __combine_pydantic_objects(pydantic_objects: List[BaseModel], dict_objects: List[dict] = None) -> BaseModel:
        """
        Combine Pydantic objects directly without converting to dictionaries.
        
        Args:
        pydantic_objects (List[BaseModel]): List of Pydantic objects of the same type.
        dict_objects (List[dict], optional): Additional dictionaries to merge.
        
        Returns:
        BaseModel: A new Pydantic object with merged values.
        """
        if not pydantic_objects:
            return {}
            
        model_type = type(pydantic_objects[0])
        combined_values = {}
        
        # Get all field names from the model
        field_names = set()
        for obj in pydantic_objects:
            field_names.update(obj.model_fields_set)
            field_names.update(obj.model_fields.keys())
        
        # Add fields from dictionaries if any
        if dict_objects:
            for d in dict_objects:
                field_names.update(d.keys())
        
        # Combine values for each field
        for field_name in field_names:
            values_to_combine = []
            
            # Collect values from Pydantic objects
            for obj in pydantic_objects:
                if hasattr(obj, field_name):
                    value = getattr(obj, field_name)
                    if value is not None:
                        values_to_combine.append(value)
            
            # Collect values from dictionaries
            if dict_objects:
                for d in dict_objects:
                    if field_name in d and d[field_name] is not None:
                        values_to_combine.append(d[field_name])
            
            if values_to_combine:
                combined_values[field_name] = DocumentsDistiller.__merge_field_values(values_to_combine)
        
        return model_type(**combined_values)
    
    @staticmethod
    def __merge_field_values(values: List[Any]) -> Any:
        """
        Merge multiple values for a single field based on their types.
        
        Args:
        values (List[Any]): List of values to merge.
        
        Returns:
        Any: Merged value.
        """
        if not values:
            return None
        
        if len(values) == 1:
            return values[0]
        
        first_value = values[0]
        
        # Handle lists - extend all lists together
        if isinstance(first_value, list):
            combined_list = []
            for value in values:
                if isinstance(value, list):
                    combined_list.extend(value)
                else:
                    combined_list.append(value)
            return combined_list
        
        # Handle strings - concatenate with spaces
        elif isinstance(first_value, str):
            return ' '.join(str(v) for v in values if v)
        
        # Handle dictionaries - merge all dicts
        elif isinstance(first_value, dict):
            combined_dict = {}
            for value in values:
                if isinstance(value, dict):
                    combined_dict.update(value)
            return combined_dict
        
        # For other types, return the last non-None value
        else:
            return next((v for v in reversed(values) if v is not None), first_value)
    
    @staticmethod
    def __combine_via_dicts(object_list: List[Union[dict, BaseModel]]) -> Union[dict, BaseModel]:
        """
        Fallback method that combines objects via dictionary conversion.
        
        Args:
        object_list (List[Union[dict, BaseModel]]): List of objects to combine.
        
        Returns:
        Union[dict, BaseModel]: Combined object.
        """
        combined_dict = {}
        original_type = None
        
        for obj in object_list:
            if isinstance(obj, BaseModel) and original_type is None:
                original_type = type(obj)
                d = obj.model_dump()
            elif isinstance(obj, BaseModel):
                d = obj.model_dump()
            else:
                d = obj
            
            for key, value in d.items():
                if key in combined_dict:
                    combined_dict[key] = DocumentsDistiller.__merge_field_values([combined_dict[key], value])
                else:
                    combined_dict[key] = value
        
        if original_type is not None:
            return original_type(**combined_dict)
        
        return combined_dict


    async def distill(self, documents: List[str], output_data_structure, IE_query:str) -> Union[dict, BaseModel]:
        """
        Distill information from multiple documents based on a specific information extraction query.
        
        Args:
        documents (List[str]): A list of documents from which to extract information.
        output_data_structure: The data structure definition for formatting the output JSON.
        IE_query (str): The query to provide to the language model for extracting information.
        
        Returns:
        Union[dict, BaseModel]: A combined object representing distilled information from all documents.
                               Returns the same type as the objects returned by the language model.
        """
        # Use the batch processing capability of the async method
        output_jsons = await self.langchain_output_parser.extract_information_as_json_for_context(
            contexts=documents, 
            system_query=IE_query, 
            output_data_structure=output_data_structure
        )
        
        return DocumentsDistiller.__combine_objects(output_jsons)


