from typing import Literal, List

class DataHandler:
    def __init__(self):
        pass

    def process(self, data:dict, data_type:Literal['entity', 'relation']):
        data = data.copy()
        if data_type == 'relation':
            
            data["startNode"] = data["startNode"].lower()
            data["endNode"] = data["endNode"].lower()
            
            data["name"] = data["name"].replace(" ", "_").replace("-", "_").replace(".", "_").replace("&", "and")
        
        elif data_type == 'entity':
            data["label"] = data["label"].replace(" ", "_").replace("-", "_").replace(".", "_").replace("&", "and")
        
        return data

    def remove_duplicates(self, records:dict, data_type:Literal['entity', 'relation']):
        seen = set()
        unique_records = []

        for record in records:
            if data_type == 'relation':
                identifier = (record['startNode'], record['endNode'], record['name'])
            elif data_type == 'entity':
                identifier = record['name']

            if identifier not in seen:
                seen.add(identifier)
                unique_records.append(record)

        return unique_records

    def handle_data(self, data:dict, data_type:Literal['entity', 'relation']):
        processed_data = [self.process(item, data_type=data_type) for item in data]
        unique_data = self.remove_duplicates(records=processed_data, data_type=data_type)
        return unique_data
    
    def find_relations_with_isolated_entities(self, global_entities:List[dict], relations:List[dict]):
        isolated_entities = relations.copy()
        for rel in relations:
            if rel["startNode"] in global_entities and rel["endNode"] in global_entities:
                isolated_entities.remove(rel)
        return isolated_entities
    
    