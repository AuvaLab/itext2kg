import numpy as np
import logging
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Tuple, Union
from ..models import Entity, Relationship


class Matcher:
    """
    Class to handle the matching and processing of entities or relations based on cosine similarity or name matching.
    """
    def __init__(self):
        pass
    
    def find_match(self, obj1: Union[Entity, Relationship], list_objects: List[Union[Entity, Relationship]], threshold: float = 0.8) -> Union[Entity, Relationship]:
        """
        Find a matching Entity or Relationship object based on name or high cosine similarity.
        :param obj1: The Entity or Relationship to find matches for.
        :param list_objects: List of Entities or Relationships to match against.
        :param match_type: Type of match to perform, either 'entity' or 'relation'.
        :param threshold: Cosine similarity threshold.
        :return: The best match or the original object if no match is found.
        """
        name1 = obj1.name 
        unique_ID1 = obj1.properties_info['unique_id'] if 'unique_id' in obj1.properties_info.keys() else None
        label1 = obj1.label if isinstance(obj1, Entity) else None
        emb1 = np.array(obj1.properties.embeddings).reshape(1, -1)
        best_match = None
        best_cosine_sim = threshold

        duplicates = []
        for obj2 in list_objects:
            name2 = obj2.name
            unique_ID2 = obj2.properties_info['unique_id'] if 'unique_id' in obj2.properties_info.keys() else None
            label2 = obj2.label if isinstance(obj2, Entity) else None
            emb2 = np.array(obj2.properties.embeddings).reshape(1, -1)
            
            if unique_ID2 and unique_ID1:
                if unique_ID1 == unique_ID2:
                    logging.info(f"[INFO] Wohoo! Unique ID matched --- [{obj1.name}:{obj1.label}] --merged--> [{obj2.name}:{obj2.label}]")
                    return obj1
            
            if name1 == name2 and label1 == label2:
                return obj1
            
            cosine_sim = cosine_similarity(emb1, emb2)[0][0]
            if isinstance(obj1, Relationship):
                if obj1.name != obj2.name and obj1.startEntity == obj2.startEntity and obj1.endEntity == obj2.endEntity:
                    cosine_sim = 1.0
            
            if cosine_sim >= threshold:
                best_cosine_sim = cosine_sim
                best_match = obj2
    
        if best_match:
            if isinstance(obj1, Relationship):
                logging.info(f"[INFO] Wohoo! consine sim {cosine_sim} >= {threshold} Relation was matched --- [{obj1.name}] --merged --> [{best_match.name}] ")
                obj1.name = best_match.name
                obj1.properties.embeddings = best_match.properties.embeddings
                
            elif isinstance(obj1, Entity):
                logging.info(f"[INFO] Wohoo!  {cosine_sim} >= {threshold} Entity was matched --- [{obj1.name}:{obj1.label}] --merged--> [{obj1.name}:{best_match.label}]")
                return best_match
        return obj1

    def create_union_list(self, list1: List[Union[Entity, Relationship]], list2: List[Union[Entity, Relationship]]) -> List[Union[Entity, Relationship]]:
        """
        Create a union of two lists (Entity or Relationship objects), avoiding duplicates.
        If it's a relationship, matching will be based on the relationship's name.
        If it's a Entity, matching will be based on both the Entity's name and label.
        
        :param list1: First list of Entity or Relationship objects.
        :param list2: Second list of Entity or Relationship objects.
        :return: A unified list of Entity or Relationship objects.
        """
        union_list = list1.copy()

        # Store existing names and labels in the union list
        existing_Entity_key = {(obj.name, obj.label) for obj in union_list if isinstance(obj, Entity)}
        existing_relation_names = {obj.name for obj in union_list if isinstance(obj, Relationship)}

        for obj2 in list2:
            if isinstance(obj2, Entity):
                # For Entities, check both name and label to avoid duplicates
                if (obj2.name, obj2.label) not in existing_Entity_key:
                    union_list.append(obj2)
                    existing_Entity_key.add((obj2.name, obj2.label))

            elif isinstance(obj2, Relationship):
                # For relationships, check based on the name only
                if obj2.name not in existing_relation_names:
                    union_list.append(obj2)
                    existing_relation_names.add(obj2.name)

        return union_list
    
    
    def process_lists(self, 
                      list1: List[Union[Entity, Relationship]], 
                      list2: List[Union[Entity, Relationship]], 
                      threshold: float = 0.8
                      ) -> Tuple[List[Union[Entity, Relationship]], List[Union[Entity, Relationship]]]:
        """
        Process two lists to generate new lists based on specified conditions.
        :param list1: First list to process (local items).
        :param list2: Second list to be compared against (global items).
        :param for_entity_or_relation: Specifies whether the processing is for entities or relations.
        :return: (matched_local_items, new_global_items)
        """
        list3 = [self.find_match(obj1, list2, threshold=threshold) for obj1 in list1] #matched_local_items
        list4 = self.create_union_list(list3, list2) #new_global_items
        return list3, list(set(list4))

    def find_longest_string(self, names):
        """
    Finds the longest string in a list of strings.

    Args:
        names: A list of strings.

    Returns:
        The longest string in the list.
        Returns None if the input list is empty or contains non-string elements.
    """

        if not names:
            return None  # Handle empty list case

        longest_string = ""
        for name in names:
            if not isinstance(name, str):
                logging.info(f"Warning: Skipping non-string element: {name}")
                continue  # Skip non-string elements

            if len(name) > len(longest_string):
                longest_string = name

        return longest_string
    
    def match_entities_and_update_relationships(
                                                self,
                                                entities1: List[Entity],
                                                entities2: List[Entity],
                                                relationships1: List[Relationship],
                                                relationships2: List[Relationship],
                                                rel_threshold: float = 1,
                                                ent_threshold: float = 1
                                            ) -> Tuple[List[Entity], List[Relationship]]:
        """
        Match two lists of entities (Entities) and update the relationships list accordingly.
        :param entities1: First list of entities to match.
        :param entities2: Second list of entities to match against.
        :param relationships1: First list of relationships to update.
        :param relationships2: Second list of relationships to compare.
        :param rel_threshold: Cosine similarity threshold for relationships.
        :param ent_threshold: Cosine similarity threshold for entities.
        :return: Updated entities list and relationships list.
        """
        # Step 1: Match the entities and relations from both lists
        matched_entities1, global_entities = self.process_lists(entities1, entities2, ent_threshold)
        # matched_relations, _ = self.process_lists(relationships1, relationships2, rel_threshold)

        # Step 2: Create a mapping from old entity names to matched entity names
        entity_name_mapping = {
            entity: matched_entity 
            for entity, matched_entity in zip(entities1, matched_entities1) 
            if entity != matched_entity
        }

        # Step 3: Update relationships based on matched entities
        def update_relationships(relationships: List[Relationship]) -> List[Relationship]:
            updated_relationships = []
            for rel in relationships:
                updated_rel = rel.model_copy()  # Create a copy to modify
                # Update the 'startEntity' and 'endEntity' names with matched entity names
                if rel.startEntity in entity_name_mapping:
                    updated_rel.startEntity = entity_name_mapping[rel.startEntity]
                if rel.endEntity in entity_name_mapping:
                    updated_rel.endEntity = entity_name_mapping[rel.endEntity]
                updated_relationships.append(updated_rel)
            return updated_relationships

        # Step 4: Extend relationships2 with updated relationships
        relationships2.extend(update_relationships(relationships1))

        return global_entities, relationships2
    
    def merge_entities_relationship_by_unique_id(self, entities: List[Entity], relationships: List[Relationship]) -> Tuple[List[Entity], List[Relationship]]:
        """Merges entities and updates relationships based on unique_ID."""
        logging.info("Merging entities and relationships by unique_ID...")

        # Group entities by unique_ID
        grouped_entities = {}
        processed_entities = set()
        entities_output = []
        unique_entity=[]
        relationships_output = []
        
        for entity in entities:
            if entity.name == "alzheimer's disease (ad)":
                entity.properties_info['unique_id'] = "MESH:D000544"
            if entity.properties_info.get('unique_id'):
                unique_id = entity.properties_info.get('unique_id')
                if unique_id and unique_id != '-':
                    if unique_id not in grouped_entities:
                        grouped_entities[unique_id] = []
                    grouped_entities[unique_id].append(entity) #The grouped entities now has the actual object in it, not just names!
                else:
                    entities_output.append(entity)
            else:
                entities_output.append(entity)
            processed_entities.add(entity)
                
        #Process it in the next steps
        #Set to store entities that have already been processed, because some entities were added multiple times.

        for unique_ID, entity_group in grouped_entities.items():
            #If only 1 item there's no duplicates!
            if len(entity_group) <= 1:
                for entity in entity_group:
                    if entity not in processed_entities: #Check if it's already been proccessed and added
                        entities_output.append(entity)
                        processed_entities.add(entity)
                continue #Skip to the next object

            #If there is more than one entity, start the merging process
            # 1. Find the "best" entity to use as the representative
            best_entity = self.find_longest_string([entity.name for entity in entity_group if entity.name is not None and entity.name !=""]) #Protect from empty names!
            for entity in entity_group:
                if entity.name == best_entity:
                    main_entity = entity # Set the main Entity that you'll be referring too. It has the longest name!

            # Loop to assign relationships
            for r in relationships: #Iterate through all relationships
                #Update it, if the start or end entities are within the same object
                if hasattr(r, 'startEntity') and r.startEntity in entity_group:
                    r.startEntity = main_entity
                if hasattr(r, 'endEntity') and r.endEntity in entity_group:
                    r.endEntity = main_entity

            #Add everything in the correct spot.
            entities_output.append(main_entity) #The main_entity

            for entity in entity_group:
                if entity not in processed_entities: #Ensure it's not a duplicate from a previously run.
                    processed_entities.add(entity)
                    print ("WARNING: this entity wasn't updated!")

        #After you go throguh it all. Update the objects in the set and just the unique ones!
        for r in relationships:
            if r not in relationships_output:
                if r.startEntity != r.endEntity: #Add None Protection & Remove Reflexive
                    if r.startEntity in entities_output and r.endEntity in entities_output:
                        relationships_output.append(r)

        logging.info(f"Merged duplicate entities, original:{len(entities)} -> deduped:{len(entities_output)}, and relationships, original:{len(relationships)} -> deduped:{len(relationships_output)}.")
                    
        return entities_output, relationships_output
    
    def merge_relationships(self, relationships: List[Relationship]) -> List[Relationship]:
        """
        Merges relationships in a list where the start and end entities are the same,
        removing duplicates. If multiple relationships share the same start and end entities,
        a random name is chosen from the available names, and only one relationship object remains.

        Args:
            global_relationships (list): A list of relationship objects.

        Returns:
            list: A new list with merged and deduplicated relationships.
        """

        merged_relationships = []
        processed = set()  # To keep track of relationship indices already merged

        for i, ri in enumerate(relationships):
            if i in processed:
                continue  # Skip if already processed
            if ri.startEntity == ri.endEntity:
                continue
            
            for j, rj in enumerate(relationships):
                if i != j and ri.startEntity == rj.startEntity and ri.endEntity == rj.endEntity:
                    # Merge relationships with the same start and end entities
                    processed.add(j)  # Mark relationship as processed
        
            merged_relationships.append(ri)  # Add the merged relationship
            processed.add(i)

        return merged_relationships
    

