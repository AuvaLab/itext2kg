


def change_rel(data):
    data = data.copy()
    data["startNode"] =  data["startNode"].lower()
    data["endNode"] = data["endNode"].lower()
    
    data["name"] = data["name"].replace(" ", "_").replace("-", "_").replace(".", "_").replace("&", "and")
    return data

def change_ent(data):
    data = data.copy()
    data["label"] = data["label"].replace(" ", "_").replace("-", "_").replace(".", "_").replace("&", "and")
    return data


def delete_duplicates_for_rel(records):
    seen = set()
    unique_records = []
    
    for record in records:
        # Define a key based on values that identify uniqueness
        identifier = (record['startNode'], record['endNode'], record['name'])
        
        # Check if the identifier is already in the seen set
        if identifier not in seen:
            # Add the identifier to the set
            seen.add(identifier)
            # Append the record to the unique list
            unique_records.append(record)
    
    #triplets = list(map(lambda triplet : (triplet["startNode"], triplet["name"], triplet["endNode"]), unique_records))
    return unique_records


def delete_duplicates_for_ent(records):
    seen = set()
    unique_records = []
    
    for record in records:
        # Define a key based on values that identify uniqueness
        identifier = (record['name'])
        
        # Check if the identifier is already in the seen set
        if identifier not in seen:
            # Add the identifier to the set
            seen.add(identifier)
            # Append the record to the unique list
            unique_records.append(record)
    
    #triplets = list(map(lambda triplet : (triplet["startNode"], triplet["name"], triplet["endNode"]), unique_records))
    return unique_records

