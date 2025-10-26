import re
import numpy as np
from typing import List, Union, Optional
from pydantic import BaseModel, Field, ConfigDict
from dateutil import parser
from itext2kg.atom.models.entity import Entity

class BaseModelWithConfig(BaseModel):
    model_config = ConfigDict(
        arbitrary_types_allowed=True,
        validate_assignment=False,
        extra="ignore"
    )

LABEL_PATTERN = re.compile(r'[^a-zA-Z0-9]+')  # For cleaning labels

class RelationshipProperties(BaseModelWithConfig):
    embeddings:   Optional[np.ndarray]  = None
    atomic_facts:       List[str]         = []
    t_obs:   List[float] = []
    t_start:      List[float] = []
    t_end:    List[float] = []


# -------------------------------------------
# Relationship model
# -------------------------------------------
class Relationship(BaseModelWithConfig):
    startEntity: Entity = Field(default_factory=Entity)
    endEntity:   Entity = Field(default_factory=Entity)
    name:        str    = ""
    properties:  RelationshipProperties = Field(default_factory=RelationshipProperties)

    def process(self) -> "Relationship":
        self.name = LABEL_PATTERN.sub("_", self.name).replace("&", "and").lower()
        return self
    
    def combine_timestamps(
        self,
        timestamps: Union[List[float], List[str]],
        temporal_aspect: str  # Should be one of: "t_obs", "t_start", "t_end"
    ) -> None:
        # If timestamps is not empty, process based on element type.
        processed_timestamps: List[float] = []
        if timestamps:
            if isinstance(timestamps[0], str):
                for ts in timestamps:
                    try:
                        # Cast to str to satisfy type checker since we know it's a string from isinstance check
                        parsed_dt = parser.parse(str(ts))
                        if parsed_dt is not None:
                            processed_timestamps.append(parsed_dt.timestamp())
                    except Exception as e:
                        # Log the error but continue processing other timestamps
                        print(f"Warning: Could not parse timestamp '{ts}': {e}. Skipping this timestamp.")
                        # Keep the place empty by simply not adding anything to the list
                        continue
            elif isinstance(timestamps[0], float):
                # Cast the list to List[float] since we know it contains floats
                processed_timestamps = [float(t) for t in timestamps]  # Explicit conversion to ensure proper typing
            else:
                raise ValueError("Invalid timestamp format. Please provide a list of strings or a list of floats.")
        
        # Extend the appropriate property with timestamps (even if the list is empty).
        if temporal_aspect == "t_obs":
            self.properties.t_obs.extend(processed_timestamps)
        elif temporal_aspect == "t_start":
            self.properties.t_start.extend(processed_timestamps)
        elif temporal_aspect == "t_end":
            self.properties.t_end.extend(processed_timestamps)
        else:
            raise ValueError("Invalid temporal aspect. Please provide either 'timestamps', 't_start' or 't_end'.")

    def combine_atomic_facts(self, atomic_facts: List[str]) -> None:
        """Combines atomic facts by appending the new atomic fact if it's different from existing ones."""
        self.properties.atomic_facts.extend(atomic_facts)
            
    def __eq__(self, other) -> bool:
        """Checks equality without considering timestamps."""
        if isinstance(other, Relationship):
            return (self.startEntity == other.startEntity
                    and self.endEntity == other.endEntity
                    and self.name == other.name)
        return False
    
    def __eq_with_timestamps__(self, other) -> bool:
        """Checks equality considering timestamps as a major differentiator."""
        if isinstance(other, Relationship):
            return (self.startEntity == other.startEntity
                    and self.endEntity == other.endEntity
                    and self.name == other.name
                    and set(self.properties.timestamps) == set(other.properties.timestamps))  # Timestamps must match exactly
        return False

    def __hash__(self) -> int:
        return hash((self.name, self.startEntity, self.endEntity))

    def __repr__(self) -> str:
        return (f"Relationship(name={self.name!r}, "
                f"startEntity={self.startEntity!r}, "
                f"endEntity={self.endEntity!r}, "
                f"properties={self.properties!r})")
