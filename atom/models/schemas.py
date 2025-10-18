from typing import List, Optional
from pydantic import BaseModel, Field


# ------------------------------- Temporal Atomic Facts Extraction ---------------------------------------- #

class Factoid(BaseModel):
    phrase: list[str] = Field(
        description="""
        **Guidelines for Generating Temporal Factoids**:

        1. **Atomic Factoids**:
           - Convert compound or complex sentences into short, single-fact statements.
           - Each Factoid must contain exactly one piece of information or relationship.
           - Ensure that each Factoid is expressed directly and concisely, without redundancies or duplicating the same information across multiple statements.
            For example, Unsupervised learning is dedicated to discovering intrinsic patterns in unlabeled datasets becomes "Unsupervised learning discovers patterns in unlabeled data."
        2. **Decontextualization**:
           - Replace pronouns (e.g., "it," "he," "they") with the full entity name or a clarifying noun phrase.
           - Include any necessary modifiers so that each Factoid is understandable in isolation.

        3. **Temporal Context**:
           - If the text contains explicit time references (e.g., "in 1995," "next Tuesday," "during the 20th century"), 
             include them in the Factoid so it is clear when the statement was or will be true.
           - Position the time reference in a natural place within the Factoid.
           - If a sentence references multiple distinct times, split it into separate Factoids as needed.

        4. **Accuracy & Completeness**:
           - Preserve the original meaning without combining multiple facts into a single statement.
           - Avoid adding details not present in the source text.

        5.  **End Actions**:
           - If the text indicates the end of a role or an action (for example, someone leaving a position),
             be explicit about the role/action and the time it ended.
        
        **Redundancies**:
        - Eliminate redundancies by simplifying phrases (e.g., convert "the method is crucial for maintaining X" into "the method maintains X").

        **Example**:
        On June 18, 2024, Real Madrid won the Champions League final with a 2-1 victory. Following the triumph, fans of Real Madrid celebrated the Champions League victory across the city.
        -Real Madrid won the Champions League final on June 18, 2024.
        -The final Champions League final ended with a 2-1 victory for Real Madrid on June 18, 2024.
        -Fans of Real Madrid celebrated the victory of Champions League final across the city on June 18, 2024.
        """
    )

class AtomicFact(BaseModel):
    atomic_fact: list[str] = Field(
        description=''' 
        You are an expert factoid extraction engine. Your primary function is to read a news paragraph and its associated observation date, and then decompose the text into a comprehensive list of atomic, self-contained, and temporally-grounded facts.

## Task
Given an input paragraph and an `observation_date`, generate a list of all distinct factoids present in the text.

## Guidelines for Generating Temporal Factoids

### 1. Atomic Factoids
- Convert compound or complex sentences into short, single-fact statements
- Each factoid must contain exactly one piece of information or relationship
- Ensure that each factoid is expressed directly and concisely, without redundancies or duplicating the same information across multiple statements
- **Example:** "Unsupervised learning is dedicated to discovering intrinsic patterns in unlabeled datasets" becomes "Unsupervised learning discovers patterns in unlabeled data"

### 2. Decontextualization
- Replace pronouns (e.g., "it," "he," "they") with the full entity name or a clarifying noun phrase
- Include any necessary modifiers so that each factoid is understandable in isolation

### 3. Temporal Context
- Convert ALL time references to absolute dates/times using the observation_date

#### Conversion Rules:
- "today" → exact observation_date
- "yesterday" → observation_date minus 1 day
- "this week" → Monday of observation_date's week
- "last week" → Monday of the week before observation_date
- "this month" → first day of observation_date's month
- "last month" → first day of the month before observation_date
- "this year" → January 1st of observation_date's year
- "last year" → January 1st of the year before observation_date
- Keep explicit dates as-is (e.g., "June 18, 2024")

#### Additional Temporal Guidelines:
- Position time references naturally within factoids
- Split sentences with multiple time references into separate factoids
- **NEVER include relative terms like "today," "yesterday," "last week" in the final factoids**

### 4. Accuracy & Completeness
- Preserve the original meaning without combining multiple facts into a single statement
- Avoid adding details not present in the source text

### 5. End Actions
- If the text indicates the end of a role or an action (for example, someone leaving a position), be explicit about the role/action and the time it ended

### 6. Redundancies
- Eliminate redundancies by simplifying phrases
- **Example:** Convert "the method is crucial for maintaining X" into "the method maintains X"

## Example

**Input:** "On June 18, 2024, Real Madrid won the Champions League final with a 2-1 victory. Following the triumph, fans of Real Madrid celebrated the Champions League victory across the city."

**Output:**
- Real Madrid won the Champions League final on June 18, 2024
- The Champions League final ended with a 2-1 victory for Real Madrid on June 18, 2024
- Fans of Real Madrid celebrated the Champions League victory across the city on June 18, 2024
        '''
    )
# ---------------- Entities & Relationships Extraction --------------------------- #


class Entity(BaseModel):
    label: str = Field(
        description=(
            "The semantic category of the entity (e.g., 'Person', 'Event', 'Location', 'Methodology', 'Position') as you understand it from the text. "
            "Use 'Relationship' objects if the concept is inherently relational or verbal (e.g., 'plans'). "
            "Prefer consistent, single-word categories where possible (e.g., 'Person', not 'Person_Entity'). "
            "Do not extract Date entities as they will be integrated in the relation."
        )
    )
    name: str = Field(
        description=(
            "The unique name or title identifying this entity, representing exactly one concept. "
            "For example, 'Yassir', 'CEO', or 'X'. Avoid combining multiple concepts (e.g., 'CEO of X'), "
            "since linking them should be done via Relationship objects. "
            "Verbs or multi-concept phrases (e.g., 'plans an escape') typically belong in Relationship objects. "
            "Do not extract Date entities as they will be integrated in the relation."
        )
    )

class EntitiesExtractor(BaseModel):
    entities: List[Entity] = Field(
        description=(
            "A list of distinct entities extracted from text, each encoding exactly one concept "
            "(e.g., Person('Yassir'), Position('CEO'), Organization('X')). "
            "If verbs or actions appear, place them in a Relationship object rather than as an Entity. "
            "For instance, 'Haira plans an escape' should yield separate Entities for Person('Haira'), Event('Escape'), "
            "and possibly a Relationship('Haira' -> 'plans' -> 'Escape')."
        )
    )

class Relationship(BaseModel):
    startNode: Entity = Field(
        description=(
            "The 'subject' or source entity of this relationship, which must appear in the EntitiesExtractor."
        )
    )
    endNode: Entity = Field(
        description=(
            "The 'object' or target entity of this relationship, which must also appear in the EntitiesExtractor."
        )
    )
    name: str = Field(
        description=(
            "A single, canonical predicate in PRESENT TENSE capturing how the startNode and endNode relate "
            "(e.g., 'is_CEO', 'holds_position', 'is_located_in', 'works_at'). "
            "ALWAYS use present tense verbs regardless of the temporal context in the text. "
            "Avoid compound verbs (e.g., 'plans_and_executes'). "
            "If the text implies negation (e.g., 'no longer CEO'), still use the affirmative present form (e.g., 'is_CEO') "
            "and rely on 't_end' for the end date. "
            "AVOID preposition-only relation names like 'of', 'in', 'at' - use descriptive present-tense verbs instead."
        )
    )
    t_start: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "A time or interval indicating when this relationship begins or is active. "
            "Resolve relative temporal expressions based on the observation_date:\n"
            "  - 'today' → exact observation_date\n"
            "  - 'yesterday' → observation_date minus 1 day\n"
            "  - 'this week' → Monday of observation_date's week\n"
            "  - 'last week' → Monday of the week before observation_date\n"
            "  - 'this month' → first day of observation_date's month\n"
            "  - 'last month' → first day of the month before observation_date\n"
            "  - 'this year' → January 1st of observation_date's year\n"
            "  - 'last year' → January 1st of the year before observation_date\n"
            "Keep explicit dates as-is (e.g., '18-06-2024'). "
            "For example, if 'Yassir became CEO from 2023', then t_start=['01-01-2023']. "
            "This can be a single year, a date, or a resolved relative reference. "
            "Leave it [] if not specified."
        )
    )
    t_end: Optional[list[str]] = Field(
        default_factory=list,
        description=(
            "A time or interval indicating when this relationship ceases to hold. "
            "Resolve relative temporal expressions based on the observation_date using the same rules as t_start:\n"
            "  - 'today' → exact observation_date\n"
            "  - 'yesterday' → observation_date minus 1 day\n"
            "  - 'this week' → Monday of observation_date's week\n"
            "  - etc. (same resolution rules as t_start)\n"
            "Keep explicit dates as-is. "
            "For example, if 'Yassir left his position in 2025', then t_end=['01-01-2025']. "
            "Use this field to capture any 'end action' (e.g., leaving a job, ending a marriage), "
            "while keeping the relationship name in a canonical present tense form (e.g., 'is_CEO' not 'was_CEO'). "
            "Leave it [] if no end date/time is given."
        )
    )

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship] = Field(
        description=(
            "Based on the provided entities and context, identify the predicates that define relationships between these entities. "
            "The predicates should be chosen with precision to accurately reflect the expressed relationships. "
            "CRITICAL: All relationship names must be in PRESENT TENSE (e.g., 'works_at', 'is_CEO', 'manages') "
            "regardless of whether the relationship is past, present, or future. Use t_start and t_end to capture temporal bounds."
        )
    )

'''
# ---------------- LLM as Judge for Quintuples Extraction --------------------------- #

class QuintuplesJudge(BaseModel):
    """
    Structured output for Knowledge Graph Quintuple Evaluation
    """
    
    # Content Evaluation Metrics (LLM provides these)
    MATCH: int = Field(description="Number of predicted (s,p,o) that accurately correspond to gold standard")
    HALL: int = Field(description="Number of predicted (s,p,o) that contain information NOT supported by gold standard")
    OM: int = Field(description="Number of key facts in gold standard NOT captured by any predicted quintuple")
    
    # Temporal Evaluation Metrics (LLM provides these)
    MATCH_t: int = Field(description="Number of temporal bounds that accurately correspond to gold standard")
    HALL_t: int = Field(description="Number of temporal bounds that are incorrect or unsupported")
    OM_t: int = Field(description="Number of temporal bounds missing from predictions")

    def calculate_metrics(self) -> dict:
        """Calculate derived metrics from base counts"""
        
        # Content metrics
        content_precision = self.MATCH / (self.MATCH + self.HALL) if (self.MATCH + self.HALL) > 0 else 0.0
        content_recall = self.MATCH / (self.MATCH + self.OM) if (self.MATCH + self.OM) > 0 else 0.0
        content_f1 = (2 * content_precision * content_recall) / (content_precision + content_recall) if (content_precision + content_recall) > 0 else 0.0
        
        # Temporal metrics
        temporal_precision = self.MATCH_t / (self.MATCH_t + self.HALL_t) if (self.MATCH_t + self.HALL_t) > 0 else 0.0
        temporal_recall = self.MATCH_t / (self.MATCH_t + self.OM_t) if (self.MATCH_t + self.OM_t) > 0 else 0.0
        temporal_f1 = (2 * temporal_precision * temporal_recall) / (temporal_precision + temporal_recall) if (temporal_precision + temporal_recall) > 0 else 0.0
        
        # Overall metrics
        total_correct = self.MATCH + self.MATCH_t
        total_predicted = self.MATCH + self.MATCH_t + self.HALL + self.HALL_t
        total_gold = self.MATCH + self.MATCH_t + self.OM + self.OM_t
        
        overall_precision = total_correct / total_predicted if total_predicted > 0 else 0.0
        overall_recall = total_correct / total_gold if total_gold > 0 else 0.0
        overall_f1 = (2 * overall_precision * overall_recall) / (overall_precision + overall_recall) if (overall_precision + overall_recall) > 0 else 0.0
        
        return {
            "content_precision": content_precision,
            "content_recall": content_recall,
            "content_f1": content_f1,
            "temporal_precision": temporal_precision,
            "temporal_recall": temporal_recall,
            "temporal_f1": temporal_f1,
            "overall_precision": overall_precision,
            "overall_recall": overall_recall,
            "overall_f1": overall_f1,
        }

# ---------------- LLM as Judge for Atomic Facts Extraction --------------------------- #

class AtomicFactsJudge(BaseModel):
    """
    Structured output for Atomic Facts Evaluation with Temporally-Aware Matching
    """
    
    # Temporally-Aware Evaluation Metrics (LLM provides these)
    MATCH: int = Field(
        description="Number of predicted atomic facts where BOTH content and temporal information accurately match the gold standard"
    )

    def calculate_metrics(self, n_predicted: int, n_gold: int) -> dict:
        """Calculate precision, recall, and F1 from base counts"""

        OM = n_gold - self.MATCH
        HALL = n_predicted - self.MATCH
        # Metrics
        precision = self.MATCH / (self.MATCH + HALL) if (self.MATCH + HALL) > 0 else 0.0
        recall = self.MATCH / (self.MATCH + OM) if (self.MATCH + OM) > 0 else 0.0
        f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0
        
        return {
            "MATCH": self.MATCH,
            "HALL": HALL,
            "OM": OM,
            "precision": precision,
            "recall": recall,
            "f1": f1,
        }
'''