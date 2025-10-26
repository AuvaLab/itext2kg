from typing import List, Optional
from pydantic import BaseModel, Field, field_validator



# ----------------- Company Website  ---------------- # 
class FeatureDescription(BaseModel):
    feature_name : str = Field(description="the name of the product or service provided by the company")
    feature_caracteristics : List = Field(description="caracteristics of this product of service")
    
class JobDescription(BaseModel):
    job_title : str = Field(description="the title of the opening job")
    job_caracteristics : List = Field(description="caracteristics of opening job")

class EmployeeDescription(BaseModel):
    employee_name : str = Field(description="the name of the employee")
    employee_function : List = Field(description="the function of the employee")
    
class InformationRetriever(BaseModel):
    company_name: str = Field(description="name of the company")
    products_or_services : FeatureDescription = Field(description="the features of the products or services provided by the company")
    jobs : JobDescription = Field(description="the opening jobs proposed by the company")
    clients : List = Field(description= "the clients of the company")
    team : EmployeeDescription = Field(description="the team of the company")
    
# ----------------- Scientific Article ---------------- # 
class Author(BaseModel):
    name : str=Field(description="The name of the article's author. The right name should be near the article's title.")
    affiliation: str = Field(description="The affiliation of the article's author")
    
class ArticleDescription(BaseModel):
    title : str = Field(description="The title of the scientific article")
    authors : List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract:str = Field(description="Brief summary of the article's abstract")

    
class Article(BaseModel):
    title : str = Field(description="The title of the scientific article")
    authors : List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract:str = Field(description="The article's abstract")
    key_findings:str = Field(description="The key findings of the article")
    limitation_of_sota : str=Field(description="limitation of the existing work")
    proposed_solution : str = Field(description="the proposed solution in details")
    paper_limitations : str=Field(description="The limitations of the proposed solution of the paper")
    
# ---------------- Entities & Relationships Extraction --------------------------- #

    
class Entity(BaseModel):
    label: str = Field(
        description=(
            "The semantic category of the entity (e.g., 'Person', 'Event', 'Location', 'Methodology', 'Position'). "
            "Use 'Relationship' objects if the concept is inherently relational or verbal (e.g., 'plans'). "
            "Prefer consistent, single-word categories where possible (e.g., 'Person', not 'Person_Entity'). "
        )
    )
    name: str = Field(
        description=(
            "The unique name or title identifying this entity, representing exactly one concept. "
            "For example, 'Yassir', 'CEO', or 'X'. Avoid combining multiple concepts (e.g., 'CEO of X'), "
            "since linking them should be done via Relationship objects. "
            "Verbs or multi-concept phrases (e.g., 'plans an escape') typically belong in Relationship objects. "
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
            "A single, canonical predicate capturing how the startNode and endNode relate (e.g., 'is_CEO', "
            "'holds_position', 'located_in'). Avoid compound verbs (e.g., 'plans_and_executes'). "
            "AVOID relation names as prepositions 'of', 'in' or similar."
        )
    )

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship] = Field(
        description=(
            "Based on the provided entities and context, identify the predicates that define relationships between these entities. "
            "The predicates should be chosen with precision to accurately reflect the expressed relationships."
        )
    )
    

class EntitiesExtractor(BaseModel):
    entities: List[Entity] = Field(
        description=(
            "A list of distinct entities extracted from text, each encoding exactly one concept "
            "(e.g., Person('Yassir'), Position('CEO'), Organization('X')). "
            "If verbs or actions appear, place them in a Relationship object rather than as an Entity. "
            "For instance, 'haira plans an escape' should yield separate Entities for Person('Haira'), Event('Escape'), "
            "and possibly a Relationship('Haira' -> 'plans' -> 'Escape')."
        )
    )

    
# ---------------------------- CV ------------------------------------- #

class WorkExperience(BaseModel):
    title: str
    company: str
    location: str
    start_date: str
    end_date: str
    responsibilities: List[str]

class Education(BaseModel):
    degree: str
    institution: str
    location: str
    start_date: str
    end_date: str
    coursework: Optional[List[str]]

class CV(BaseModel):
    name: str = Field(..., description="The name of the profile")
    phone_number: str = Field(..., description="The phone number of the profile")
    email: Optional[str] = Field(None, description="The email address of the profile")
    linkedin: Optional[str] = Field(None, description="The LinkedIn profile URL")
    summary: str = Field(..., description="A summary or professional profile")
    work_experience: List[WorkExperience] = Field(..., description="List of work experiences")
    education: List[Education] = Field(..., description="List of educational qualifications")
    skills: List[str] = Field(..., description="List of skills")
    certifications: Optional[List[str]] = Field(None, description="List of certifications")
    languages: Optional[List[str]] = Field(None, description="List of languages known")
    volunteer_work: Optional[List[str]] = Field(None, description="List of volunteer work experiences")
    


# ---------------------------- Facts ------------------------------------- #

class Facts(BaseModel):
    facts: list[str] = Field(
        description="""
        **Guidelines for Generating Facts**:

        1. **Facts**:
           - Extract the facts from the text.
           - Convert compound or complex sentences into short, single-fact statements.
           - Each Fact must contain exactly one piece of information or relationship.
           - Ensure that each Fact is expressed directly and concisely, without redundancies or duplicating the same information across multiple statements.
           
        2. **Decontextualization**:
           - Replace pronouns (e.g., "it," "he," "they") with the full entity name or a clarifying noun phrase.
           - Include any necessary modifiers so that each Fact is understandable in isolation.

        3. **Temporal Context**:
           - If the text contains explicit time references (e.g., "in 1995," "next Tuesday," "during the 20th century"), 
             include them in the Fact so it is clear when the statement was or will be true.
           - Position the time reference in a natural place within the Fact.
           - If a sentence references multiple distinct times, split it into separate Facts as needed.

        4. **Accuracy & Completeness**:
           - Preserve the original meaning without combining multiple facts into a single statement.
           - Avoid adding details not present in the source text.

        5.  **End Actions**:
           - If the text indicates the end of a role or an action (for example, someone leaving a position),
             be explicit about the role/action and the time it ended.
        
        **Redundancies**:
        - Eliminate redundancies by simplifying phrases (e.g., convert "the method is crucial for maintaining X" into "the method maintains X").
        
        **Example**:
        -During the height of the Cold War, the Apollo 11 mission, which was launched by NASA from Kennedy Space Center on July 16, 1969, successfully landed the first two humans, Neil Armstrong and Buzz Aldrin, on the Moon on July 20, 1969, a monumental achievement that was watched by millions worldwide and effectively ended the Space Race between the United States and the Soviet Union.
        Facts:[
        "The Apollo 11 mission was launched by NASA.",
        "The Apollo 11 mission was launched from Kennedy Space Center.",
        "The Apollo 11 mission was launched on July 16, 1969.",
        "The Apollo 11 mission landed the first two humans on the Moon.",
        "Neil Armstrong was one of the first two humans to land on the Moon.",
        "Buzz Aldrin was one of the first two humans to land on the Moon.",
        "The Moon landing occurred on July 20, 1969.",
        "The Apollo 11 mission's success effectively ended the Space Race.",
        "The Space Race was between the United States and the Soviet Union."
        ]
        """
    )
    @field_validator('facts', mode='before')
    @classmethod
    def validate_facts(cls, v):
        if isinstance(v, str):
            return [v]  # Convert single string to list
        return v  # Return as-is if already a list


