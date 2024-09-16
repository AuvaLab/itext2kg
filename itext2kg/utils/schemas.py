from typing import List, Optional
from pydantic import BaseModel, Field



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
    name : str=Field(description="The name of the author")
    affiliation: str = Field(description="The affiliation of the author")
    
class ArticleDescription(BaseModel):
    title : str = Field(description="The title of the scientific article")
    authors : List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract:str = Field(description="Brief summary of the article's abstract")
    key_findings:str = Field(description="The key findings of the article")

    
class Article(BaseModel):
    title : str = Field(description="The title of the scientific article")
    authors : List[Author] = Field(description="The list of the article's authors and their affiliation")
    abstract:str = Field(description="The article's abstract")
    key_findings:str = Field(description="The key findings of the article")
    limitation_of_sota : str=Field(description="limitation of the existing work")
    proposed_solution : str = Field(description="the proposed solution in details")
    paper_limitations : str=Field(description="The limitations of the proposed solution of the paper")
    
# ---------------- Entities & Relationships Extraction --------------------------- #

class Property(BaseModel):
    name : str = Field("The name of the entity. An entity should encode ONE concept.")
    
class Entity(BaseModel):
    label : str = Field("The type or category of the entity, such as 'Process', 'Technique', 'Data Structure', 'Methodology', 'Person', etc. This field helps in classifying and organizing entities within the knowledge graph.")
    name : str = Field("The specific name of the entity. It should represent a single, distinct concept and must not be an empty string. For example, if the entity is a 'Technique', the name could be 'Neural Networks'.")
    
class EntitiesExtractor(BaseModel):
    entities : List[Entity] = Field("All the entities presented in the context. The entities should encode ONE concept.")
    
class Relationship(BaseModel):
    startNode: str = Field("The starting entity, which is present in the entities list.")
    endNode: str = Field("The ending entity, which is present in the entities list.")
    name: str = Field("The predicate that defines the relationship between the two entities. This predicate should represent a single, semantically distinct relation.")

class RelationshipsExtractor(BaseModel):
    relationships: List[Relationship] = Field("Based on the provided entities and context, identify the predicates that define relationships between these entities. The predicates should be chosen with precision to accurately reflect the expressed relationships.")
    
    
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