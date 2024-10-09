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

class Property(BaseModel):
    name : str = Field("The name of the entity. An entity should encode ONE concept.")
    
class Entity(BaseModel):
    label : str = Field("The type or category of the entity, such as 'Process', 'Technique', 'Data Structure', 'Methodology', 'Person', etc. This field helps in classifying and organizing entities within the knowledge graph.")
    name : str = Field("The specific name of the entity. It should represent a single, distinct concept and must not be an empty string. For example, if the entity is a 'Technique', the name could be 'Neural Networks'.")
    
class EntitiesExtractor(BaseModel):
    entities : List[Entity] = Field("All the entities presented in the context. The entities should encode ONE concept.")
    
class Relationship(BaseModel):
    startNode: Entity = Field("The starting entity, which is present in the entities list.")
    endNode: Entity = Field("The ending entity, which is present in the entities list.")
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
    
# ---------------------------- News ------------------------------------- #

class Fact(BaseModel):
    statement: str = Field(description="A factual statement mentioned in the news article")
    source: Optional[str] = Field(description="The source of the fact, if mentioned")
    relevance: Optional[str] = Field(description="The relevance or importance of the fact to the overall article")

class ArticleContent(BaseModel):
    headline: str = Field(description="The title or headline of the news article")
    subheading: Optional[str] = Field(description="The subheading or supporting title of the article")
    facts: List[Fact] = Field(description="List of factual statements covered in the article")
    keywords: List[str] = Field(description="List of keywords or topics covered in the article")
    publication_date: str = Field(description="The publication date of the article")
    location: Optional[str] = Field(description="The location relevant to the article")

class NewsArticle(BaseModel):
    title: str = Field(description="The title or headline of the news article")
    author: Author = Field(description="The author of the article")
    content: ArticleContent = Field(description="The body and details of the news article")

# ---------------------------- Novels ------------------------------------- #
class Character(BaseModel):
    name: str = Field(description="The name of the character in the novel")
    role: str = Field(description="The role of the character in the story, e.g., protagonist, antagonist, etc.")
    description: Optional[str] = Field(description="A brief description of the character's background or traits")

class PlotPoint(BaseModel):
    chapter_number: int = Field(description="The chapter number where this event occurs")
    event: str = Field(description="A significant event or plot point that occurs in the chapter")

class Novel(BaseModel):
    title: str = Field(description="The title of the novel")
    author: str = Field(description="The author of the novel")
    genre: str = Field(description="The genre of the novel")
    characters: List[Character] = Field(description="The list of main characters in the novel")
    plot_summary: str = Field(description="A brief summary of the overall plot")
    key_plot_points: List[PlotPoint] = Field(description="Key plot points or events in the novel")
    themes: Optional[List[str]] = Field(description="Main themes explored in the novel, e.g., love, revenge, etc.")