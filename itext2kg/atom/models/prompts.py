from enum import Enum


class Prompt(Enum):
    EXAMPLES = """ 
    FEW SHOT EXAMPLES \n

    * Michel served as CFO at Acme Corp from 2019 to 2021. He was hired by Beta Inc in 2021, but left that role in 2023.
    -> (Michel, is_CFO_of, Acme Corp, ["01-01-2019"], ["01-01-2021"]), (Michel, works_at, Beta Inc, ["01-01-2021"], ["01-01-2023"])

    * Subsequent experiments confirmed the role of microRNAs in modulating cell growth.
    -> (Experiments, confirm_role_of, microRNAs, [], []), (microRNAs, modulate, Cell Growth, [], [])

    * Researchers used high-resolution imaging in a study on neural plasticity.
    -> (Researchers, use, High-Resolution Imaging, [], []), (High-Resolution Imaging, is_used_in, Study on Neural Plasticity, [], [])

    * Sarah was a board member of GreenFuture until 2019.
    -> (Sarah, is_board_member_of, GreenFuture, [], ["01-01-2019"])

    * Dr. Lee was the head of the Oncology Department until 2022.
    -> (Dr. Lee, is_head_of, Oncology Department, [], ["01-01-2022"])

    * Activity-dependent modulation of receptor trafficking is crucial for maintaining synaptic efficacy.
    -> (Activity-Dependent Modulation, involves, Receptor Trafficking, [], []), (Receptor Trafficking, maintains, Synaptic Efficacy, [], [])

    * (observation_date = 2024-06-15) John Doe is no longer the CEO of GreenIT a few months ago.
    -> (John Doe, is_CEO_of, GreenIT, [], ["2024-03-15"])
    # "a few months ago" ≈ 3 months → 2024-06-15 minus 3 months = 2024-03-15

    * John Doe's marriage is happening on 26-02-2026.
    -> (John Doe, has_status, Married, ["2026-02-26"], [])

    * (observation_date = 2024-03-20) The AI Summit conference started yesterday and will end tomorrow.
    -> (AI Summit, has_status, Started, ["2024-03-19"], ["2024-03-21"])

    * The independence day of Morocco is celebrated on January 1st each year since 1956.
    -> (Morocco, celebrates, Independence Day, ["1956-01-01"], [])

    * (observation_date = 2024-08-10) The product launch event is scheduled for next month.
    -> (Product Launch, has_status, Scheduled, ["2024-09-01"], [])
    # "next month" = first day of September 2024
    
    """
    
    @staticmethod
    def temporal_system_query(obs_timestamp: str) -> str:
        return f""" 
        Observation Time : {obs_timestamp}
        
        You are a top-tier algorithm designed for extracting information in structured 
        formats to build a knowledge graph.
        Try to capture as much information from the text as possible without 
        sacrificing accuracy. Do not add any information that is not explicitly mentioned in the text
        Remember, the knowledge graph should be coherent and easily understandable, 
        so maintaining consistency in entity references is crucial.
        """

    @staticmethod
    def atomic_facts_system_query(obs_timestamp: str, domains: list[str] | None = None) -> str:
        """System prompt for atomic-fact extraction.

        When ``domains`` is empty/None, no domain classification is requested.
        When non-empty, each fact must be labeled with one of the allowed domains.
        """
        domains = list(domains or [])
        base = f"""
Observation Time: {obs_timestamp}

You are an expert factoid extraction engine. Read the input paragraph and
decompose it into atomic, self-contained, temporally-grounded facts in
SIMPLE PRESENT TENSE. Decontextualize pronouns. Convert relative time
references using the observation date. Do not invent facts.
"""
        if not domains:
            return base + """
Do NOT assign domain labels. Leave the domain field empty for every fact.
"""
        allowed = ", ".join(f"'{d}'" for d in domains)
        return base + f"""
For each fact, assign exactly one domain from this allowed list: [{allowed}].
Do not invent domains outside this list. If a fact does not fit any allowed
domain, discard it rather than inventing a new domain label.
"""
