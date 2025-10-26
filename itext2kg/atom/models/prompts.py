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

'''
class AtomicFactsPrompt(Enum):
    system_query = """
    # Atomic Facts Evaluation Task

    You are an expert evaluator for factual information extraction. Your task is to meticulously compare a list of predicted atomic facts against a gold standard and calculate specific metrics.
    """
    @staticmethod
    def atomic_facts_system_query(ground_truth: list[str], predicted: list[str]) -> str:
        return f"""
    ## Input Format
    - **Gold Standard**: A set of reference atomic facts with their temporal information
    - **Predicted Atomic Facts**: A list of atomic facts with associated temporal information

    ## Evaluation Framework

    ### Phase 1: Content Evaluation
    Evaluate each predicted atomic fact's content against the gold standard without considering the temporal information:

    **MATCH**: A predicted atomic fact that accurately corresponds to a key fact explicitly stated in the gold standard.

    ### Phase 2: Temporal Evaluation
    **Only for atomic facts classified as MATCH in Phase 1**, evaluate their temporal components:

    **Temporal Match (MATCH_t)**: The predicted temporal information accurately corresponds to the temporal bounds stated or reasonably inferable from the gold standard for the matched fact.

    **Temporal Omission (OM_t)**: The gold standard specifies temporal bounds for a fact, but the predicted atomic fact either:
    - Provides no temporal information (null/empty temporal information)
    - Provides incomplete temporal information (missing t_start or t_end when both should be specified)

    ## Evaluation Instructions

    1. **First Pass**: Compare each predicted atomic fact against the gold standard
    - Count MATCH

    2. **Second Pass**: For each MATCH case only, evaluate temporal accuracy
    - Count MATCH_t, OM_t cases

    3. **Output the following counts**:
    - MATCH: [number]
    - MATCH_t: [number]
    - OM_t: [number]

    ## Important Notes
    - Be precise: semantic equivalence counts as a match (e.g., "John Smith" = "J. Smith" if referring to same entity)
    - Temporal evaluation only applies to content matches
    - Consider reasonable temporal tolerance based on the domain and precision of the gold standard

    ---
    Calculate for the following inputs:
    gold_standard: {ground_truth}
    predicted_atomic_facts: {predicted}
    """
'''

'''
class AtomicFactsPrompt(Enum):
    system_query = """
    # Atomic Facts Evaluation Task

    You are an expert evaluator for factual information extraction. Your task is to identify predicted atomic facts that match the gold standard in BOTH content and temporal accuracy.
    """
    
    @staticmethod
    def atomic_facts_system_query(ground_truth: list[str], predicted: list[str]) -> str:
        return f"""
## Evaluation Framework: Temporally-Aware Atomic Facts

Your task is to count how many predicted atomic facts fully match the gold standard (both content AND temporal information).

### What Constitutes a MATCH:

A predicted atomic fact is a **MATCH** if and only if:
1. **Content Accuracy**: The factual content accurately corresponds to a fact explicitly stated in the gold standard
   - Semantic equivalence counts (e.g., "John Smith" = "J. Smith" if same entity)
   - The core information must be the same
   
2. **Temporal Accuracy**: The temporal information is correct
   - If the gold standard specifies temporal bounds, the prediction must match them (within reasonable tolerance)
   - If the gold standard fact is atemporal, the prediction should not add temporal information
   - Both start and end dates must be accurate if specified in gold standard

### What is NOT a MATCH:

A predicted fact is NOT a match if:
- Content is not supported by the gold standard
- Content matches but temporal information is incorrect, hallucinated, or extends beyond what's stated
- Content matches but temporal information is missing when gold standard specifies it
- Content matches but temporal information is incomplete (e.g., missing start or end date when both should be present)

### Evaluation Instructions:

1. Go through each predicted atomic fact
2. For each prediction, ask:
   - Does the content match a gold standard fact?
   - Does the temporal information also match (or is appropriately absent)?
   - Only if BOTH are true → count as MATCH

3. **Output only the MATCH count**

### Important Notes:
- Be precise with temporal matching - consider domain-appropriate tolerance
- Atemporal facts in gold standard should remain atemporal in predictions
- A gold standard fact can only match one prediction (1-to-1 mapping)

---

**Input Data:**
- Gold Standard Facts: {ground_truth}
- Predicted Atomic Facts: {predicted}
"""
'''