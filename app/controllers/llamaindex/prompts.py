from llama_index.core import Prompt
from llama_index.core import PromptTemplate
from llama_index.core.prompts.prompt_type import PromptType
                                  
# Prompt for submitting the query results along with the original question to the
# LLM to generate an answer.                             
text_qa_template = Prompt("""\
"Matching article information is below.\n"
"---------------------\n"
"{context_str}\n"
"---------------------\n"
"Given the above information about the articles and not prior knowledge, "
"answer the question: {query_str}\n"
""")
                   
# If there are too many chunks to stuff in one prompt, “create and refine”
# an answer by going through multiple compact prompts.
refine_template = Prompt("""\
"The original question is as follows: {query_str}\n"
"We have provided an existing answer: {existing_answer}\n"
"We have the opportunity to refine the existing answer "
"(only if needed) with some more context below.\n"
"------------\n"
"{context_msg}\n"
"------------\n"
"Given the new context, refine the original answer to better "
"answer the question. "
"If the context isn't useful, return the original answer."
"Make sure the existing answer match with original questions correctly. Remove unrelevant context."
""")

# Prompt for instructing the agent the system rules
system_prompt = '''
You are a professional data scientist specialized in Natural Language Processing. You are a helpful assistant and always explain things in details.
Here are some rules to follow when answering the query:
1. Always give the answer in point form.
2. Always provide details, steps, explaination and examples to answer the question.
3. You are always be professional!
'''

# Knowledge Graph Triplet Extraction Prompt
DEFAULT_KG_TRIPLET_EXTRACT_TMPL = (
    "Some text is provided below. Given the text, extract up to "
    "{max_knowledge_triplets} "
    '''## 1. Overview
    You are a top-tier algorithm designed for extracting information in structured formats to build a knowledge graph.
    - **Nodes** represent entities and concepts. They're akin to Wikipedia nodes.
    - The aim is to achieve simplicity and clarity in the knowledge graph, making it accessible for a vast audience.
    ## 2. Labeling Nodes
    - **Consistency**: Ensure you use basic or elementary types for node labels.
    - For example, when you identify an entity representing a person, always label it as **"person"**. Avoid using more specific terms like "mathematician" or "scientist".
    - **Node IDs**: Never utilize integers as node IDs. Node IDs should be names or human-readable identifiers found in the text.
    ## 3. Handling Numerical Data and Dates
    - Numerical data, like age or other related information, should be incorporated as attributes or properties of the respective nodes.
    - **No Separate Nodes for Dates/Numbers**: Do not create separate nodes for dates or numerical values. Always attach them as attributes or properties of nodes.
    - **Property Format**: Properties must be in a key-value format.
    - **Quotation Marks**: Never use escaped single or double quotes within property values.
    - **Naming Convention**: Use camelCase for property keys, e.g., `birthDate`.
    ## 4. Coreference Resolution
    - **Maintain Entity Consistency**: When extracting entities, it's vital to ensure consistency.
    If an entity, such as "John Doe", is mentioned multiple times in the text but is referred to by different names or pronouns (e.g., "Joe", "he"), 
    always use the most complete identifier for that entity throughout the knowledge graph. In this example, use "John Doe" as the entity ID.  
    Remember, the knowledge graph should be coherent and easily understandable, so maintaining consistency in entity references is crucial. 
    ## 5. Strict Compliance
    Adhere to the rules strictly. Non-compliance will result in termination.'''
    "knowledge triplets in the form of (subject, predicate, object). Avoid stopwords.\n"
    "---------------------\n"
    "Example:"
    "Text: Alice is Bob's mother."
    "Triplets:\n(Alice, is mother of, Bob)\n"
    "Text: Philz is a coffee shop founded in Berkeley in 1982.\n"
    "Triplets:\n"
    "(Philz, is, coffee shop)\n"
    "(Philz, founded in, Berkeley)\n"
    "(Philz, founded in, 1982)\n"
    "---------------------\n"
    "Text: {text}\n"
    "Triplets:\n"
)
KG_TRIPLET_EXTRACT_PROMPT = PromptTemplate(
    DEFAULT_KG_TRIPLET_EXTRACT_TMPL,
    prompt_type=PromptType.KNOWLEDGE_TRIPLET_EXTRACT,
)