def build_triple_extraction_prompt(schema, example_dialogue, example_out, case_dialogue):
    return f"""
TASK:
Extract all valid, fully grounded knowledge graph triples from the input text according to the provided ontology schema.

Your objective is to map the dialogue into clean, precise, non-redundant semantic relations with ZERO inferred entities, ZERO missing-value placeholders, and ZERO narrative hallucinations.

STRICT EXTRACTION CONSTRAINTS:

1. ATOMIC NAMED ENTITIES ONLY & NO COMPOSED ROLES (CRITICAL DO NOT):
   - Every Subject and Object MUST be an explicit, standalone atomic entity (e.g., proper nouns, explicit geographic locations) or a factual literal (e.g., numbers, currencies, URLs).
   - DO NOT create composed, possessive, or relative entities (NEVER combine entity names with relational nouns or possessive markers).
   - DO NOT extract abstract, emotional, or non-actionable attributes (e.g., `hasRole` must represent an explicit official, legal, or occupational status, NEVER subjective behavioral characteristics).

2. STRICT ATTRIBUTE, IDENTIFIER & CREDENTIAL BOUNDARIES (CRITICAL DO):
   - `hasEmail` MUST capture explicit email addresses associated with an Entity. NEVER assign email addresses to `website` or `hasPhoneNumber`.
   - `hasCredentialPassword` MUST capture explicit authentication passwords or access tokens associated with an Account entity.
   - `hasAccountIdentifier` MUST capture official account numbers, IBANs, or wallet strings. DO NOT output passwords or random strings as account identifiers.
   - `hasPhoneNumber` MUST be used exclusively for raw phone numbers associated with an Entity. NEVER use phone numbers to infer an entity's physical location (`locatedIn`).
   - `hasPurpose` MUST capture ONLY the direct, operational or commercial objective of a transaction. DO NOT map interpersonal sentiments, emotional expressions, or narrative desires as a transaction purpose.
   - `locatedIn` MUST be a concrete, explicitly named geopolitical entity (e.g., city, nation, or standard administrative area). NEVER extract relative distance descriptions, generic placeholders, or infer locations from communication channels.
   - Keep Objects strictly atomic: NO inline comments or notes inside parentheses or after symbols.

3. ABSOLUTE REJECTION OF MISSING VALUES & PLACEHOLDERS (CRITICAL DO NOT):
   - Extract a triple ONLY if the exact value is explicitly declared in the input text.
   - NEVER generate triples with placeholder values such as "(not provided)", "unknown", "N/A", "null", or primitive type names (e.g., "string", "double"). If an attribute value is missing in the text, DO NOT output a triple for that predicate.

4. ONE TRIPLE PER DISTINCT INTENT & FACTUAL GROUNDING:
   - Extract a relation ONLY if it is directly and unambiguously declared in the text.
   - Do NOT split a single dialogue act into multiple redundant triples by extracting near-synonyms or speculative statements.

5. SYMMETRIC COMMUNICATION & SCHEMA COMPLIANCE:
   - For every communication relation (`communicatesWith`), output BOTH directional triples (e.g., `PersonA | communicatesWith | PersonB` AND `PersonB | communicatesWith | PersonA`).
   - The Subject MUST always be the active initiator of the relation. Strictly comply with domain/range constraints.

ALLOWED ONTOLOGY SCHEMA:
{schema}

FEW-SHOT MAPPING REFERENCE:
EXAMPLE INPUT:
{example_dialogue}

EXAMPLE OUTPUT:
{example_out}

APPLICATION TASK:
Apply the structural logic directly to the target input text. Be factually thorough, preserve distinct domain actions, and aggressively reject non-atomic entities, structural type declarations, missing placeholders, or misplaced identifiers.

TARGET INPUT:
{case_dialogue}

OUTPUT (Provide only the raw triples, one per line using the format: Subject | Predicate | Object):
""".strip()


def build_explanation_prompt(dialogue_text, triple):
    entity, attribute, value = triple

    return f"""
ROLE:
You are an expert semantic analyst. Your job is to draft a precise justification explaining why the source text validates the specific relation between the subject and the object.

CONTEXT SOURCE TEXT:
\"\"\"
{dialogue_text}
\"\"\"

TARGET TRIPLE TO EXPLAIN:
Subject: {entity}
Predicate: {attribute}
Object: {value}

TASK:
Write exactly ONE elegant, natural, and fully completed sentence that explains the underlying reasoning behind this semantic connection. 

HUMAN-LIKE REASONING & FACTUAL PROOF CONSTRAINTS:

1. LITERAL PROOF & DIRECT EVIDENCE (CRITICAL DO):
   - Base your sentence strictly on explicit statements or direct actions declared in the text.
   - CRITICAL DO NOT: DO NOT invent psychological theories, hidden motives, or unmentioned backstories. NEVER use inferential or speculative phrasing that attributes unstated psychological intent. State purely what was explicitly declared or performed.

2. AVOID SEMANTIC TAUTOLOGY:
   - DO NOT simply repeat the technical name of the predicate or entities. Describe the real-world interaction in clear prose.

3. FACTUAL INTEGRITY & COMPLETE SENTENCE:
   - Ensure the active party in your sentence matches the true Subject. End with a definitive period.

4. ABSOLUTE TEXT FORMATTING (STRICT NO MARKDOWN):
   - Output ONLY plain text. Do NOT use markdown syntax (no bolding, italics, quotes, or bullet points). Start directly with the first word.

OUTPUT ONLY THE SINGLE COMPLETED REASONING SENTENCE:
""".strip()

def build_ner_prompt(dialogue_text):
    return f"""
TASK:
Extract ONLY specific Named Entities (People, Organizations, Locations, Accounts, Transactions, Messages) from the chat.
IGNORE all abstract ideas, generic words, URLs, and sentences.

STRICT ONTOLOGY (ONLY THESE ARE ALLOWED):
[Person, Organization, Location, Account, Transaction, Message]

STRICT EXTRACTION RULES:
1. NO CONCEPTS: Do not extract abstract concepts, informational terms, or guidance descriptions (e.g., "Education", "Support", "Guidance", "Journey", "Opportunity", "Details"). These are NOT entities.
2. NO URLS: Completely ignore all http/www links.
3. UNIQUE IDs: You MUST use the format [Class][Number]. 
   - CORRECT: Person1, Organization1, Transaction1
   - INCORRECT: Lucy, Lucy1, Education1
4. ONE CLASS PER NAME: If 'Lucy' is Person1, she cannot be anything else.

OUTPUT FORMAT (ONLY TWO LINES PER ENTITY):
EntityID, rdf:type, Class
EntityID, label, Name

EXAMPLE:
Person1, rdf:type, Person
Person1, label, John Doe
Organization1, rdf:type, Organization
Organization1, label, Google

INPUT CHAT:
{dialogue_text}

OUTPUT:
""".strip()

def build_ner_explanation_prompt(dialogue_text, entity_id, entity_class, entity_name):
    return f"""
ROLE:
You are an explanation generator for Named Entities extracted from a chat history.
Your task is to explain why a specific entity was assigned a specific category (rdf:type).

CONTEXT:
The data below contains a chat history.

CHAT HISTORY:
{dialogue_text}

ENTITY CLASSIFICATION (RDF:TYPE TRIPLE):
Triple: {entity_id} | rdf:type | {entity_class}
Entity Name (Label): {entity_name}

TASK:
Write ONLY ONE concise sentence explaining why the word/phrase "{entity_name}" is categorized as a "{entity_class}" in the context of the chat.

RULES:
- Explain WHY this entity fits its category based on the chat. (For example, explain what action or context in the chat shows that "{entity_name}" acts like a {entity_class}).
- Base the explanation strictly on the provided chat context.
- The explanation MUST explicitly mention the entity name ("{entity_name}").
- Do NOT repeat or summarize the entire chat.
- Do NOT explain the extraction process or mention AI models.
- Output exactly ONE complete sentence ending with a period.
- Maximum ONE sentence.
- Stop immediately after the first sentence.
- Natural language only.

OUTPUT:
""".strip()

def build_minimal_counterfactual_prompt(dialogue_text, triple, original_explanation):
    entity = triple["entity"]
    attribute = triple["attribute"]
    value = triple["value"]

    return f"""<|im_start|>system
You are a strict causal logic validator for knowledge graphs. You output ONLY the final counterfactual sentence. No notes, no markdown formatting, and never start with punctuation marks like colons.<|im_end|>
<|im_start|>user
CONTEXT:
We have a chat history, an extracted knowledge graph triple, and the original reasoning explanation for why that triple was extracted.

CHAT HISTORY:
{dialogue_text}

TRIPLE:
{entity} | {attribute} | {value}

ORIGINAL EXPLANATION:
{original_explanation}

TASK:
Formulate exactly ONE natural, grammatically complete counterfactual explanation sentence.

STRICT CAUSALITY FRAMEWORK:
Identify the exact and minimal text trigger from the chat history.
1. If the attribute represents a dynamic action, request, or transaction, the trigger is the statement where the subject performs that action.
2. If the attribute represents a static property or syntax-specific literal, the trigger is the occurrence of the value string itself.

STRICT CONTENT GENERATION RULES (ANTI-TAUTOLOGY):
1. NO TEMPLATE WORDS: You are STRICTLY FORBIDDEN from reusing generic words like 'Alternative', 'Placeholder', 'Another', 'Different', 'Service', 'Name', 'Value', or 'Object' inside your substitute text. Do not repeat the words used in these instructions.
2. CONTEXT-BASED INFERENCE: You must generate a highly specific, plausible, and realistic substitute value that mirrors the exact data type and domain of '{value}':
   - If '{value}' follows a strict technical syntax (e.g., URL, email, ID) -> Synthesize a new, unique, valid literal adhering to that exact syntax, matching the domain implied by the context.
   - If '{value}' represents a specific concept, intent, action, or object -> Analyze the surrounding context and infer a distinct, logically plausible alternative concept from the same ontological level that is NOT a synonym or antonym.

REQUIRED LOGICAL TEMPLATE:
"If [Insert the EXACT Text Trigger] had not been mentioned, the model would NOT have extracted the relation '{attribute}' between '{entity}' and '{value}', but could have captured [Insert your freshly inferred, highly specific substitute value] instead."

Counterfactual Explanation:<|im_end|>
<|im_start|>assistant\n""".strip()