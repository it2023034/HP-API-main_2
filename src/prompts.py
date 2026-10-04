def build_triple_extraction_prompt(schema, example_dialogue, example_out, case_dialogue):
    return f"""
TASK:
Extract all valid, fully grounded knowledge graph triples from the input text according to the provided ontology schema.

Your objective is to map the dialogue into clean, precise, non-redundant semantic relations with ZERO inferred entities, ZERO missing-value placeholders, and ZERO narrative hallucinations.

STRICT EXTRACTION CONSTRAINTS:

1. ATOMIC NAMED ENTITIES & QUALIFIED INSTANCES (CRITICAL DO NOT):
   - Every Subject and Object MUST be an explicit, standalone atomic entity (e.g., proper nouns, explicit geographic locations) or a factual literal (e.g., numbers, currencies, URLs).
   - NEVER use raw categorical or class labels as standalone Subject or Object entities when a unique identifier or specific contextual sub-type is available. If a unique ID/key exists in the text, use that identifier directly as the entity.
   - DO NOT create composed, possessive, or relative entities (NEVER combine entity names with relational nouns or possessive markers).
   - DO NOT extract abstract, emotional, or non-actionable attributes (e.g., `hasRole` must represent an explicit official, legal, or occupational status, NEVER subjective behavioral characteristics).

2. STRICT ATTRIBUTE, IDENTIFIER & CREDENTIAL BOUNDARIES (CRITICAL DO):
   - `hasEmail` MUST capture explicit email addresses containing '@'. NEVER assign email addresses or handles to `website` or `hasPhoneNumber`.
   - `hasCredentialPassword` MUST capture explicit authentication passwords or access tokens associated with an Account entity.
   - `hasAccountIdentifier` MUST capture official account numbers, IBANs, or wallet strings. DO NOT output passwords or random strings as account identifiers.
   - `hasPhoneNumber` MUST be used exclusively for raw phone numbers associated with an Entity. NEVER use phone numbers to infer an entity's physical location (`locatedIn`).
   - `hasPurpose` MUST capture ONLY direct operational objectives explicitly stated in the text. DO NOT map interpersonal sentiments, narrative desires, or inferred scenarios.
   - EXACT LEXICAL GROUNDING FOR ROLES & PURPOSES: Extracted values for `hasRole` and `hasPurpose` MUST be derived directly from explicit noun phrases in the input text. NEVER infer unstated professions, external backgrounds, or speculative motives.
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

1. LITERAL PROOF & SEARCH HEURISTICS (CRITICAL DO):
   - Perform an internal search for references to "{entity}" and "{value}" across the context text.
   - Base your sentence strictly on explicit statements, communications, or direct actions declared in the text.
   - Evaluate the real-world connection or interaction between "{entity}" and "{value}" rather than asserting its absence.
   - CRITICAL DO NOT: NEVER write that the relation is "not mentioned", "not referenced", or "lacks evidence".
   - CRITICAL DO NOT: DO NOT invent psychological theories, hidden motives, or unmentioned backstories. NEVER use inferential or speculative phrasing that attributes unstated psychological intent. State purely what was explicitly declared or performed.

2. AVOID SEMANTIC TAUTOLOGY:
   - DO NOT simply repeat the technical name of the predicate or entities. Describe the real-world interaction in clear prose.

3. FACTUAL INTEGRITY & COMPLETE SENTENCE:
   - Ensure the active party in your sentence matches the true Subject. End with a definitive period.

4. ABSOLUTE TEXT FORMATTING (STRICT NO MARKDOWN):
   - Output ONLY plain text. Do NOT use markdown syntax (no bolding, italics, quotes, or bullet points). Start directly with the first word.

OUTPUT ONLY THE SINGLE COMPLETED REASONING SENTENCE:
""".strip()

def build_ner_prompt(dialogue_text: str) -> str:
    """Builds prompt for extracting named entities according to strict ontology classes."""
    return f"""
TASK:
Extract ONLY specific Named Entities (People, Organizations, Locations, Accounts, Transactions, Messages) explicitly mentioned in the text.
IGNORE all abstract ideas, generic words, URLs, and sentences.

STRICT ONTOLOGY (ONLY THESE ARE ALLOWED):
[Person, Organization, Location, Account, Transaction, Message]

STRICT EXTRACTION RULES:
1. NO CONCEPTS: Do NOT extract abstract terms, informational words, or general guidance descriptions.
2. NO URLS: Completely ignore all http/www links and email strings.
3. EXACT LABELS: Output the exact, clean entity name or label as declared in the text.

OUTPUT FORMAT (ONLY TWO LINES PER ENTITY):
[Class], rdf:type, Class
[Class], label, Name

EXAMPLE:
Person, rdf:type, Person
Person, label, John Doe
Organization, rdf:type, Organization
Organization, label, Google

INPUT CHAT:
{dialogue_text}

OUTPUT:
""".strip()


def build_ner_explanation_prompt(dialogue_text: str, entity_id: str, entity_class: str, entity_name: str) -> str:
    """Builds prompt for generating single-sentence entity classification reasoning."""
    return f"""
ROLE:
You are an expert semantic analyst explaining Named Entity classification in a Knowledge Graph.

CONTEXT CHAT HISTORY:
\"\"\"
{dialogue_text}
\"\"\"

TARGET CLASSIFICATION:
Entity: {entity_name}
Category: {entity_class}

TASK:
Write exactly ONE concise, fact-based sentence explaining why "{entity_name}" is classified as a "{entity_class}" based on its presence, mentions, or actions in the chat context.

RULES:
- Base the explanation strictly on explicit actions, mentions, or references in the text.
- The explanation MUST explicitly include the entity name "{entity_name}".
- GENERAL REASONING HEURISTICS:
  * First, perform a search for "{entity_name}" across the chat context.
  * If the entity is a direct participant, explain its actions or role.
  * If the entity appears briefly (e.g., in credentials, metadata, system logs, beneficiary names, or brief references), explain its existence based on that specific contextual appearance.
  * Evaluate the semantic function of "{entity_name}" in the context rather than asserting its absence.
- Do NOT invent unstated background facts.
- Output ONLY plain text (NO markdown, bolding, or bullet points).
- Maximum ONE sentence ending with a period.

OUTPUT ONLY THE SINGLE EXPLANATION SENTENCE:
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