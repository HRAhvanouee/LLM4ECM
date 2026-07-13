"""LLM agents and tools for Engineering Change Management AAS conversion.

This module contains three layers:
- a nondeterministic subagent that turns textual feedback into AAS records;
- a deterministic subagent that enriches those records with classifications and AML IDs;
- a main coordinator agent that routes user requests to the correct subagent.
"""

from datetime import datetime
import uuid
import copy
import json
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain.tools import tool
from langchain_core.messages import HumanMessage
from langchain_ollama import ChatOllama
from langgraph.checkpoint.memory import InMemorySaver


BASE_DIR = Path(__file__).parent


def _load_json_file(path: Path) -> Any:
    """Load a UTF-8 JSON file from disk."""
    with open(path, "r", encoding="utf-8") as file:
        return json.load(file)


# ============================================================
# Subagent 1: Nondeterministic Engineering Change
# ============================================================
# This section loads the AAS schema context and defines the prompt/tooling
# for transforming itemized textual engineering feedback into the
# Nondeterministic Engineering Change submodel.

schema_file_path = BASE_DIR / "Submodel_NondeterministicEngineeringChange.json"
schema_item_change_path = BASE_DIR / "Schema_item_nondeterministic.json"

aas_schema_json = json.dumps(_load_json_file(schema_file_path), indent=2)
schema_items_json = json.dumps(_load_json_file(schema_item_change_path), indent=2)


@tool(
    "get_current_datetime",
    description="Returns the current datetime in ISO 8601 format.",
)
def get_current_datetime() -> str:
    """Return the current datetime as an ISO 8601 string."""
    return datetime.now().isoformat()


prompt_nondeterministic_agent = f"""
You are an expert Asset Administration Shell (AAS) engineering change data generator.

Your task is to convert numbered textual engineering feedback into a STRICT AAS-compliant JSON structure.

============================================================
INPUT FORMAT
============================================================

The input is a numbered list of engineering feedback items (e.g., 1-, 2-, 3-...).

Each item may contain one or multiple engineering operations.

============================================================
OUTPUT RULES (HARD CONSTRAINTS)
============================================================

- Output MUST be valid JSON only (no markdown, no comments, no explanation)
- Output MUST follow the provided schema exactly: {aas_schema_json}
- Output MUST contain exactly ONE submodel:
  "nondeterministicEngineeringChange"

- Inside this submodel there MUST be EXACTLY ONE:
  "ChangeRecords" SubmodelElementList (SML)

- The ChangeRecords SML MUST contain a list of value items (one per operation)

- Do NOT create additional submodels
- Do NOT create additional ChangeRecords lists
- Do NOT add any fields not defined in the schema
- If information is missing → use null (never guess)

============================================================
CHANGE EXTRACTION RULES
============================================================

For each numbered feedback item:

1. Parse it into atomic engineering operations
   - One operation = one action (add / delete / update / modify / change / upgrade)

2. If a sentence contains multiple operations:
   - Split into multiple ChangeRecords entries

3. Each ChangeRecord MUST contain:
   - ChangeDescription
   - DateOfRecord

4. Extract only explicit information from the text

============================================================
SUPPORTED OPERATIONS
============================================================

Allowed values for operations:
- add
- delete
- update
- modify
- change
- upgrade

If multiple operations exist in one sentence → split into multiple records.

============================================================
DATE RULE (MANDATORY)
============================================================

- You MUST use the result of get_current_datetime tool
- DateOfRecord MUST be ISO 8601 (xs:dateTime format)
- Same timestamp source for all records in the output

============================================================
OUTPUT STRUCTURE LOGIC
============================================================

Final JSON must follow this hierarchy:

- nondeterministicEngineeringChange
  - ChangeRecords (SML)
    - value: [ list of ChangeRecord items ]

Each item corresponds to exactly one engineering operation.

============================================================
SCHEMA CONSTRAINT
============================================================

Use this schema as the ONLY source of truth:
{aas_schema_json}

Use this field definition guide:
{schema_items_json}

Do not modify schema structure or metadata.

============================================================
EXAMPLE BEHAVIOR
============================================================

Example 1:
Input:
1- Add temperature sensor to Tank1 and upgrade firmware
2- Delete Motor X

Output behavior:
- Record 1: add sensor
- Record 2: upgrade firmware
- Record 3: delete motor

All must be inside ONE ChangeRecords SML.

Example 2:

Input:
 1- Add temperature sensor to module A and upgrade it with next firmware
 2- Update motor speed from 1500 RPM to 1800 RPM and delete pressure valve in subsystem B.

Output behavior:
- Record 1 : Add temperature sensor to module A 
- Record 2 : upgrade temperature sensor with next firmware
- Record 3 : Update motor speed from 1500 RPM to 1800 RPM
- Record 4 : delete pressure valve in subsystem B.

All must be inside ONE ChangeRecords SML.

============================================================
FINAL INSTRUCTION
============================================================

Return ONLY the final JSON object matching the schema.
"""


# ============================================================
# Subagent 2: Deterministic Engineering Change
# ============================================================
# This section keeps the deterministic mapping helpers and prompt exactly in
# the shape expected by the existing agent wiring.

schema_file_path_deterministic = BASE_DIR / "Submodel_DeterministicEngineeringChange.json"
schema_item_change_deterministic_path = BASE_DIR / "Schema_item_deterministic.json"
schema_new_technical_data_path = BASE_DIR / "Submodel_New_TechnicalData_InternalElement.json"

aas_deterministic_schema_json = _load_json_file(schema_file_path_deterministic)
schema_items_deterministic_json = _load_json_file(schema_item_change_deterministic_path)
schema_new_technical_data_json = _load_json_file(schema_new_technical_data_path)
# Keyword maps provide deterministic fallback classification for VDMA 24903 fields.
# They are intentionally simple and transparent because their outputs go into
# fixed AAS classification properties.
ITEM_CATEGORY_MAP = {
    "ACEL": [
        "electronics",
        "semiconductor",
        "pcb",
        "chip",
        "ic",
        "sensor",
        "electronic",
    ],
    "DACE": ["data", "certificate", "database", "key", "encryption", "digital"],
    "SERV": ["service", "maintenance", "logistics", "cleaning", "inspection"],
    "DOCU": ["document", "manual", "datasheet", "instruction", "specification"],
    "ELME": ["relay", "switch", "electromechanical", "contactor"],
    "FLUI": ["oil", "gas", "fluid", "hydraulic oil", "fuel"],
    "AUXM": ["chemical", "cleaning", "auxiliary", "consumable"],
    "HYDR": ["hydraulic", "pump", "cylinder", "hose"],
    "MECH": ["gear", "shaft", "mechanical", "bolt", "screw"],
    "MULT": ["multiple", "various categories", "system level"],
    "PAEL": ["passive", "resistor", "capacitor", "inductor"],
    "PNEU": ["pneumatic", "valve", "air system"],
    "RAWM": ["raw material", "metal", "granulate", "textile"],
    "SWFW": ["software", "firmware", "code", "application"],
    "OTHR": ["other", "misc"],
    "CCBL": ["cable", "connector", "wiring"],
    "ASSY": ["assembly", "system assembly", "subsystem"],
}

REASON_MAP = {
    "PDN": ["discontinue", "end of life", "abkündigung"],
    "MANAQ": ["acquisition", "merge", "transfer manufacturer"],
    "ALERT": ["warning", "alert", "issue", "degradation"],
    "SOFTW": ["software change", "firmware update"],
    "LABEL": ["label", "marking", "packaging label"],
    "CHARA": ["parameter", "characteristic", "attribute change"],
    "DOCUM": ["documentation update", "docs change"],
    "NRND": ["not recommended", "obsolete recommendation"],
    "FIT": ["fit", "dimension mismatch", "tolerance"],
    "FORM": ["appearance", "color", "shape", "surface"],
    "FUNCT": ["function", "performance", "behavior change"],
    "INSOL": ["bankruptcy", "insolvent"],
    "CORR": ["correction", "fix documentation"],
    "SHIP": ["delivery", "shipping", "logistics change"],
    "MATER": ["material change", "substance change"],
    "PRODS": ["production start", "new production"],
    "PPROC": ["production process", "manufacturing change"],
    "PSITE": ["production site", "factory change"],
    "CANCN": ["cancel pcN", "revoke pcn"],
    "CANDN": ["resume production", "cancel pdn"],
    "RECA": ["recall", "withdraw"],
    "TESTP": ["test process", "testing change"],
    "TESTS": ["test site", "testing location"],
    "ORCOD": ["type code change", "ordering code"],
}


@tool
def get_item_categorie(ChangeDescription: str) -> str:
    """Classify VDMA 24903 item category from engineering change description."""
    text = ChangeDescription.lower()

    for code, keywords in ITEM_CATEGORY_MAP.items():
        for kw in keywords:
            if kw in text:
                return code

    return "OTHR"


@tool
def get_reason_change(ChangeDescription: str) -> str:
    """Classify VDMA 24903 reason code from engineering change description."""
    text = ChangeDescription.lower()

    for code, keywords in REASON_MAP.items():
        for kw in keywords:
            if kw in text:
                return code

    return "CHARA"


@tool
def get_change_type(ChangeDescription: str) -> str:
    """Determine deterministic AAS change type from a change description."""
    text = ChangeDescription.lower()

    if any(w in text for w in ["remove", "delete", "drop", "eliminate"]):
        return "DELETE"

    if any(w in text for w in ["replace all", "overwrite", "fully replace"]):
        return "PUT"

    if any(w in text for w in ["add", "create", "insert", "new"]):
        return "POST"

    if any(w in text for w in ["update", "modify", "change", "adjust", "fix"]):
        return "PATCH"

    return "PATCH"


@tool
def classify_deterministic_change(ChangeDescription: str) -> str:
    """Return deterministic classification values for one change description."""
    text = ChangeDescription.lower()

    change_type = "PATCH"
    if any(w in text for w in ["remove", "delete", "drop", "eliminate"]):
        change_type = "DELETE"
    elif any(w in text for w in ["replace all", "overwrite", "fully replace"]):
        change_type = "PUT"
    elif any(w in text for w in ["add", "create", "insert", "new"]):
        change_type = "POST"

    reason_of_change = "CHARA"
    for code, keywords in REASON_MAP.items():
        if any(kw in text for kw in keywords):
            reason_of_change = code
            break

    item_category = "OTHR"
    for code, keywords in ITEM_CATEGORY_MAP.items():
        if any(kw in text for kw in keywords):
            item_category = code
            break

    return json.dumps(
        {
            "ChangeType": change_type,
            "ReasonOfChange": reason_of_change,
            "ItemCategory": item_category,
        }
    )


def _normalize_match_text(value: str) -> str:
    """Normalize free text so AML names and descriptions can be compared."""
    return re.sub(r"[^a-z0-9]+", " ", value.lower()).strip()


def set_aml_internal_elements(elements) -> None:
    """Persist imported AML InternalElement records for deterministic lookup."""
    aml_internal_elements_path.write_text(
        json.dumps(elements, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )


def get_aml_internal_elements():
    """Return imported AML InternalElement records from the latest uploaded AML file."""
    if not aml_internal_elements_path.exists():
        return []

    try:
        return json.loads(aml_internal_elements_path.read_text(encoding="utf-8") or "[]")
    except (OSError, json.JSONDecodeError):
        return []


@tool
def get_aml_element(ChangeDescription: str) -> str:
    """Return the AML InternalElement ID affected by a change description.

    The UI stores imported AML InternalElements as name/id records. This tool
    scores each record against the ChangeDescription and returns the best ID for
    the deterministic AMLElement FragmentReference.
    """
    description = _normalize_match_text(ChangeDescription)
    if not description:
        return ""

    description_tokens = set(description.split())
    best_element = None
    best_score = 0

    for element in get_aml_internal_elements():
        name = str(element.get("name", ""))
        element_id = str(element.get("id", ""))
        normalized_name = _normalize_match_text(name)
        normalized_id = _normalize_match_text(element_id)
        if not normalized_name:
            continue

        # Exact phrase matches are strongest; shared tokens handle partial names.
        name_tokens = set(normalized_name.split())
        score = len(description_tokens & name_tokens)
        if normalized_name in description:
            score += 100
        if normalized_id and normalized_id in description:
            score += 80

        if score > best_score:
            best_score = score
            best_element = element

    if not best_element or best_score <= 0:
        return ""

    return str(best_element.get("id") or best_element.get("name") or "")


TOOLS = [
    get_change_type,
    get_reason_change,
    get_item_categorie,
    classify_deterministic_change,
    get_aml_element,
]

# The deterministic agent receives all classification and AML lookup tools.
DETERMINISTIC_TOOLS = TOOLS

prompt_deterministic = """

You are a STRICT DETERMINISTIC ENGINEERING CHANGE TRANSFORMATION ENGINE.

You convert a NondeterministicEngineeringChange AAS JSON into a DeterministicEngineeringChange AAS JSON.

---

## INPUT

You receive:
- One NondeterministicEngineeringChange Submodel JSON
- Contains ChangeRecords list

Each ChangeRecord contains:
- ChangeDescription (natural language)
- DateOfRecord (ISO date string)

---

## TASK

For EACH ChangeRecord:

Create exactly ONE corresponding Deterministic ChangeRecord.

Maintain order exactly.

---

## TOOL USAGE (MANDATORY)

For each ChangeDescription, you MUST call exactly once each:

- classify_deterministic_change(ChangeDescription)
- get_aml_element(ChangeDescription)

classify_deterministic_change returns ChangeType, ReasonOfChange, and ItemCategory together.
get_aml_element returns the AML InternalElement ID for the affected part, or an empty string if no imported AML element matches.
You MUST NOT guess values.
You MUST NOT proceed without tool output.

---

## OUTPUT MAPPING

Each output ChangeRecord must include:

- ChangeType → ChangeType from classify_deterministic_change
- ReasonOfChange → ReasonOfChange from classify_deterministic_change
- ItemCategory → ItemCategory from classify_deterministic_change
- DateOfRecord → copied from input
- AMLElement.value.keys[type=FragmentReference].value → ID from get_aml_element

---

## STRICT RULES

- Output must match deterministicEngineeringChange schema exactly
- Preserve ordering
- No missing items
- No extra items
- No explanations
- No markdown
- No text outside JSON

---

## FAILURE CONDITIONS

Use tools when needed; if unavailable, estimate conservatively.

"""

# ============================================================
# Subagent 3: AML UPDATE Agent
# ============================================================
# This section uses the deterministicEngineerinChange submodel
# and TechnicalData submodel to update Aml Molel




def _json_documents_from_text(text: Any) -> list[Any]:
    """Parse one or more adjacent JSON documents from model/UI output."""
    if isinstance(text, (dict, list)):
        return [text]

    source = str(text or "").strip()
    if not source:
        return []

    decoder = json.JSONDecoder()
    docs = []
    index = 0
    while index < len(source):
        while index < len(source) and source[index].isspace():
            index += 1
        if index >= len(source):
            break
        try:
            doc, end = decoder.raw_decode(source, index)
        except json.JSONDecodeError:
            normalized = normalize_json_output(source[index:])
            try:
                docs.append(json.loads(normalized))
            except json.JSONDecodeError:
                pass
            break
        docs.append(doc)
        index = end
    return docs


def _local_xml_name(tag: str) -> str:
    return tag.rsplit("}", 1)[-1] if "}" in tag else tag


def _xml_namespace(tag: str) -> str:
    return tag[1:].split("}", 1)[0] if tag.startswith("{") else ""


def _xml_tag(namespace: str, local_name: str) -> str:
    return f"{{{namespace}}}{local_name}" if namespace else local_name


def _parse_xml_fragment(xml_text: str) -> ET.Element | None:
    try:
        return ET.fromstring(str(xml_text).strip())
    except ET.ParseError:
        return None


def _xml_to_string(element: ET.Element) -> str:
    return ET.tostring(element, encoding="unicode")


def _parse_xml_document(xml_text: str) -> ET.Element | None:
    try:
        return ET.fromstring(str(xml_text).strip())
    except ET.ParseError:
        return None


def _element_identity_candidates(aml_element: Any) -> set[str]:
    candidates: set[str] = set()
    text = str(aml_element or "").strip()
    if text:
        candidates.add(text)

    parsed = _parse_xml_fragment(text)
    if parsed is not None:
        for key in ("ID", "Id", "id", "Name"):
            value = parsed.attrib.get(key, "").strip()
            if value:
                candidates.add(value)

    for key in ("ID", "Id", "id", "Name", "value"):
        for match in re.finditer(
            rf'"{key}"\s*:\s*"([^"]+)"|{key}\s*=\s*["\']([^"\']+)["\']',
            text,
            flags=re.IGNORECASE,
        ):
            value = (match.group(1) or match.group(2) or "").strip()
            if value:
                candidates.add(value)

    return {candidate.lower() for candidate in candidates if candidate}


def _find_aml_internal_element(root: ET.Element, aml_element: Any) -> ET.Element | None:
    candidates = _element_identity_candidates(aml_element)
    if not candidates:
        return None

    for element in root.iter():
        if _local_xml_name(element.tag) != "InternalElement":
            continue
        values = {
            element.attrib.get("ID", ""),
            element.attrib.get("Id", ""),
            element.attrib.get("id", ""),
            element.attrib.get("Name", ""),
        }
        if any(value.strip().lower() in candidates for value in values if value):
            return element
    return None


def _parent_map(root: ET.Element) -> dict[ET.Element, ET.Element]:
    return {child: parent for parent in root.iter() for child in list(parent)}


def _remove_relations_for_element(root: ET.Element, element: ET.Element) -> None:
    identifiers = {
        element.attrib.get("ID", ""),
        element.attrib.get("Id", ""),
        element.attrib.get("id", ""),
        element.attrib.get("Name", ""),
    }
    identifiers = {value for value in identifiers if value}
    if not identifiers:
        return

    parents = _parent_map(root)
    for candidate in list(root.iter()):
        if candidate is element:
            continue
        if _local_xml_name(candidate.tag) not in {"InternalLink", "ExternalInterface", "RoleRequirements"}:
            continue
        relation_text = " ".join(
            [str(value) for value in candidate.attrib.values()]
            + [candidate.text or ""]
        )
        if any(identifier in relation_text for identifier in identifiers):
            parent = parents.get(candidate)
            if parent is not None:
                parent.remove(candidate)


def _replace_element_in_parent(root: ET.Element, old: ET.Element, new: ET.Element) -> None:
    parent = _parent_map(root).get(old)
    if parent is None:
        return
    children = list(parent)
    index = children.index(old)
    parent.remove(old)
    parent.insert(index, new)


def _find_direct_xml_child(element: ET.Element, local_name: str, name: str | None = None):
    for child in list(element):
        if _local_xml_name(child.tag) != local_name:
            continue
        if name is None or child.attrib.get("Name", "").lower() == name.lower():
            return child
    return None


def _technical_properties_from_text(technical_data: Any) -> list[dict[str, str]]:
    properties = []
    for doc in _json_documents_from_text(technical_data):
        technical_properties = _find_element(doc, "TechnicalProperties")
        if not technical_properties:
            continue
        for item in technical_properties.get("value", []):
            if not isinstance(item, dict) or item.get("modelType") != "Property":
                continue
            properties.append(
                {
                    "parameter": str(item.get("idShort") or "Parameter"),
                    "value": str(item.get("value") or ""),
                    "value_type": str(item.get("valueType") or "xs:string"),
                }
            )
    return properties


def _ensure_aml_attribute(element: ET.Element, parameter: str, value_type: str) -> ET.Element:
    namespace = _xml_namespace(element.tag)
    attribute = _find_direct_xml_child(element, "Attribute", parameter)
    if attribute is None:
        attribute = ET.Element(_xml_tag(namespace, "Attribute"))
        attribute.set("Name", parameter)
        element.insert(0, attribute)
    attribute.set("AttributeDataType", value_type or attribute.attrib.get("AttributeDataType", "xs:string"))
    attribute.set("LLM4ECMChange", "updated")
    return attribute


def _set_aml_attribute_value(attribute: ET.Element, value: str) -> None:
    namespace = _xml_namespace(attribute.tag)
    value_element = _find_direct_xml_child(attribute, "Value")
    if value_element is None:
        value_element = ET.SubElement(attribute, _xml_tag(namespace, "Value"))
    value_element.text = value


def _apply_technical_properties_to_element(
    element: ET.Element,
    properties: list[dict[str, str]],
    change_type: str,
) -> ET.Element:
    element.set("LLM4ECMChange", change_type)
    for prop in properties:
        attribute = _ensure_aml_attribute(
            element,
            prop["parameter"],
            prop.get("value_type", "xs:string"),
        )
        attribute.set("LLM4ECMChange", change_type)
        _set_aml_attribute_value(attribute, prop.get("value", ""))
    return element


def _name_from_change_description(change_description: str, fallback: str) -> str:
    match = re.search(
        r"\bnamed\s+[\"']?(?P<name>[A-Za-z0-9_. -]+)",
        str(change_description),
        flags=re.IGNORECASE,
    )
    if match:
        return match.group("name").strip(" .'\"") or fallback
    return fallback


@tool
def operation_delete(ChangeDescription: str, AML_model: str, AML_element: str, TechnicalData: str = "") -> str:
    """DELETE: remove the referenced AML element and relations from the AML model."""
    root = _parse_xml_document(AML_model)
    if root is None:
        return AML_model

    element = _find_aml_internal_element(root, AML_element)
    if element is None:
        return _xml_to_string(root)

    _remove_relations_for_element(root, element)
    parent = _parent_map(root).get(element)
    if parent is not None:
        parent.remove(element)
    root.set("LLM4ECMChange", "DELETE")
    root.set("LLM4ECMAction", "removed-referenced-element-and-relations")
    return _xml_to_string(root)


@tool
def operation_patch(ChangeDescription: str, AML_model: str, AML_element: str, TechnicalData: str) -> str:
    """PATCH: partially update existing AML attributes from TechnicalDataChanges."""
    root = _parse_xml_document(AML_model)
    if root is None:
        return AML_model

    element = _find_aml_internal_element(root, AML_element)
    if element is None:
        return _xml_to_string(root)

    properties = _technical_properties_from_text(TechnicalData)
    _apply_technical_properties_to_element(element, properties, "PATCH")
    root.set("LLM4ECMChange", "PATCH")
    return _xml_to_string(root)


@tool
def operation_put(ChangeDescription: str, AML_model: str, AML_element: str, TechnicalData: str) -> str:
    """PUT: fully replace the referenced AML element with a new representation and relations."""
    root = _parse_xml_document(AML_model)
    if root is None:
        return AML_model

    old_element = _find_aml_internal_element(root, AML_element)
    if old_element is None:
        return _xml_to_string(root)

    namespace = _xml_namespace(old_element.tag)
    new_element = ET.Element(_xml_tag(namespace, "InternalElement"))
    new_element.set("Name", old_element.attrib.get("Name", _name_from_change_description(ChangeDescription, "InternalElement")))
    new_element.set("ID", old_element.attrib.get("ID") or old_element.attrib.get("Id") or old_element.attrib.get("id") or str(uuid.uuid4()))
    if old_element.attrib.get("RefBaseSystemUnitPath"):
        new_element.set("RefBaseSystemUnitPath", old_element.attrib["RefBaseSystemUnitPath"])
    new_element.set("LLM4ECMChange", "PUT")
    new_element.set("LLM4ECMAction", "fully-replaced-element-and-relations")

    properties = _technical_properties_from_text(TechnicalData)
    _apply_technical_properties_to_element(new_element, properties, "PUT")
    _remove_relations_for_element(root, old_element)
    _replace_element_in_parent(root, old_element, new_element)
    root.set("LLM4ECMChange", "PUT")
    return _xml_to_string(root)


@tool
def operation_post(ChangeDescription: str, AML_model: str, AML_element: str, TechnicalData: str) -> str:
    """POST: create and add new AML elements with attributes and relations."""
    root = _parse_xml_document(AML_model)
    if root is None:
        return AML_model

    parent_element = _find_aml_internal_element(root, AML_element) or root
    properties = _technical_properties_from_text(TechnicalData)
    description = str(ChangeDescription or "")

    if re.search(r"\battribute\b", description, flags=re.IGNORECASE):
        _apply_technical_properties_to_element(parent_element, properties, "POST")
        parent_element.set("LLM4ECMAction", "created-attributes")
    else:
        namespace = _xml_namespace(parent_element.tag)
        child = ET.Element(_xml_tag(namespace, "InternalElement"))
        child.set("Name", _name_from_change_description(description, "NewInternalElement"))
        child.set("ID", str(uuid.uuid4()))
        child.set("LLM4ECMChange", "POST")
        child.set("LLM4ECMAction", "created-element-with-attributes-and-relations")
        _apply_technical_properties_to_element(child, properties, "POST")
        parent_element.append(child)
        parent_element.set("LLM4ECMChange", "POST")

    root.set("LLM4ECMChange", "POST")
    return _xml_to_string(root)


@tool
def operation_apply_aml_changes(deterministic_and_technical_json: Any, AML_model: str) -> str:
    """Apply all deterministic/TechnicalData changes to AML XML in one optimized batch."""
    return update_aml_model_direct(deterministic_and_technical_json, AML_model)


aml_TOOLS = [
    operation_apply_aml_changes,
    operation_delete,
    operation_post,
    operation_patch,
    operation_put,
]


prompt_aml = """
You are a STRICT AML UPDATE ENGINE.

Your only task is to apply deterministicEngineeringChange AAS JSON plus referenced TechnicalData AAS JSON to an imported AML XML model.

INPUTS YOU RECEIVE
1. deterministicEngineeringChange AAS JSON
   - Contains ChangeRecords.
   - Each ChangeRecord contains ChangeType: DELETE, PATCH, PUT, or POST.
   - ItemOfChange contains AMLElement, TechnicalDataChanges, and AMLRelationsList references.
2. TechnicalData AAS JSON documents
   - Contain TechnicalProperties with parameter, value, and valueType.
   - Use the TechnicalDataChanges reference to choose the matching TechnicalData document.
3. Imported AML XML model
   - This is the current model shown in the Updated AML Model UI box.

MANDATORY TOOL USAGE
For speed, call operation_apply_aml_changes exactly once.

Pass:
- deterministic_and_technical_json: the complete deterministicEngineeringChange and TechnicalData JSON input
- AML_model: the complete imported AML XML model

Do not call operation_delete, operation_patch, operation_put, or operation_post yourself unless operation_apply_aml_changes is unavailable.
The batch tool applies each ChangeRecord in order and internally chooses the correct operation according to ChangeType.

OPERATION MEANING
- DELETE removes the referenced element and its relations from the AML model.
- PATCH partially updates existing attributes defined by TechnicalDataChanges.
- PUT fully replaces the existing element with a new representation and new relations.
- POST creates and adds new elements with attributes and relations.

OUTPUT RULES
Return valid JSON only, no markdown and no explanation.
The JSON must have exactly this shape:
{
  "updated_aml_model": "<complete updated AML XML string>"
}

The updated XML must include the LLM4ECMChange/LLM4ECMAction markers returned by the tools so the UI can highlight updates.
Do not invent AML elements, attributes, IDs, or relation targets that are not provided by the change records, AML model, or TechnicalData.
"""


# ============================================================
# LLM runtime state
# ============================================================
# The UI imports these helpers and reads the last generated JSON from memory
# or from disk. The checkpointers require callers to pass thread_id in config.

DEFAULT_MODEL = "gemma4:latest"

main_agent_checkpointer = InMemorySaver()
nondeterministic_agent_checkpointer = InMemorySaver()
deterministic_agent_checkpointer = InMemorySaver()
aml_agent_checkpointer = InMemorySaver()
aml_agent_cache: dict[str, Any] = {}

nondeterministic_agent_call_count = 0
nondeterministic_submodel_json_path = (
    BASE_DIR / "NondeterministicEngineeringChange_latest.json"
)
last_nondeterministic_submodel_json = ""
deterministic_agent_call_count = 0
deterministic_submodel_json_path = (
    BASE_DIR / "DeterministicEngineeringChange_latest.json"
)
last_deterministic_submodel_json = ""
aml_internal_elements_path = BASE_DIR / "AML_InternalElements_latest.json"


def reset_agent_short_memory():
    """Clear all agent checkpoint memory for a fresh UI session."""
    global main_agent_checkpointer
    global nondeterministic_agent_checkpointer
    global deterministic_agent_checkpointer
    global aml_agent_checkpointer
    global aml_agent_cache

    main_agent_checkpointer = InMemorySaver()
    nondeterministic_agent_checkpointer = InMemorySaver()
    deterministic_agent_checkpointer = InMemorySaver()
    aml_agent_checkpointer = InMemorySaver()
    aml_agent_cache = {}

# ============================================================
# JSON output helpers
# ============================================================


def _strip_to_json_object(text: str) -> str:
    """Keep only the outer JSON object when model output has stray suffix text."""
    start = text.find("{")
    end = text.rfind("}")
    if start == -1 or end == -1 or end < start:
        return text.strip()
    return text[start:end + 1].strip()


def _repair_common_json_output(text: str) -> str:
    """Repair common LLM JSON typos without changing valid JSON content."""
    repaired = _strip_to_json_object(text)
    value_pattern = r'("(?:\\.|[^"\\])*"|-?\d+(?:\.\d+)?(?:[eE][+-]?\d+)?|true|false|null)\s*:\s*([,}\]])'
    previous = None
    while previous != repaired:
        previous = repaired
        repaired = re.sub(value_pattern, r'\1\2', repaired)
    return repaired


def normalize_json_output(content: str) -> str:
    """Remove optional markdown fences, repair common typos, and pretty-print JSON."""
    text = content.strip()

    if text.startswith("```"):
        lines = text.splitlines()
        if lines and lines[0].strip().startswith("```"):
            lines = lines[1:]
        if lines and lines[-1].strip() == "```":
            lines = lines[:-1]
        text = "\n".join(lines).strip()

    for candidate in (text, _repair_common_json_output(text)):
        try:
            return json.dumps(json.loads(candidate), indent=2, ensure_ascii=False)
        except json.JSONDecodeError:
            continue
    return text

def _set_nested_element_value(root: dict[str, Any], id_short: str, value: Any) -> None:
    """Set the first nested AAS element with the given idShort."""
    element = _find_element(root, id_short)
    if element is not None:
        element["value"] = value


def _set_aml_fragment_reference(record: dict[str, Any], aml_element_id: str) -> None:
    aml_element = _find_element(record, "AMLElement")
    if not aml_element:
        return

    reference_value = aml_element.setdefault(
        "value",
        {"type": "ModelReference", "keys": []},
    )
    reference_value["type"] = "ModelReference"
    keys = reference_value.setdefault("keys", [])
    while len(keys) < 3:
        keys.append({"type": "FragmentReference", "value": ""})
    keys[2]["type"] = "FragmentReference"
    keys[2]["value"] = aml_element_id


def _tool_result_json(tool_object, change_description: str) -> dict[str, Any]:
    try:
        return json.loads(_invoke_string_tool(tool_object, change_description))
    except json.JSONDecodeError:
        return {}


def build_deterministic_submodel_direct(nondeterministic_json: Any) -> str:
    """Build deterministic JSON without LLM tool binding for Ollama models that lack tools."""
    records = extract_nondeterministic_records(nondeterministic_json) or []
    deterministic_submodel = copy.deepcopy(aas_deterministic_schema_json)
    change_records = _find_element(deterministic_submodel, "ChangeRecords")
    if change_records is None:
        return json.dumps(deterministic_submodel, indent=2, ensure_ascii=False)

    change_records["value"] = []
    template_record = schema_items_deterministic_json[0]

    for record in records:
        change_description = record.get("ChangeDescription", "")
        classification = _tool_result_json(
            classify_deterministic_change,
            change_description,
        )
        aml_element_id = _invoke_string_tool(get_aml_element, change_description)

        deterministic_record = copy.deepcopy(template_record)
        _set_nested_element_value(
            deterministic_record,
            "ChangeType",
            classification.get("ChangeType", "PATCH"),
        )
        _set_nested_element_value(
            deterministic_record,
            "DateOfRecord",
            record.get("DateOfRecord", ""),
        )
        _set_nested_element_value(
            deterministic_record,
            "ReasonId",
            classification.get("ReasonOfChange", "CHARA"),
        )
        _set_nested_element_value(
            deterministic_record,
            "ItemCategory",
            classification.get("ItemCategory", "OTHR"),
        )
        _set_aml_fragment_reference(deterministic_record, aml_element_id)
        change_records["value"].append(deterministic_record)

    deterministic_json = json.dumps(
        deterministic_submodel,
        indent=2,
        ensure_ascii=False,
    )
    displayed_deterministic_json = append_new_technical_data_json(
        deterministic_json,
        nondeterministic_json,
    )

    global deterministic_agent_call_count, last_deterministic_submodel_json
    deterministic_agent_call_count += 1
    last_deterministic_submodel_json = displayed_deterministic_json
    deterministic_submodel_json_path.write_text(
        displayed_deterministic_json,
        encoding="utf-8",
    )

    return displayed_deterministic_json


def convert_feedback_to_nondeterministic_direct(model_name: str, feedback_text: str) -> str:
    """Convert feedback without tool binding by injecting the timestamp into the prompt."""
    timestamp = datetime.now().isoformat()
    direct_prompt = prompt_nondeterministic_agent.replace(
        "- You MUST use the result of get_current_datetime tool\n"
        "- DateOfRecord MUST be ISO 8601 (xs:dateTime format)\n"
        "- Same timestamp source for all records in the output",
        "- DateOfRecord MUST be ISO 8601 (xs:dateTime format)\n"
        f"- Use this exact DateOfRecord value for all records: {timestamp}",
    )
    selected_model = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0,
    )
    response = selected_model.invoke(
        [
            HumanMessage(
                content=(
                    f"{direct_prompt}\n\n"
                    "Convert the following itemized feedback. Use only the actual "
                    "feedback items below.\n\n"
                    f"{feedback_text}"
                )
            )
        ]
    )
    submodel_json = normalize_json_output(response.content)

    global nondeterministic_agent_call_count, last_nondeterministic_submodel_json
    nondeterministic_agent_call_count += 1
    last_nondeterministic_submodel_json = submodel_json
    nondeterministic_submodel_json_path.write_text(submodel_json, encoding="utf-8")

    return submodel_json


def get_nondeterministic_agent_call_count():
    """Return how many times the nondeterministic subagent has been called."""
    return nondeterministic_agent_call_count


def get_last_nondeterministic_submodel_json():
    """Return the latest nondeterministic submodel JSON for the UI."""
    if last_nondeterministic_submodel_json:
        return last_nondeterministic_submodel_json

    if not nondeterministic_submodel_json_path.exists():
        return ""

    try:
        return nondeterministic_submodel_json_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def get_deterministic_agent_call_count():
    """Return how many times the deterministic subagent has been called."""
    return deterministic_agent_call_count


def get_last_deterministic_submodel_json():
    """Return the latest deterministic submodel JSON for the UI."""
    if last_deterministic_submodel_json:
        return last_deterministic_submodel_json

    if not deterministic_submodel_json_path.exists():
        return ""

    try:
        return deterministic_submodel_json_path.read_text(encoding="utf-8")
    except OSError:
        return ""


def reset_latest_nondeterministic_submodel_json():
    """Clear the latest nondeterministic JSON shown by the UI."""
    global last_nondeterministic_submodel_json
    last_nondeterministic_submodel_json = ""

    try:
        nondeterministic_submodel_json_path.write_text("", encoding="utf-8")
    except OSError:
        pass


def reset_latest_deterministic_submodel_json():
    """Clear the latest deterministic JSON shown by the UI."""
    global last_deterministic_submodel_json
    last_deterministic_submodel_json = ""

    try:
        deterministic_submodel_json_path.write_text("", encoding="utf-8")
    except OSError:
        pass


def reset_latest_submodel_jsons():
    """Clear both latest submodel JSON outputs shown by the UI."""
    reset_latest_nondeterministic_submodel_json()
    reset_latest_deterministic_submodel_json()


def _iter_elements(element):
    """Yield nested AAS submodel elements from dict/list structures."""
    if isinstance(element, dict):
        yield element
        for key in ("submodelElements", "value"):
            value = element.get(key)
            if isinstance(value, (list, dict)):
                yield from _iter_elements(value)
    elif isinstance(element, list):
        for item in element:
            yield from _iter_elements(item)


def _find_element(element, id_short: str):
    """Find the first nested AAS element with a matching idShort."""
    for candidate in _iter_elements(element):
        if candidate.get("idShort") == id_short:
            return candidate

    return None


def _parse_json_object(json_text):
    """Accept either an already-parsed object or a JSON string."""
    if isinstance(json_text, dict):
        return json_text

    normalized = normalize_json_output(str(json_text))
    return json.loads(normalized)


def extract_nondeterministic_records(nondeterministic_json):
    """Extract the compact record list that the deterministic LLM agent needs."""
    try:
        submodel = _parse_json_object(nondeterministic_json)
    except (TypeError, json.JSONDecodeError):
        return None

    change_records = _find_element(submodel, "ChangeRecords")
    if not change_records:
        return None

    records = []
    for record in change_records.get("value", []):
        description_element = _find_element(record, "ChangeDescription")
        date_element = _find_element(record, "DateOfRecord")
        description = description_element.get("value", "") if description_element else ""
        date_of_record = date_element.get("value", "") if date_element else ""

        if description:
            records.append(
                {
                    "ChangeDescription": description,
                    "DateOfRecord": date_of_record,
                }
            )

    return records


def _invoke_string_tool(tool_object, change_description: str) -> str:
    """Call a LangChain tool that accepts ChangeDescription."""
    result = tool_object.invoke({"ChangeDescription": change_description})
    return "" if result is None else str(result)


def _extract_changed_technical_property(change_description: str) -> dict[str, str]:
    """Extract the changed parameter name and new value from free text."""
    text = " ".join(str(change_description).split())
    named_match = re.search(
        r"\b(?:add|create|insert)\s+(?:a\s+)?(?:new\s+)?(?P<parameter>[A-Za-z0-9_/#(). -]+?)\s+named\s+[\"']?(?P<value>[^\"'.;,\n]+)",
        text,
        flags=re.IGNORECASE,
    )
    if named_match:
        parameter = named_match.group("parameter").strip(" -:")
        value = named_match.group("value").strip(" -:")
        if parameter and value:
            return {"parameter": f"{parameter.title()} Name", "value": value}

    patterns = [
        r"\b(?:change|modify|update|adjust|set|replace)\s+(?:the\s+)?(?P<parameter>[A-Za-z0-9_/#(). -]+?)\s+of\s+[A-Za-z0-9_/#(). -]+?\s+(?:from\s+.+?\s+)?(?:to|as|=)\s+(?P<value>[^.;,\n]+)",
        r"\b(?:change|modify|update|adjust|set|replace)\s+(?:the\s+)?(?P<parameter>[A-Za-z0-9_/#(). -]+?)\s+(?:from\s+.+?\s+)?(?:to|as|=)\s+(?P<value>[^.;,\n]+)",
        r"\b(?:increase|decrease)\s+(?:the\s+)?(?P<parameter>[A-Za-z0-9_/#(). -]+?)\s+(?:from\s+.+?\s+)?to\s+(?P<value>[^.;,\n]+)",
        r"\b(?:add|create|insert)\s+(?:new\s+)?(?P<parameter>[A-Za-z0-9_/#(). -]+?)\s+(?:with\s+value|value|as|=)\s+(?P<value>[^.;,\n]+)",
        r"\b(?P<parameter>[A-Za-z0-9_/#(). -]+?)\s+(?:is|shall be|should be|must be)\s+(?:changed|modified|updated|adjusted|set)\s+to\s+(?P<value>[^.;,\n]+)",
        r"\b(?P<parameter>[A-Za-z0-9_/#(). -]+?)\s*[:=]\s*(?P<value>[^.;,\n]+)",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, flags=re.IGNORECASE)
        if match:
            parameter = match.group("parameter").strip(" -:")
            value = match.group("value").strip(" -:")
            value = re.sub(
                r"\s+(?:for|on|in)\s+(?:the\s+)?[A-Za-z][A-Za-z0-9_ -]*$",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip(" -:")
            if parameter and value:
                if parameter.islower():
                    parameter = parameter.title()
                return {"parameter": parameter, "value": value}

    return {"parameter": "Parameter", "value": ""}


def _infer_aas_value_type(value: str) -> str:
    """Keep numeric values numeric-looking in AAS while preserving units as text."""
    stripped = str(value).strip()
    if re.fullmatch(r"[-+]?\d+", stripped):
        return "xs:integer"
    if re.fullmatch(r"[-+]?(?:\d+\.\d*|\d*\.\d+)", stripped):
        return "xs:double"
    return "xs:string"


def _build_technical_property(parameter: str, value: str) -> dict[str, Any]:
    template_properties = _find_element(schema_new_technical_data_json, "TechnicalProperties")
    template_property = None
    if template_properties:
        for item in template_properties.get("value", []):
            if isinstance(item, dict) and item.get("modelType") == "Property":
                template_property = item
                break

    property_element = copy.deepcopy(template_property or {})
    property_element["category"] = property_element.get("category", "PARAMETER")
    property_element["idShort"] = parameter or "Parameter"
    property_element["valueType"] = _infer_aas_value_type(value)
    property_element["value"] = value
    property_element["modelType"] = property_element.get("modelType", "Property")
    return property_element


def _sanitize_id_part(value: str) -> str:
    """Convert an AML element name into a stable AAS id/idShort fragment."""
    cleaned = re.sub(r"[^A-Za-z0-9_]+", "_", str(value).strip())
    cleaned = re.sub(r"_+", "_", cleaned).strip("_")
    return cleaned or "InternalElement"


def _aml_element_name_from_id(element_id: str) -> str:
    """Resolve an AML InternalElement id back to its imported AML name."""
    for element in get_aml_internal_elements():
        if str(element.get("id", "")) == str(element_id):
            return str(element.get("name") or element_id)
    return str(element_id or "InternalElement")


def _new_technical_data_id(element_name: str) -> str:
    return f"https://www.aut.ruhr-uni-bochum.de/New_SM_{_sanitize_id_part(element_name)}"


def _technical_data_element_name(aml_element_name: str) -> str:
    """Use the affected AML InternalElement as the generated TechnicalData target."""
    return aml_element_name


def _fill_general_information(technical_data: dict[str, Any], element_name: str) -> None:
    """Replace InternalElement placeholders in GeneralInformation with the AML name."""
    general_information = _find_element(technical_data, "GeneralInformation")
    if not general_information:
        return

    target_fields = {
        "ManufacturerName",
        "ManufacturerProductDesignation",
        "ManufacturerArticleNumber",
        "ManufacturerOrderCode",
    }
    for item in general_information.get("value", []):
        if not isinstance(item, dict) or item.get("idShort") not in target_fields:
            continue
        current_value = str(item.get("value", ""))
        item["value"] = current_value.replace("InternalElement", element_name) or element_name


def build_new_technical_data_entries(nondeterministic_json: Any) -> list[dict[str, Any]]:
    """Create generated technical-data submodels with source-record metadata."""
    records = extract_nondeterministic_records(nondeterministic_json) or []
    entries = []
    technical_data_counts: dict[str, int] = {}

    for record_index, record in enumerate(records):
        change_description = record.get("ChangeDescription", "")
        classification_text = _invoke_string_tool(
            classify_deterministic_change,
            change_description,
        )
        aml_element = _invoke_string_tool(get_aml_element, change_description)
        aml_element_name = _aml_element_name_from_id(aml_element)

        try:
            classification = json.loads(classification_text)
        except json.JSONDecodeError:
            classification = {}

        change_type = str(classification.get("ChangeType", "")).upper()
        if change_type not in {"POST", "PATCH", "PUT"}:
            continue

        changed_property = _extract_changed_technical_property(change_description)
        technical_data_element_name = _technical_data_element_name(aml_element_name)
        technical_data = copy.deepcopy(schema_new_technical_data_json)
        sanitized_element_name = _sanitize_id_part(technical_data_element_name)
        technical_data_counts[sanitized_element_name] = (
            technical_data_counts.get(sanitized_element_name, 0) + 1
        )
        technical_data_index = technical_data_counts[sanitized_element_name]
        technical_data_id_part = f"{sanitized_element_name}_{technical_data_index}"

        technical_data["idShort"] = f"New_TechnicalData_{technical_data_id_part}"
        technical_data["id"] = _new_technical_data_id(technical_data_id_part)
        _fill_general_information(technical_data, technical_data_element_name)

        technical_properties = _find_element(technical_data, "TechnicalProperties")
        if technical_properties is not None:
            technical_properties["value"] = [
                _build_technical_property(
                    changed_property["parameter"],
                    changed_property["value"],
                )
            ]

        entries.append(
            {
                "record_index": record_index,
                "change_type": change_type,
                "aml_element_id": aml_element,
                "aml_element_name": aml_element_name,
                "technical_data_element_name": technical_data_element_name,
                "technical_data_index": technical_data_index,
                "technical_data_id": technical_data["id"],
                "submodel": technical_data,
            }
        )

    return entries


def build_new_technical_data_submodels(nondeterministic_json: Any) -> list[dict[str, Any]]:
    """Create one new technical-data submodel for each POST/PATCH/PUT change."""
    return [entry["submodel"] for entry in build_new_technical_data_entries(nondeterministic_json)]


def _set_reference_first_key_value(reference_element: dict[str, Any], value: str) -> None:
    reference_value = reference_element.setdefault(
        "value",
        {"type": "ModelReference", "keys": []},
    )
    reference_value["type"] = "ModelReference"
    keys = reference_value.setdefault("keys", [])
    if not keys:
        keys.append({"type": "Submodel", "value": value})
    keys[0]["type"] = "Submodel"
    keys[0]["value"] = value


def _set_deterministic_technical_data_references(
    deterministic_submodel: dict[str, Any],
    technical_data_entries: list[dict[str, Any]],
) -> None:
    change_records = _find_element(deterministic_submodel, "ChangeRecords")
    if not change_records:
        return

    references_by_record = {
        entry["record_index"]: entry["technical_data_id"] for entry in technical_data_entries
    }
    for record_index, record in enumerate(change_records.get("value", [])):
        technical_data_id = references_by_record.get(record_index)
        if not technical_data_id:
            continue
        technical_data_changes = _find_element(record, "TechnicalDataChanges")
        if technical_data_changes:
            _set_reference_first_key_value(technical_data_changes, technical_data_id)


def append_new_technical_data_json(
    deterministic_json: str,
    nondeterministic_json: Any,
) -> str:
    """Append generated New TechnicalData JSON after deterministic output."""
    technical_data_entries = build_new_technical_data_entries(nondeterministic_json)
    if not technical_data_entries:
        return deterministic_json

    try:
        deterministic_submodel = json.loads(deterministic_json)
    except json.JSONDecodeError:
        deterministic_submodel = None

    if isinstance(deterministic_submodel, dict):
        _set_deterministic_technical_data_references(
            deterministic_submodel,
            technical_data_entries,
        )
        deterministic_json = json.dumps(
            deterministic_submodel,
            indent=2,
            ensure_ascii=False,
        )

    technical_data_json = "\n\n".join(
        json.dumps(entry["submodel"], indent=2, ensure_ascii=False)
        for entry in technical_data_entries
    )
    return f"{deterministic_json}\n\n{technical_data_json}"


def compact_deterministic_agent_input(nondeterministic_json):
    """Keep the deterministic LLM prompt small while preserving source data."""
    records = extract_nondeterministic_records(nondeterministic_json)
    if records is None:
        return str(nondeterministic_json)

    return json.dumps({"ChangeRecords": records}, indent=2, ensure_ascii=False)


def _invoke_deterministic_agent(
    agent_deterministic,
    nondeterministic_json: Any,
) -> str:
    """Convert nondeterministic AAS JSON into deterministic AAS JSON."""
    compact_input = compact_deterministic_agent_input(nondeterministic_json)
    deterministic_schema = json.dumps(
        aas_deterministic_schema_json,
        indent=2,
        ensure_ascii=False,
    )
    deterministic_item_template = json.dumps(
        schema_items_deterministic_json,
        indent=2,
        ensure_ascii=False,
    )

    # The deterministic agent receives compact records plus the schema context.
    response = agent_deterministic.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Convert these Nondeterministic Engineering Change records "
                        "to a Deterministic Engineering Change submodel.\n\n"
                        "Use this deterministic submodel skeleton as the output root:\n"
                        f"{deterministic_schema}\n\n"
                        "Use this deterministic ChangeRecord template for each input record:\n"
                        f"{deterministic_item_template}\n\n"
                        "Input records:\n"
                        f"{compact_input}"
                    )
                )
            ]
        },
        config={
            "configurable": {"thread_id": "deterministic_agent"},
            "recursion_limit": 40,
        },
    )

    deterministic_json = normalize_json_output(response["messages"][-1].content)
    displayed_deterministic_json = append_new_technical_data_json(
        deterministic_json,
        nondeterministic_json,
    )

    global deterministic_agent_call_count, last_deterministic_submodel_json
    deterministic_agent_call_count += 1
    last_deterministic_submodel_json = displayed_deterministic_json
    deterministic_submodel_json_path.write_text(
        displayed_deterministic_json,
        encoding="utf-8",
    )

    return displayed_deterministic_json


def _extract_updated_aml_model(agent_output: str, fallback: str) -> str:
    normalized = normalize_json_output(agent_output)
    try:
        payload = json.loads(normalized)
    except json.JSONDecodeError:
        return agent_output.strip() or fallback

    if isinstance(payload, dict):
        updated = payload.get("updated_aml_model") or payload.get("updatedAmlModel")
        if isinstance(updated, str) and updated.strip():
            return updated.strip()
    return fallback


def _reference_key_values(element: dict[str, Any] | None) -> list[str]:
    """Return values from an AAS ReferenceElement's keys."""
    if not isinstance(element, dict):
        return []

    value = element.get("value")
    if not isinstance(value, dict):
        return []

    keys = value.get("keys", [])
    if not isinstance(keys, list):
        return []

    values = []
    for key in keys:
        if isinstance(key, dict) and key.get("value"):
            values.append(str(key["value"]))
    return values


def _element_value_text(element: dict[str, Any] | None) -> str:
    """Return a compact string value from an AAS element."""
    if not isinstance(element, dict):
        return ""

    value = element.get("value", "")
    if isinstance(value, str):
        return value
    if isinstance(value, (int, float, bool)):
        return str(value)
    if isinstance(value, dict):
        reference_values = _reference_key_values(element)
        if reference_values:
            return reference_values[-1]
    return ""


def _document_identity_values(doc: dict[str, Any]) -> set[str]:
    values = {
        str(doc.get("id", "")),
        str(doc.get("idShort", "")),
    }
    return {value for value in values if value}


def _technical_data_documents_by_reference(docs: list[Any]) -> dict[str, dict[str, Any]]:
    lookup: dict[str, dict[str, Any]] = {}
    for doc in docs:
        if not isinstance(doc, dict):
            continue
        if _find_element(doc, "TechnicalProperties") is None:
            continue
        for value in _document_identity_values(doc):
            lookup[value] = doc
    return lookup


def _matching_technical_data_json(
    record: dict[str, Any],
    technical_data_by_reference: dict[str, dict[str, Any]],
) -> str:
    technical_data_changes = _find_element(record, "TechnicalDataChanges")
    for reference in _reference_key_values(technical_data_changes):
        doc = technical_data_by_reference.get(reference)
        if doc:
            return json.dumps(doc, ensure_ascii=False)
    return ""


def _extract_deterministic_change_records(
    deterministic_and_technical_json: Any,
) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    docs = _json_documents_from_text(deterministic_and_technical_json)
    deterministic_doc = None
    for doc in docs:
        if isinstance(doc, dict) and _find_element(doc, "ChangeRecords") is not None:
            deterministic_doc = doc
            break

    if deterministic_doc is None:
        return [], _technical_data_documents_by_reference(docs)

    change_records = _find_element(deterministic_doc, "ChangeRecords")
    records = change_records.get("value", []) if isinstance(change_records, dict) else []
    records = [record for record in records if isinstance(record, dict)]
    return records, _technical_data_documents_by_reference(docs)


def update_aml_model_direct(
    deterministic_and_technical_json: Any,
    aml_model: str,
) -> str:
    """Apply deterministic/technical AAS updates to AML XML without an LLM round trip."""
    records, technical_data_by_reference = _extract_deterministic_change_records(
        deterministic_and_technical_json
    )
    if not records:
        raise ValueError("No deterministic ChangeRecords found in the provided context.")

    updated_aml = aml_model
    for record in records:
        change_type = _element_value_text(_find_element(record, "ChangeType")).upper()
        change_description = _element_value_text(_find_element(record, "ChangeDescription"))
        aml_element = _element_value_text(_find_element(record, "AMLElement"))
        technical_data = _matching_technical_data_json(record, technical_data_by_reference)

        tool_args = {
            "ChangeDescription": change_description,
            "AML_model": updated_aml,
            "AML_element": aml_element,
            "TechnicalData": technical_data,
        }

        if change_type == "DELETE":
            updated_aml = operation_delete.invoke(tool_args)
        elif change_type == "PUT":
            updated_aml = operation_put.invoke(tool_args)
        elif change_type == "POST":
            updated_aml = operation_post.invoke(tool_args)
        else:
            updated_aml = operation_patch.invoke(tool_args)

    return str(updated_aml)


def _invoke_aml_agent(
    agent_aml,
    deterministic_and_technical_json: Any,
    aml_model: str,
) -> str:
    """Apply deterministic/technical AAS updates to the imported AML XML model."""
    response = agent_aml.invoke(
        {
            "messages": [
                HumanMessage(
                    content=(
                        "Call operation_apply_aml_changes exactly once, then return only "
                        "the required JSON object containing updated_aml_model.\n\n"
                        "deterministic_and_technical_json:\n"
                        f"{deterministic_and_technical_json}\n\n"
                        "AML_model:\n"
                        f"{aml_model}"
                    )
                )
            ]
        },
        config={
            "configurable": {"thread_id": "aml_agent"},
            "recursion_limit": 20,
        },
    )
    return _extract_updated_aml_model(response["messages"][-1].content, aml_model)


def _get_aml_agent(model_name: str):
    """Return a cached AML update agent for the selected model."""
    agent_aml = aml_agent_cache.get(model_name)
    if agent_aml is not None:
        return agent_aml

    selected_model = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0,
    )
    agent_aml = create_agent(
        model=selected_model,
        tools=aml_TOOLS,
        system_prompt=prompt_aml,
        checkpointer=aml_agent_checkpointer,
    )
    aml_agent_cache[model_name] = agent_aml
    return agent_aml


def update_aml_model(
    model_name: str,
    deterministic_and_technical_json: Any,
    aml_model: str,
) -> str:
    """Apply deterministic/technical AAS updates for the UI Update button via aml_agent."""
    agent_aml = _get_aml_agent(model_name)
    return _invoke_aml_agent(agent_aml, deterministic_and_technical_json, aml_model)


def convert_nondeterministic_to_deterministic(
    model_name: str,
    nondeterministic_json: Any,
) -> str:
    """Hybrid Flask/UI entry point: LLM deterministic JSON, Python TechnicalData."""
    selected_model = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0,
    )
    agent_deterministic = create_agent(
        model=selected_model,
        tools=DETERMINISTIC_TOOLS,
        system_prompt=prompt_deterministic,
        checkpointer=deterministic_agent_checkpointer,
    )

    return _invoke_deterministic_agent(agent_deterministic, nondeterministic_json)


# ============================================================
# Main agent builder
# ============================================================


def build_main_agent(model_name: str):
    """Build the main ECM coordinator agent and its two subagents."""
    selected_model = ChatOllama(
        model=model_name,
        base_url="http://localhost:11434",
        temperature=0,
    )

    agent_nondeterministic = create_agent(
        model=selected_model,
        tools=[get_current_datetime],
        system_prompt=prompt_nondeterministic_agent,
        checkpointer=nondeterministic_agent_checkpointer,
    )

    agent_deterministic = create_agent(
        model=selected_model,
        tools=DETERMINISTIC_TOOLS,
        system_prompt=prompt_deterministic,
        checkpointer=deterministic_agent_checkpointer,
    )
    
    agent_aml = create_agent(
        model=selected_model,
        tools=aml_TOOLS,
        system_prompt=prompt_aml,
        checkpointer=aml_agent_checkpointer,
    )

    @tool
    def call_subagent_nondeterministic(feedback_text: str) -> str:
        """Convert itemized engineering feedback into nondeterministic AAS JSON."""
        response = agent_nondeterministic.invoke(
            {
                "messages": [
                    HumanMessage(
                        content=(
                            "I have engineering feedback, and I want to convert this "
                            "itemized feedback into the Nondeterministic Engineering "
                            "Change submodel. Use only the actual feedback items below.\n\n"
                            f"{feedback_text}"
                        )
                    )
                ]
            },
            config={"configurable": {"thread_id": "nondeterministic_agent"}},
        )

        submodel_json = normalize_json_output(response["messages"][-1].content)

        global nondeterministic_agent_call_count, last_nondeterministic_submodel_json
        nondeterministic_agent_call_count += 1
        last_nondeterministic_submodel_json = submodel_json
        nondeterministic_submodel_json_path.write_text(submodel_json, encoding="utf-8")

        return submodel_json

    @tool
    def call_subagent_deterministic(nondeterministic_json: Any) -> str:
        """Convert nondeterministic AAS JSON with the deterministic subagent."""
        return _invoke_deterministic_agent(agent_deterministic, nondeterministic_json)

    @tool
    def call_subagent_aml(deterministic_and_technical_json: Any, aml_model: str) -> str:
        """Update imported AML XML from deterministic and TechnicalData AAS JSON."""
        return _invoke_aml_agent(agent_aml, deterministic_and_technical_json, aml_model)

    # The main agent only routes work; generation happens inside subagents.
    return create_agent(
        model=selected_model,
        tools=[
            call_subagent_nondeterministic,
            call_subagent_deterministic,
            call_subagent_aml,
        ],
        system_prompt=system_prompt,
        checkpointer=main_agent_checkpointer,
    )


system_prompt = """
You are the MAIN COORDINATOR for an Engineering Change Management (ECM) system based on Asset Administration Shell (AAS).

Your ONLY job is to route requests to the correct tool(s).

You MUST NEVER:
- generate or modify AAS JSON yourself
- summarize tool outputs
- reformat tool outputs
- add explanations when a tool is used

---

## AVAILABLE TOOLS

1. call_subagent_nondeterministic(feedback_text)
→ Converts raw engineering feedback into a Nondeterministic AAS Submodel JSON

2. call_subagent_deterministic(nondeterministic_json)
→ Converts Nondeterministic AAS JSON into Deterministic AAS JSON

3. call_subagent_aml(deterministic_and_technical_json, aml_model)
→ Applies Deterministic AAS JSON and TechnicalData JSON to imported AML XML

---

## ROUTING RULES (STRICT)

### Case 1 — Nondeterministic conversion
If user provides raw engineering feedback (bullet points, numbered list, free text):
→ CALL: call_subagent_nondeterministic(feedback_text)

### Case 2 — Deterministic conversion
If user provides Nondeterministic AAS JSON:
→ CALL: call_subagent_deterministic(nondeterministic_json)

### Case 3 — Full pipeline request (both steps)
If user requests "full conversion", "end-to-end", or both models:
1. CALL call_subagent_nondeterministic(feedback_text)
2. TAKE its output EXACTLY as input
3. CALL call_subagent_deterministic(output_json)

### Case 4 — AML XML update
If user provides Deterministic AAS JSON, TechnicalData JSON, and imported AML XML, or asks to update/apply changes to AML:
→ CALL: call_subagent_aml(deterministic_and_technical_json, aml_model)

### Case 5 — General ECM question
If user asks conceptual questions:
→ answer briefly without tools

---

## CRITICAL OUTPUT RULE

When a tool is used:
→ RETURN ONLY THE FINAL TOOL OUTPUT
→ NO explanation
→ NO markdown
→ NO additional text

---

## DATA INTEGRITY RULES

- Never remove engineering details
- Never rephrase technical fields
- Never invent missing values
- Always pass full user input to subagents unchanged
"""
