"""
rule_engine.py
--------------
THE TRADITIONAL AI COMPONENT OF THIS PROJECT.

This module implements a Knowledge-Based Rule Engine that performs
forward-chaining-style reasoning over region features extracted by
feature_extraction.py.

There is NO machine learning, NO statistical model, and NO training
anywhere in this file. Every decision is made through explicit,
human-readable IF-THEN rules that reason over facts (region features)
using a knowledge base of thresholds.

Traditional AI concepts demonstrated here:
    1. Knowledge Representation  -> KnowledgeBase dataclass / dict
    2. Knowledge Base            -> minimum_area, minimum_width, minimum_height
    3. Facts                     -> region features (area, width, height, ...)
    4. IF-THEN Rules             -> evaluate_rules()
    5. Forward-Chaining Inference-> facts are matched against rules in
                                     sequence to reach a final conclusion
    6. Explicit Decision Logic   -> every decision returns the reasoning trace
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any
import pandas as pd


@dataclass
class KnowledgeBase:
    """
    The Knowledge Base holds the domain knowledge (thresholds) that the
    rule engine reasons with. In a traditional (symbolic) AI system,
    this represents the declarative knowledge given to the system by
    a domain expert -- here, the student/user configures it via the
    Streamlit sidebar.
    """
    minimum_area: int = 500
    minimum_width: int = 20
    minimum_height: int = 20

    def as_dict(self) -> Dict[str, Any]:
        return {
            "minimum_area": self.minimum_area,
            "minimum_width": self.minimum_width,
            "minimum_height": self.minimum_height,
        }


def evaluate_rules(region: Dict[str, Any], kb: KnowledgeBase) -> Dict[str, Any]:
    """
    Apply forward-chaining-style rule evaluation to a single region.

    Facts (input):
        region["area"], region["width"], region["height"]

    Rules (knowledge base driven):
        RULE 1: IF area   >= kb.minimum_area   THEN area_condition   = TRUE
        RULE 2: IF width  >= kb.minimum_width  THEN width_condition  = TRUE
        RULE 3: IF height >= kb.minimum_height THEN height_condition = TRUE
        RULE 4: IF area_condition AND width_condition AND height_condition
                THEN region = RELEVANT (SELECTED)
                ELSE region = IRRELEVANT (REJECTED)

    This is a simple forward-chaining inference: each rule fires based on
    facts already known/derived, and the final rule combines the
    intermediate conclusions (area_condition, width_condition,
    height_condition) to reach the final decision -- exactly the pattern
    used in classical rule-based expert systems.

    Returns a dict containing the original facts, each intermediate
    condition, the final decision, and a human-readable reasoning trace
    (used by the Streamlit UI to display "AI Rule-Based Decision").
    """
    area = region.get("area", 0)
    width = region.get("width", 0)
    height = region.get("height", 0)

    # --- Fact -> Rule -> Derived Fact (forward chaining) ---
    area_condition = area >= kb.minimum_area
    width_condition = width >= kb.minimum_width
    height_condition = height >= kb.minimum_height

    # Final rule: combine derived facts into the ultimate decision
    is_relevant = area_condition and width_condition and height_condition
    decision = "SELECTED" if is_relevant else "REJECTED"

    reasoning_trace = [
        {
            "rule": "RULE 1: Area Condition",
            "condition": f"area >= minimum_area  ({area} >= {kb.minimum_area})",
            "result": area_condition,
        },
        {
            "rule": "RULE 2: Width Condition",
            "condition": f"width >= minimum_width  ({width} >= {kb.minimum_width})",
            "result": width_condition,
        },
        {
            "rule": "RULE 3: Height Condition",
            "condition": f"height >= minimum_height  ({height} >= {kb.minimum_height})",
            "result": height_condition,
        },
        {
            "rule": "RULE 4: Final Decision (AND of all conditions)",
            "condition": "area_condition AND width_condition AND height_condition",
            "result": is_relevant,
        },
    ]

    return {
        "region_id": region.get("region_id"),
        "area": area,
        "width": width,
        "height": height,
        "area_condition": area_condition,
        "width_condition": width_condition,
        "height_condition": height_condition,
        "decision": decision,
        "reasoning_trace": reasoning_trace,
    }


def run_rule_engine(regions_df: pd.DataFrame, kb: KnowledgeBase) -> List[Dict[str, Any]]:
    """
    Run the forward-chaining rule engine over every detected region.

    This is the "Rule-Based Reasoning" step in the overall architecture:

        Feature Extraction -> Knowledge Base -> Rule Engine -> Reasoning -> Decision

    Returns a list of per-region decision dictionaries (see evaluate_rules).
    """
    results = []
    if regions_df is None or len(regions_df) == 0:
        return results

    for _, region in regions_df.iterrows():
        result = evaluate_rules(region.to_dict(), kb)
        results.append(result)

    return results


def summarize_decisions(decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
    """
    Compute simple region-selection statistics from the rule engine's
    decisions. These are NOT "AI accuracy" metrics -- they simply
    describe how many regions were selected vs rejected by the rules.
    """
    total = len(decisions)
    selected = sum(1 for d in decisions if d["decision"] == "SELECTED")
    rejected = total - selected
    percentage = round((selected / total) * 100, 2) if total > 0 else 0.0

    return {
        "total_regions": total,
        "selected_regions": selected,
        "rejected_regions": rejected,
        "selection_percentage": percentage,
    }
