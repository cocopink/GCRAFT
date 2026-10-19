"""
Baseline prompt templates for WikiTQ and TabFact.
Prompt format follows the paper figures:
  - Figure 16: End-to-End QA (zero-shot)
  - Figure 17: Few-Shot QA (2-shot, direct answer)
  - Figure 18: Chain-of-Thought (2-shot, CoT)
"""

import sys
import os

# ---------------------------------------------------------------------------
# WikiTQ few-shot example data (from paper figures)
# ---------------------------------------------------------------------------

WIKITQ_EXAMPLE_TABLE_CYCLIST = """col : Rank | Cyclist | Team | Time | UCI ProTour; Points
row 1 : 1 | Alejandro Valverde (ESP) | Caisse d'Epargne | 5h 29' 10" | 40
row 2 : 2 | Alexandr Kolobnev (RUS) | Team CSC Saxo Bank | s.t. | 30
row 3 : 3 | Davide Rebellin (ITA) | Gerolsteiner | s.t. | 25
row 4 : 4 | Paolo Bettini (ITA) | Quick Step | s.t. | 20
row 5 : 5 | Franco Pellizotti (ITA) | Liquigas | s.t. | 15
row 6 : 6 | Denis Menchov (RUS) | Rabobank | s.t. | 11
row 7 : 7 | Samuel Sanchez (ESP) | Euskaltel-Euskadi | s.t. | 7
row 8 : 8 | Stephane Goubert (FRA) | Ag2r-La Mondiale | + 2" | 5
row 9 : 9 | Haimar Zubeldia (ESP) | Euskaltel-Euskadi | + 2" | 3
row 10 : 10 | David Moncoutie (FRA) | Cofidis | + 2" | 1"""

# ---------------------------------------------------------------------------
# TabFact few-shot example data (adapted from existing general_cot)
# ---------------------------------------------------------------------------

TABFACT_EXAMPLE_1_TABLE = """table caption: 2008 sidecarcross world championship
col : position | driver / passenger | equipment | bike no | points
row 1 : 1 | Daniel Willemsen / Reto Grutter | ktm - ayr | 1 | 531
row 2 : 2 | Kristers Sergis / Kaspars Stupelis | ktm - ayr | 3 | 434
row 3 : 3 | Jan Hendrickx / Tim Smeuninx | zabel - vmc | 2 | 421
row 4 : 4 | Joris Hendrickx / Kaspars Liepins | zabel - vmc | 8 | 394
row 5 : 5 | Marco Happich / Meinrad Schelbert | zabel - mefo | 7 | 317"""

TABFACT_EXAMPLE_1_STATEMENT = "bike number 3 is the only one to use equipment ktm - ayr."

TABFACT_EXAMPLE_2_TABLE = """table caption: 1957 vfl season
col : home team | home team score | away team | away team score | venue | crowd | date
row 1 : footscray | 6.6 (42) | north melbourne | 8.13 (61) | western oval | 13325 | 10 august 1957
row 2 : essendon | 10.15 (75) | south melbourne | 7.13 (55) | windy hill | 16000 | 10 august 1957
row 3 : st kilda | 1.5 (11) | melbourne | 6.13 (49) | junction oval | 17100 | 10 august 1957
row 4 : hawthorn | 14.19 (103) | geelong | 8.7 (55) | brunswick street oval | 12000 | 10 august 1957
row 5 : fitzroy | 8.14 (62) | collingwood | 8.13 (61) | glenferrie oval | 22000 | 10 august 1957"""

TABFACT_EXAMPLE_2_STATEMENT = "collingwood was the away team playing at the brunswick street oval venue."


# ===========================================================================
# WikiTQ prompt builders
# ===========================================================================

def build_wikitq_e2e_prompt(table_str, question):
    """Figure 16: End-to-End QA (zero-shot)"""
    return (
        f"Here is the table to answer this question. Answer the question.\n"
        f"/*\n{table_str}\n*/\n"
        f"Question: {question}\n"
        f"The answer is:"
    )


def build_wikitq_few_shot_prompt(table_str, question):
    """Figure 17: Few-Shot QA (2-shot, direct answer format)"""
    # Example 1: direct answer
    example_1 = (
        f"/*\n{WIKITQ_EXAMPLE_TABLE_CYCLIST}\n*/\n"
        f"Question: which country had the most cyclists finish within the top 10?\n"
        f"The answer is: Italy.\n"
    )
    # Example 2: CoT instruction but direct answer (matching Figure 17)
    example_2 = (
        # f"Here is the table to answer this question. Please provide your explanation first, then "
        # f"answer the question in a short phrase starting by 'therefore, the answer is:'\n"
        f"/*\n{WIKITQ_EXAMPLE_TABLE_CYCLIST}\n*/\n"
        f"Question: how many players got less than 10 points?\n"
        f"The answer is: 4.\n"
    )
    return (
        f"{example_1}\n"
        f"{example_2}\n"
        f"Here is the table to answer this question. Answer the question.\n"
        f"/*\n{table_str}\n*/\n"
        f"Question: {question}\n"
        f"The answer is:"
    )


def build_wikitq_cot_prompt(table_str, question):
    """Figure 18: Chain-of-Thought (2-shot, CoT format)"""
    cot_instruction = (
        "Here is the table to answer this question. Please provide your explanation first, then "
        "answer the question in a short phrase starting by 'therefore, the answer is:'"
    )
    # Example 1
    example_1 = (
        f"{cot_instruction}\n"
        f"/*\n{WIKITQ_EXAMPLE_TABLE_CYCLIST}\n*/\n"
        f"Question: which country had the most cyclists finish within the top 10?\n"
        f"Explanation: ITA occurs three times in the table, more than any others. "
        f"Therefore, the answer is: Italy.\n"
    )
    # Example 2
    example_2 = (
        f"{cot_instruction}\n"
        f"/*\n{WIKITQ_EXAMPLE_TABLE_CYCLIST}\n*/\n"
        f"Question: how many players got less than 10 points?\n"
        f"Explanation: Samuel Sanchez, Stephane Goubert, Haimar Zubeldia and David Moncoutie "
        f"received less than 10 points. Therefore, the answer is: 4.\n"
    )
    return (
        f"{example_1}\n"
        f"{example_2}\n"
        f"{cot_instruction}\n"
        f"/*\n{table_str}\n*/\n"
        f"Question: {question}\n"
        f"Explanation:"
    )


# ===========================================================================
# TabFact prompt builders
# ===========================================================================

def build_tabfact_e2e_prompt(table_str, statement):
    """TabFact End-to-End (zero-shot, adapted from Figure 16)"""
    return (
        f"Here is the statement about the table and the task is to tell whether "
        f"the statement is true or false.\n"
        f"/*\n{table_str}\n*/\n"
        f"Statement: {statement}\n"
        f"The answer is:"
    )


def build_tabfact_few_shot_prompt(table_str, statement):
    """TabFact Few-Shot (2-shot, direct answer, adapted from Figure 17)"""
    # Example 1
    example_1 = (
        f"/*\n{TABFACT_EXAMPLE_1_TABLE}\n*/\n"
        f"Statement: {TABFACT_EXAMPLE_1_STATEMENT}\n"
        f"The answer is: NO.\n"
    )
    # Example 2
    example_2 = (
        f"Here is the statement about the table and the task is to tell whether "
        f"the statement is true or false.\n"
        f"/*\n{TABFACT_EXAMPLE_2_TABLE}\n*/\n"
        f"Statement: {TABFACT_EXAMPLE_2_STATEMENT}\n"
        f"The answer is: NO.\n"
    )
    return (
        f"{example_1}\n"
        f"{example_2}\n"
        f"Here is the statement about the table and the task is to tell whether "
        f"the statement is true or false.\n"
        f"/*\n{table_str}\n*/\n"
        f"Statement: {statement}\n"
        f"The answer is:"
    )


def build_tabfact_cot_prompt(table_str, statement):
    """TabFact Chain-of-Thought (2-shot, CoT format, adapted from Figure 18)"""
    cot_instruction = (
        "Here is the statement about the table and the task is to tell whether "
        "the statement is true or false. Please provide your explanation first, then "
        "answer in a short phrase starting by 'therefore, the answer is:'"
    )
    # Example 1
    example_1 = (
        f"{cot_instruction}\n"
        f"/*\n{TABFACT_EXAMPLE_1_TABLE}\n*/\n"
        f"Statement: {TABFACT_EXAMPLE_1_STATEMENT}\n"
        f"Explanation: Bike number 3 (Kristers Sergis / Kaspars Stupelis) uses ktm - ayr, "
        f"but bike number 1 (Daniel Willemsen / Reto Grutter) also uses ktm - ayr. "
        f"Therefore, the answer is: NO.\n"
    )
    # Example 2
    example_2 = (
        f"{cot_instruction}\n"
        f"/*\n{TABFACT_EXAMPLE_2_TABLE}\n*/\n"
        f"Statement: {TABFACT_EXAMPLE_2_STATEMENT}\n"
        f"Explanation: Collingwood played at Glenferrie Oval, not brunswick street oval. "
        f"Hawthorn played at brunswick street oval. Therefore, the answer is: NO.\n"
    )
    return (
        f"{example_1}\n"
        f"{example_2}\n"
        f"{cot_instruction}\n"
        f"/*\n{table_str}\n*/\n"
        f"Statement: {statement}\n"
        f"Explanation:"
    )


# ===========================================================================
# Unified prompt builder dispatcher
# ===========================================================================

def build_prompt(dataset, method, table_str, question_or_statement):
    """
    Build prompt for any (dataset, method) combination.

    Args:
        dataset: "wikitq" or "tabfact"
        method: "e2e", "few_shot", or "cot"
        table_str: linearized table string (from table2string)
        question_or_statement: the question (WikiTQ) or statement (TabFact)
    """
    if dataset == "wikitq":
        builders = {
            "e2e": build_wikitq_e2e_prompt,
            "few_shot": build_wikitq_few_shot_prompt,
            "cot": build_wikitq_cot_prompt,
        }
    elif dataset == "tabfact":
        builders = {
            "e2e": build_tabfact_e2e_prompt,
            "few_shot": build_tabfact_few_shot_prompt,
            "cot": build_tabfact_cot_prompt,
        }
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

    builder = builders.get(method)
    if builder is None:
        raise ValueError(f"Unknown method: {method}")

    return builder(table_str, question_or_statement)
