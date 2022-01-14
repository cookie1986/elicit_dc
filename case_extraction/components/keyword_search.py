"""Script which searches for keywords (from a schema) in a document."""
from pathlib import Path
from typing import Dict, List, Tuple, Union
from spacy.language import Language

import yaml
import spacy
from spacy.matcher import PhraseMatcher
from prefect import task

from case_extraction.case import Case, CaseField, Evidence
from case_extraction.utils.loading import load_schema


def exact_match_single(doc: str, keywords: Dict[str, List[str]]) -> List[CaseField]:
    """
    Extracts the keywords from the document for a single field.

    :param doc: The document to extract the keywords from.
    :param keywords: The keywords to search for. Form is: {field: [keywords]}

    :return: A list of CaseFields.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(doc)
    matcher = PhraseMatcher(nlp.vocab, attr="LOWER")
    for k in keywords.keys():
        patterns = [nlp.make_doc(text) for text in keywords[k]]
        matcher.add(k, patterns)
    matches = matcher(doc)
    exact_matches = {}
    for match_id, start, end in matches:
        match = doc.vocab.strings[match_id]
        span = doc[start:end]
        if match not in exact_matches:
            exact_matches[match] = [(span.text, start, end)]
        else:
            exact_matches[match] += [(span.text, start, end)]
    casefields = []
    for match in exact_matches.keys():
        casefields.append(CaseField(value=match, confidence=1.0, evidence=Evidence.from_spacy_multiple(doc, exact_matches[match])))
    return casefields

@task
def exact_match(doc: str, case: Case, keyword_path: Path, categories_path: Path) -> Case:
    """
    Match the keywords in the document with the keywords in the keywords file.

    :param doc: The document to extract the keywords from.
    :param case: The case to add the keywords to.
    :param keyword_path: The path to the keywords file.
    :param categories_path: The path to the categories file.

    :return: The case with the keywords added.
    """
    field_keywords = load_schema(keyword_path)
    categories = load_schema(categories_path)
    for field in field_keywords.keys():
        match = exact_match_single(doc, field_keywords[field])
        if match:
            setattr(case, field, match)
        else:
            default_category = categories[field][-1]
            cf = CaseField(value=default_category, confidence=0, evidence=Evidence.no_match())
            setattr(case, field, cf)
    return case
        
        
