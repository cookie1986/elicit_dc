"""This script contains classes and functions for storing case information that has been extracted from the document."""
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Literal, Optional, Set, Tuple, Union
import pandas as pd
import spacy
from spacy.language import Language
from tqdm import tqdm

from case_extraction.utils.utils import context_from_doc_char

@dataclass
class Evidence:
    """
    Class for storing evidence.
    """
    exact_context: str
    local_context: str
    wider_context: str

    @classmethod
    def no_match(cls) -> "Evidence":
        """Returns an evidence object with no match."""
        return cls("None", "No Evidence Found", "No Evidence Found")
    
    @classmethod
    def from_character_startend(cls, doc: str, start: int, end:int, local_padding: int = 100, wider_padding: int = 500, max_chars: int = 100) -> "Evidence":
        """
        Returns an evidence object from a character start and end index.
        """
        mid = (start + end) // 2
        start = int(max(start, mid - (max_chars / 2)))
        end = int(min(end, mid + (max_chars / 2)))
        exact_context = context_from_doc_char(doc, start, end, padding=0)
        local_context = context_from_doc_char(doc, start, end, local_padding)
        wider_context = context_from_doc_char(doc, start, end, wider_padding)
        return cls(exact_context, local_context, wider_context)
    
    @classmethod
    def from_spacy(cls, doc: Language, start: int, end:int, local_padding: int = 0, wider_padding: int = 10) -> "Evidence":
        """
        Returns an evidence object from a character start and end index.
        """
        exact_context = doc[start:end]
        local_start = max(0, start - local_padding)
        local_end = min(len(doc), end + local_padding)
        local_context = doc[local_start:local_end]
        wider_start = max(0, start - wider_padding)
        wider_end = min(len(doc), end + wider_padding)
        wider_context = doc[wider_start:wider_end]
        return cls(exact_context.text, local_context.text, wider_context.text)
    
    @classmethod
    def from_spacy_multiple(cls, doc: Language, evidence_list: List[Tuple[str, int, int]], wider_padding: int = 20) -> "Evidence":
        local_context = ", ".join([span for span, _, _ in evidence_list])
        widers = []
        for span, start, end in evidence_list:
            start = max(0, start - wider_padding)
            end = min(len(doc), end + wider_padding)
            widers.append(doc[start:end].text)
        wider_context = " | ".join(widers)
        return cls(local_context, local_context, wider_context)

class CaseField:
    """
    Class for storing case information for a single variable.

    :param value: The value of the variable.
    :param confidence: The confidence of the value.
    :param evidence: The evidence for the value.
    """
    def __init__(self, value: str, confidence: float, evidence: Evidence):
        self.value = value
        self.confidence = confidence
        self.evidence = evidence
    
    def __str__(self) -> str:
        return f"{self.value} ({self.confidence})"
    
    def __mul__(self, other: float) -> "CaseField":
        self.confidence *= other
        return self
    
    def __rmul__(self, other: float) -> "CaseField":
        self.confidence *= other
        return self
    
    def __add__(self, other: Union["CaseField", list]) -> List["CaseField"]:
        if isinstance(other, CaseField):
            return [self, other]
        elif isinstance(other, list):
            return [self] + other
        else:
            raise TypeError(f"Cannot add {type(other)} to CaseField")
    
    def __radd__(self, other: Union["CaseField", list]) -> List["CaseField"]:
        if isinstance(other, CaseField):
            return [self, other]
        elif isinstance(other, list):
            return [self] + other
        else:
            raise TypeError(f"Cannot add {type(other)} to CaseField")


class Case:
    """
    Class for storing case information. 
    Can take arbitrary CaseFields as input.

    :param filename: The filename of the case.
    :param method: The method of extraction.

    """
    def __init__(self, filename: str, method: str, **kwargs):
        self.filename = filename
        self.method = method
        self.__dict__.update(kwargs)
        
    def add_dict(self, dict_to_add: dict):
        """
        Adds a dictionary of CaseFields to the case.

        :param dict_to_add: The dictionary to add.
        """
        for key, value in dict_to_add.items():
            if not hasattr(self, key):
                setattr(self, key, value)
            else:
                current = getattr(self, key)
                if isinstance(value, CaseField):
                    value = [value]
                setattr(self, key, current + value)
    
    def add_field(self, field: str, casefield: CaseField):
        """
        Adds a CaseField to the case.

        :param field: The field to add.
        :param casefield: The CaseField to add.
        """
        if not hasattr(self, field):
            setattr(self, field, casefield)
        else:
            current = getattr(self, field)
            if isinstance(casefield, CaseField):
                casefield = [casefield]
            setattr(self, field, current + casefield)
                
    @classmethod
    def manual_entry(cls, filename: Union[str, Path], **kwargs):
        """
        Class method for creating a case object using direct input for test cases.

        :param filename: The filename of the case.
        :param kwargs: The CaseFields to add to the case.
        """
        for key, value in kwargs.items():
            if isinstance(value, list) or isinstance(value, set):
                kwargs[key] = [CaseField(value=v, confidence=1, evidence="") for v in value]
            else:
                kwargs[key] = CaseField(value=value, confidence=1, evidence="")
        return cls(filename=filename, method="manual", **kwargs)

    def __str__(self):
        """
        Returns a string representation of the case.
        """
        if not hasattr(self, "defendants"):
            return "No case name extracted"
        if isinstance(self.defendants, Iterable):
            return f"R v. {', '.join([d.value for d in self.defendants])}"
        else:
            return f"R v. {self.defendants.value}"

    def to_dict(self):
        """
        Returns a dictionary representation of the case, only including the CaseFields.
        """
        return {k: v for k, v in self.__dict__.items() if isinstance(v, CaseField) or isinstance(v, list)}