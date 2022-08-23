import os
import re
import pandas as pd
from typing import DefaultDict, Dict, List, Tuple

from transformers import Pipeline, pipeline
from elicit.interface import CategoricalLabellingFunction, Extraction

import warnings

# warnings.filterwarnings("ignore")

def group_msgs(
    chat_doc: str,
    window_size: int
    ):
    """
    Group individual messages into an overlapping fixed size window.

    :param chat_doc: Document of chat messages
    :param window_size: number of messages in group

    :return: Tuple with form grouped messages, starting index, end index. Grouped messages include speaker label.
    """
    messages = chat_doc.splitlines()

    grouped_msgs = []

    for idx, row in enumerate(messages):
        speaker, msg = re.split(r'(?<=PRED|VICT):', row)
        if speaker == 'PRED':
            local_group = [row]
            for prev_idx in range(1,window_size):
                if idx-prev_idx >= 0:
                    local_group.append(messages[idx-prev_idx])
            local_group.reverse()
            grouped_msg = '\n'.join(local_group)
            start = chat_doc.index(grouped_msg)
            end = start+len(grouped_msg)           
            grouped_msgs.append((grouped_msg, start, end))

    return grouped_msgs

def compress(candidates: List[Tuple[str, float, int, int]]) -> Dict[str, float]:
    """
    Compress the list of candidate answers, summing the probabilities where the answer is the same.
    Context returned is the max of the same answer.

    :param candidates: List of candidate answers. Form is [(answer, score, start, end)]. Multiple answers can be the same, but coming from different parts of the document. These are summed together.

    :return: Dictionary of answers and their probabilities.
    """
    prob_dict = DefaultDict(float)
    prob_sum = 0.0
    max_context = {}
    max_candidate = DefaultDict(float)
    for candidate, prob, start, end in candidates:
        if prob > max_candidate[candidate]:
            max_candidate[candidate] = prob
            max_context[candidate] = {"start": start, "end": end}
    for ci in candidates:
        prob_dict[ci[0]] += ci[1]
        prob_sum += ci[1]
    return {k: v / prob_sum for k, v in prob_dict.items()}, max_context

def match_classify(
                    chat_doc: str, 
                    grouped_msgs: Tuple[str, int, int],
                    levels: List[str], 
                    model_name: str,
                    # classification_model: Pipeline, 
                    filter_threshold: float,
                    ):
    """
    Match answers from the Q&A Transformer to the levels of the variable.

    :param chat_doc: Chat messages.
    :param grouped_msgs: List of grouped messages
    :param levels: List of levels for the variable (e.g., behaviours = ['rapport','control']).
    :param classification_model: LM used to perform NLI between grouped messages and labels.

    :return: Tuple of the matched level and the confidence of the match.
    """

    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    model_loc = str("/home/darrencook/elicit/models/"+model_name)

    classification_model = pipeline(
        "zero-shot-classification", 
        model=model_loc, # fine-tuned model -> RENAME AND ADD TO EXTRACT
        # model = 'facebook/bart-large-mnli',
        device=-1 # run on CPU
    )

    candidates = []
    for msg_str, start, end in grouped_msgs:
        msg_content = re.sub(r'(PRED|VICT): ', '', msg_str)
        
        output = classification_model(
            msg_content,
            [*levels, ""],
            multi_label=True
            )

        for i in range(len(output['labels'])):
            if output['labels'][i] == '':
                abstain_conf = output['scores'][i]       
        for i in range(len(output['labels'])):
            if output['scores'][i] > abstain_conf:
                if output['scores'][i] > filter_threshold:
                    candidates.append((output['labels'][i], output['scores'][i], start, end))


    if not candidates:
        return [Extraction.abstain()]
    compressed_candidates, context = compress(candidates)

    candidates = [(o, s)
                  for o, s in compressed_candidates.items() if s > filter_threshold]

    # create a list of Extraction for each candidate
    extractions = []
    for candidate, score in candidates:
        if candidate == "":
            continue
        extractions.append(Extraction.from_character_startend(
            chat_doc,
            candidate,
            score,
            context[candidate]["start"],
            context[candidate]["end"]
        ))

    if len(extractions) == 0:
        return [Extraction.abstain()]
    else:
        return extractions


class ChatNLILabellingFunction(CategoricalLabellingFunction):
    """
    Labelling function that compares message content with behaviour label
    """
    def __init__(self, schemas, logger, **kwargs):
        super().__init__(schemas, logger, **kwargs)
        self.entail_threshold = 0.4 # change this after doing dynamic threshold from previous chapter
        self.window_size = 5 # the number of messages (current+previous) to group


    def train(self, document_name: str, variable_name: str, extraction: Extraction):
        pass

    # def load(self) -> None:
    #     self.classifier = pipeline(
    #                         "zero-shot-classification",
    #                         model="/home/darrencook/elicit/model", # fine-tuned model -> RENAME AND ADD TO EXTRACT
    #                         # model = 'facebook/bart-large-mnli',
    #                         device=-1 # run on CPU
    #     )
    def load(self) -> None:
        pass

    def extract(self, document_name: str, variable_name: str, document_text: str) -> None:
        categories = self.get_schema("categories", variable_name)
        chat_messages = group_msgs(
            chat_doc = document_text,
            window_size= self.window_size
            )
        extractions=match_classify(
            chat_doc=document_text,
            grouped_msgs=chat_messages,
            # classification_model=self.classifier,
            model_name=variable_name,
            levels=categories,
            filter_threshold = self.entail_threshold
        )
        self.push_many(document_name, variable_name, extractions)
        
        

    @property
    def labelling_method(self) -> str:
        return "NLI Message Group Labelling Function"