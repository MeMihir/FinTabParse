import pandas as pd
from utils.file_handling import dict_to_json, json_to_dict
import numpy as np


def aggregate_AlBERT(answers):
    answers = list(filter(lambda x: not isAnsEmpty(x["answer"]), answers))
    indx = max(answers, key=lambda x: x["score"])
    return indx

def isAnsEmpty(ans):
    if(type(ans)==type(1) or type(ans)==type(1.0)): return False
    return len(ans)==0 or ans=="[]"

def aggregate_TagOp(answers):
    ans = list(filter(lambda x: not isAnsEmpty(x["answer"][0]), answers))
    return ans

def preprocess_tapas(answer):
    if not ' > ' in answer:
        return ["NONE", answer.split(', ')]
    answer = answer.split(' > ')
    aggr = answer[0]
    answer = answer[1].split(', ')
    answer = map(lambda x: x.replace(',', ''), answer)
    answer = list(map(lambda x: float(x) if x.replace('.', '', 1).isdigit() else x, answer))
    return [aggr, answer]


def aggregate_TaPaS(answers):
    answers = list(map(lambda x: {
        "answer":  preprocess_tapas(x),
        "score": 1.7390611171722412,
        "chunk": 2,
        "model": "TaPaS"
    }, answers))
    return answers

def aggregate_answers(answers_path, op_path):
    answers = json_to_dict(answers_path)
    agg_ans = {}
    for que in answers.keys():
        question = answers[que]
        agg_ans[que] = {
            "question" : question["question"],
            "id" : que,
            "class" : question['class'],
            "answers" : []
        }

        models = set([ans['model'] for ans in question['answers']])
        for model in models:
            model_ans = list(filter(lambda x: x['model']==model, question["answers"]))
            if model == 'AlBERT': 
                indx = aggregate_AlBERT(model_ans)
                agg_ans[que]["answers"].append(indx)
            elif model == "TaPaS":
                indx = aggregate_AlBERT(model_ans)
                agg_ans[que]["answers"].append(indx)
            # elif model == "TagOp":
            #     ans = aggregate_TagOp(model_ans)
            #     agg_ans[que]["answers"].extend(ans)

    dict_to_json(agg_ans, op_path)

    return agg_ans
        