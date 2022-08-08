import pandas as pd
from utils.file_handling import dict_to_json, json_to_dict
import numpy as np
import copy
from collections.abc import Iterable   # import directly from collections for Python < 3.3

# Utility function to aggregate answers from multiple models

def isAnsEmpty(ans):
    if(type(ans)==type(1) or type(ans)==type(1.0)): return False
    return len(ans)==0 or ans=="[]"


def isFiltAnsEmpty(ans):
    if(type(ans)==type(1) or type(ans)==type(1.0)): return False
    try:
        return len(ans)==0 or ans=="[]" or len(ans[0])==0
    except TypeError:
        return False

# Aggregate answers from multiple models

def aggregate_AlBERT(answers):
    answers = list(filter(lambda x: not isAnsEmpty(x["answer"]), answers))
    indx = max(answers, key=lambda x: x["score"])
    return indx

def aggregate_TagOp(answers):
    ans = list(filter(lambda x: not isAnsEmpty(x["answer"][0]), answers))
    return ans

def preprocess_tapas(answer):
    ans = answer['answer']
    if not ' > ' in ans:
        answer['answer'] = ["NONE", ans.split(', ')]
    else:
        ans = ans.split(' > ')
        aggr = ans[0]
        ans = ans[1].split(', ')
        ans = map(lambda x: x.replace(',', ''), ans)
        ans = list(map(lambda x: float(x) if x.replace('.', '', 1).isdigit() else x, ans))
        answer['answer'] = [aggr, ans]
    return answer


def aggregate_tapas(ans_type, answers):
  scores = []
  agg_answers = []

  if ans_type == 'NONE': 
    agg_ans = list(filter(lambda x: x['answer'][0]=='NONE', answers))
    if len(agg_ans)==0: return (0, [], 0)
    
    agg_ans = aggregate_AlBERT(agg_ans)
    return (agg_ans['score'], agg_ans['answer'][1], len(answers))

  for ans in answers:
    if ans['answer'][0] != ans_type: continue
    scores.append(ans['score'])
    agg_answers.extend(ans['answer'][1])
  agg_answers = list(filter(lambda x: not isinstance(x, Iterable) or len(x)!=0, agg_answers))
  return (0, agg_answers, 0) if len(scores)==0 else (sum(scores)/len(scores), agg_answers, len(scores))

# Main function to aggregate answers from multiple models

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
                answers[que]['answers'] =  list(map(preprocess_tapas, answers[que]['answers']))
                score_none, agg_none, len_none = aggregate_tapas('NONE', answers[que]['answers'])
                score_avg, agg_avg, len_avg = aggregate_tapas('AVERAGE', answers[que]['answers'])
                score_count, agg_count, len_count = aggregate_tapas('COUNT', answers[que]['answers'])
                score_sum, agg_sum, len_sum = aggregate_tapas('SUM', answers[que]['answers'])
                
                if(len_none >= max([len_avg, len_count, len_sum])): agg_ans[que]['answers'] = {'answer': agg_none, 'score': score_none, 'model': 'TaPaS', 'aggregation': 'NONE'}
                if(len_avg >= max([len_none, len_count, len_sum])): agg_ans[que]['answers'] = {'answer': agg_avg, 'score': score_avg, 'model': 'TaPaS', 'aggregation': 'AVG'}
                if(len_count >= max([len_avg, len_none, len_sum])): agg_ans[que]['answers'] = {'answer': agg_count, 'score': score_count, 'model': 'TaPaS', 'aggregation': 'COUNT'}
                if(len_sum >= max([len_avg, len_count, len_none])): agg_ans[que]['answers'] = {'answer': agg_sum, 'score': score_sum, 'model': 'TaPaS', 'aggregation': 'SUM'}

    dict_to_json(agg_ans, op_path)

    return agg_ans
        
def filter_sort(input, filt, sort, output):
  answers = json_to_dict(input)
  sorted_ans = copy.deepcopy(answers)

  for que in answers:
    filt_ans = list(filter(
        lambda x: x['model'] == filt and not isFiltAnsEmpty(x["answer"]), answers[que]["answers"]
    ))
    sorted_ans[que]["answers"] = sorted(filt_ans, key=lambda x: x[sort], reverse=True)

  dict_to_json(sorted_ans, output)
  return sorted_ans