import numpy as np
import nltk
import os
from functools import reduce

from utils.file_handling import file_to_list, dict_to_json, json_to_dict
from models.paragraph_linking import get_tfidf


# get POS tagging for string
def get_pos_tagging(string):
    # get POS tagging for string
    pos_tagging = nltk.pos_tag(nltk.word_tokenize(string))
    pos_tagging = list(map(lambda x: x[1], pos_tagging))
    return " ".join(pos_tagging)


def hybridr_preprocessing(table, paragraphs, questions, linking_base_thresh=0.1, linking_dynamic_thresh=0.9):
    table = np.array(table)
    links = {}
    hybridR_ques = []
    hybirdR_table = {
        "url": "",
        "title": "table_0"
    }
    has_links = False

    for i, para in enumerate(paragraphs):
        links[f"link_{i}"] = para

  
    for i, que in enumerate(questions):
        hybridR_ques.append({
            "question_id" : que['id'],
            "question" : que['question'],
            "table_id" : f"table_0",
            "question_postag" : que['question']
        })


    hybirdR_table["header"] = []
    table_headers = table[0,:]
    for header in table_headers:
        hybirdR_table["header"].append([header, []])
    
    hybirdR_table["data"] = []
    for row in table[1:,:]:
        hybirdR_table["data"].append([])
        for cell in row:
            hybirdR_table["data"][-1].append([cell, []])

    link_cells = get_tfidf(table, paragraphs, linking_base_thresh, linking_dynamic_thresh)

    for cell in link_cells:
      if cell['r'] == 0:
        hybirdR_table["header"][cell['c']][1] = cell['paras']
      else:
        if len(cell['paras'])!=0: has_links = True
        hybirdR_table["data"][cell['r']-1][cell['c']][1] = cell['paras']

    
    return links, hybridR_ques, hybirdR_table, has_links


def prepare_hybridR_data(questions, paragraphs, table, question_preds):
  hybridR_questions = list(filter(lambda x: x['class']==0, question_preds))
  hybirdR_paragraphs, hybirdR_ques, hybirdR_table, has_links = hybridr_preprocessing(table, paragraphs, hybridR_questions)

  if(not has_links): 
    return has_links

  os.makedirs('models/HybridR/test_inputs/', exist_ok=True)
  dict_to_json(hybirdR_ques, 'models/HybridR/test_inputs/test.json')

  os.makedirs('inputs/request_tok', exist_ok=True)
  dict_to_json(hybirdR_paragraphs, 'inputs/request_tok/table_0.json')

  os.makedirs('inputs/tables_tok', exist_ok=True)
  dict_to_json(hybirdR_table, 'inputs/tables_tok/table_0.json')

  del hybirdR_ques
  del hybirdR_paragraphs
  del hybirdR_table

  preprocessing_main()

  # !rm -r /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  # !mkdir /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  # !cp -r /content/FinTabParse/inputs/request_tok /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  # !cp -r /content/FinTabParse/inputs/tables_tok /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  # !cp -r /content/FinTabParse/inputs/tables_tmp /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  return has_links


def process_hybridr_answers(answers, chunk):
  hybridR_pred = json_to_dict('models/HybridR/predictions.json')
  for pred in hybridR_pred:
    score = reduce(lambda x, y: x[4]*y[4], pred["nodes"])
    answers[pred['question_id']]['answers'].append({
        'answer': pred['pred'],
        'score': score,
        'chunk': chunk,
        'model': 'HybridR'
    })
  return answers