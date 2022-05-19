import imp
import numpy as np
import nltk
from utils.file_handling import file_to_list, dict_to_json, json_to_dict

from models.paragraph_linking import get_tfidf

def tagop_preprocessing(table, paragraphs, questions):
    tag_op_input = {}
    tag_op_input["table"] = {
        "id": "1",
        "table": table
    }

    tag_op_input["paragraphs"] = []
    for i, paras in enumerate(paragraphs):
        tag_op_input["paragraphs"].append({
            "uid": f"para_{i}",
            "order": i,
            "text": paras
        })

    tag_op_input["questions"] = []
    for i, que in enumerate(questions):
        tag_op_input["questions"].append({
            "uid": f"que_{i}",
            "order": i,
            "question": que,
            "answer": "",
            "derivation": "",
            "answer_type": "",
            "answer_from": "",
            "facts": [],
            "mapping": {
                "table": []
            },
            "rel_paragraphs": [],
            "req_comparison": False,
            "scale": ""
        })
    
    return tag_op_input

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

    for i, para in enumerate(paragraphs):
        links[f"link_{i}"] = para

  
    for i, que in enumerate(questions):
        hybridR_ques.append({
            "question_id" : f"que_{i}",
            "question" : que,
            "table_id" : f"table_0",
            "question_postag" : get_pos_tagging(que)
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
        hybirdR_table["data"][cell['r']-1][cell['c']][1] = cell['paras']

    
    return links, hybridR_ques, hybirdR_table

def prepare_AlQA_data(paragraphs):
	return " ".join(paragraphs)


def get_inputs(questions_path, paragraphs_path, table_path):
	questions = file_to_list(questions_path)
	paragraphs = file_to_list(paragraphs_path)
	table = json_to_dict(table_path)
	return questions, paragraphs, table

