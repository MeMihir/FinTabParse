import numpy as np
import nltk

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


def hybridr_preprocessing(table, paragraphs, questions):
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

    link_cols, link_rows = get_tfidf(table, paragraphs, [0], [0])
    for i, cols in enumerate(link_cols):
        for col in cols:
            hybirdR_table["header"][col][1].append(f"link_{i}")
    
    for i, rows in enumerate(link_rows):
        for row in rows:
            hybirdR_table["data"][row][0][1].append(f"link_{i}")
    
    return links, hybridR_ques, hybirdR_table