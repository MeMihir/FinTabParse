from utils.file_handling import file_to_list, dict_to_json, json_to_dict

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
            "uid": que['id'],
            "order": i,
            "question": que['question'],
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

def prepare_tagop_data(questions, paragraphs, table, question_preds):
  tagop_questions = list(filter(lambda x: x['class']==1, question_preds))
  tag_op_input = tagop_preprocessing(table, paragraphs, tagop_questions)
  dict_to_json([tag_op_input], 'models/TAT-QA/dataset_tagop/tatqa_dataset_dev.json')
  return tag_op_input


def process_tagop_answers(answers, chunk):
  tagop_pred = json_to_dict('models/TAT-QA/tag_op/pred_result_on_dev.json')
  for k in tagop_pred.keys():
    answers[k]['answers'].append({
        'answer': tagop_pred[k],
        'chunk': chunk,
        'model': 'TagOp'
    })
  return answers