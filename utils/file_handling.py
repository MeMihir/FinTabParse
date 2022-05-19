import json
import shutil
import os

def json_to_dict(json_file):
  with open(json_file, 'r') as f:
    data = json.load(f)
  return data

def file_to_list(file_name):
  """
  Reads a file and returns a list of lines
  """
  with open(file_name, 'r') as f:
    data = f.readlines()
  return list(map(lambda x: x.replace('\n', ''), data))

def dict_to_json(tag_op_input, json_file):
    with open(json_file, 'w') as f:
        json.dump(tag_op_input, f)

def list_to_file(list_data, file_name):
  """
  Writes a list to a file
  """
  with open(file_name, 'w') as f:
    for line in list_data:
      if(type(line)==type(" ")):
        f.write(line + '\n')

def read_inputs():
  inputs_path = '/content/FinTabParse/inputs/'
  paragraphs_path = os.path.join(inputs_path, 'paragraphs.txt')
  questions_path = os.path.join(inputs_path, 'questions.txt')
  table_path = os.path.join(inputs_path, 'table.json')

  questions, paragraphs, table = get_inputs(questions_path, paragraphs_path, table_path)
  
  return questions, paragraphs, table


def write_output(tag_op_input, AlQA, dataclass):
  tagop_pred = json_to_dict('models/TAT-QA/tag_op/pred_result_on_dev.json')
  shutil.move('models/TAT-QA/tag_op/pred_result_on_dev.json', 'outputs/tag_op_pred.json')

  tagop_answers = []
  for que in tag_op_input['questions']:
    tagop_answers.append({
        'id': que['uid'],
        'question': que['question'],
        'answer': f"{tagop_pred[que['uid']][0]} {tagop_pred[que['uid']][1]}"
    })
  dict_to_json(tagop_answers, 'outputs/tagop_answers.json')

  shutil.move('models/HybridR/predictions.intermediate.json', 'outputs/')
  shutil.move('models/HybridR/predictions.json', 'outputs/hybridR_predictions.json')
  hybridR_pred = json_to_dict('outputs/hybridR_predictions.json')

  hybridR_answers = []
  for que in hybridR_pred:
    hybridR_answers.append({
        "question": que["question"],
        'id': que['question_id'],
        'answer': que['pred']
    })
  dict_to_json(hybridR_answers, 'outputs/hybridR_answers.json')

def prepare_hybridR_data(questions, paragraphs, table, question_preds):
  hybridR_questions = list(map(lambda x: x[1], filter(lambda x: x[0]==0, zip(question_preds, questions))))
  hybirdR_paragraphs, hybirdR_ques, hybirdR_table = hybridr_preprocessing(table, paragraphs, hybridR_questions)

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

  !rm -r /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  !mkdir /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  !cp -r /content/FinTabParse/inputs/request_tok /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  !cp -r /content/FinTabParse/inputs/tables_tok /content/FinTabParse/models/HybridR/WikiTables-WithLinks/
  !cp -r /content/FinTabParse/inputs/tables_tmp /content/FinTabParse/models/HybridR/WikiTables-WithLinks/

def prepare_tagop_data(questions, paragraphs, table, question_preds):
  tagop_questions = list(map(lambda x: x[1], filter(lambda x: x[0]==1, zip(question_preds, questions))))
  tag_op_input = tagop_preprocessing(table, paragraphs, tagop_questions)
  dict_to_json([tag_op_input], '/content/FinTabParse/models/TAT-QA/dataset_tagop/tatqa_dataset_dev.json')
  return tag_op_input