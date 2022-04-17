from transformers import  BertTokenizer
import torch

from utils.file_handling import file_to_list, json_to_dict
from models.questions_classifier import BertClassifier

def get_inputs(questions_path, paragraphs_path, table_path):
	questions = file_to_list(questions_path)
	paragraphs = file_to_list(paragraphs_path)
	table = json_to_dict(table_path)
	return questions, paragraphs, table


def question_classifier_model(questions, model_path):
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
 	
	question_classifier = BertClassifier()
	question_classifier.load_state_dict(torch.load(model_path, map_location=torch.device('cpu')))
 
	question_infs = list(map(lambda question_input: tokenizer(question_input, padding='max_length', max_length = 512, truncation=True, return_tensors="pt"), questions))
	question_preds = []

	for que_inf in question_infs:
		question_id = que_inf['input_ids']
		question_mask = que_inf['attention_mask'].squeeze(1)
		pred = question_classifier(question_id, question_mask)
		question_preds.append(pred.argmax(dim=1).numpy()[0])
	
	return question_preds