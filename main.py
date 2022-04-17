from transformers import  BertTokenizer
import torch

from utils import file_to_list, json_to_dict
from models.questions_classifier import BertClassifier

def get_inputs():
	questions = file_to_list('/content/questions.txt')
	paragraphs = file_to_list('/content/paragraphs.txt')
	table = json_to_dict('/content/table.json')

	return questions, paragraphs, table

def load_classifier_model():
	tokenizer = BertTokenizer.from_pretrained('bert-base-cased')
	labels = {'span':0, 'math':1}
	classes = {0: 'span', 1: 'math'}
	question_classifier = BertClassifier()
	question_classifier.load_state_dict(torch.load('/content/weights/questions_classifier', map_location=torch.device('cpu')))
