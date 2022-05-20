
def prepare_AlQA_data(paragraphs):
	return " ".join(paragraphs)


def process_alqa_answers(preds, answers, chunk):
  for pred in preds:
    if pred['answers'][0]['answer'] == '[CLS]': continue
    answers[pred['id']]['answers'].append({
        'answer': pred['answers'][0]['answer'],
        'score': pred['answers'][0]['score'],
        'chunk': chunk,
        'model': 'AlBERT'
    })
  return answers

