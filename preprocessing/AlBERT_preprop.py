
def prepare_AlQA_data(paragraphs):
	return ". ".join(paragraphs)


def process_alqa_answers(preds, answers, chunk):
  for pred in preds:
    if pred['answers'][0]['answer'] == '[CLS]': continue
    answers[pred['id']]['answers'].append({
        'answer': pred['answers'][0]['answer'],
        'score': pred['answers'][0]['score'].item(),
        'chunk': chunk,
        'model': 'AlBERT'
    })
  return answers


def get_sim_score(model, questions, para_chunks):
  scores = []
  for que in questions:
    biencoder_ip = [[que['question'], chunk] for chunk in para_chunks]
    score = model.predict(biencoder_ip)
    scores.append(score)

  return scores
