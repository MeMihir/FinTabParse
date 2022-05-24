import pandas as pd

def prepare_tapas_data(table, questions):
  TaPaSQA = pd.DataFrame.from_records(table[1:], columns=table[0])
  queries = map(lambda x: x['question'], questions)
  return TaPaSQA, list(queries)

def process_tapas_answers(predicted_answer_coordinates, predicted_aggregation_indices, table, answers, chunk, scores):
  id2aggregation = {0: "NONE", 1: "SUM", 2: "AVERAGE", 3: "COUNT"}
  aggregation_predictions_string = [id2aggregation[x] for x in predicted_aggregation_indices]

  ans = []
  for coordinates in predicted_answer_coordinates:
      if len(coordinates) == 1:
          ans.append(table.iat[coordinates[0]])
      else:
          cell_values = []
          for coordinate in coordinates:
              cell_values.append(table.iat[coordinate])
          ans.append(", ".join(cell_values))
  for k, answer, predicted_agg, score in zip(answers.keys(), ans, aggregation_predictions_string, scores):
      answers[k]['answers'].append({
          'answer': answer if predicted_agg == "NONE" else f"{predicted_agg} > {answer}",
          'score': score.item(),
          'chunk': chunk,
          'model': 'TaPaS'
      })

  return answers
