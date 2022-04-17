import json

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