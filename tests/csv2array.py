import json
import csv

def dict_to_json(tag_op_input, json_file):
    with open(json_file, 'w') as f:
        json.dump(tag_op_input, f, indent=2)

def main(ip, op):
    with open(ip) as CSVData:
        data = list(csv.reader(CSVData))
    dict_to_json(data, op)

if __name__ == "__main__":
    ip = input()
    op = input()
    main(ip, op)