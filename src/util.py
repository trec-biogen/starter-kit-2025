import json
import csv

def load_json(path):
    with open(path, 'r') as rfile:
        return json.load(rfile)

def save_json(data, path):
    with open(path, 'w') as wfile:
        json.dump(data, wfile, indent=4)
def save_csv(data_list, path_to_save, delimiter=','):
    keys = data_list[0].keys()

    with open(path_to_save, 'w', newline='') as output_file:
        dict_writer = csv.DictWriter(output_file, keys, delimiter=delimiter)
        dict_writer.writeheader()
        dict_writer.writerows(data_list)
