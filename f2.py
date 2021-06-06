
import json
import sys
from pprint import pprint

""" 
Level 2 filtering
- remove unicode characters 
- and convert to text file splitted by lines

"""

json_file = sys.argv[1]
outfile = sys.argv[2]

with open(f"./tweets/{json_file}.json", "r") as file:
    data = json.loads(file.read())
    # pprint(data)

with open(f"{outfile}.txt", "w") as j:
    for key in data:
        try:
            j.write(f"{data[key]}\n")
        except:
            continue
