import json

input_dir = "/data/path.json"

output_dir = "/output/dir/"
output_path = output_dir + "output.json"
model_name = "model-name"

import os
os.mkdir(output_dir)

result = []
lenths = []
with open(input_dir, 'r') as f:
    for l in f.readlines()[:]:
        try:
            j = json.loads(l)
            if "final_result" in j and j["final_result"] != None:
                result.append({"instruction": j["instruction"], "output": j["final_result"],
                            "generator": model_name,
                            "dataset": j["dataset"], "datasplit":"eval"})
                lenths.append(len(j["final_result"]))
            else:
                continue
        except Exception as e:
            print(e)
print(len(lenths))
print(sum(lenths) / len(lenths))
json.dump(result, open(output_path, 'w'))
