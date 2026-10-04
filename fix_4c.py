import json
import re

with open("d:/AI VIN/lab_2d_perception_student_completed.ipynb", "r", encoding="utf8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "RUN_4C = False" in src:
            src = src.replace("RUN_4C = False", "RUN_4C = True")
        if "TRAIN_IDENTITY = False" in src:
            src = src.replace("TRAIN_IDENTITY = False", "TRAIN_IDENTITY = True")
            
        lines = src.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("d:/AI VIN/lab_2d_perception_student_completed.ipynb", "w", encoding="utf8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
