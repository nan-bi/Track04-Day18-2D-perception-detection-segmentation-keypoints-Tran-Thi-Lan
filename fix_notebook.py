import json
import re

with open("d:/AI VIN/lab_2d_perception_student_completed.ipynb", "r", encoding="utf8") as f:
    nb = json.load(f)

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        if "def mask_to_yolo_seg" in src:
            src = re.sub(r"cnt = \.\.\.\s*#.*", "cnt = max(contours, key=cv2.contourArea)                      # 🧩", src)
            src = re.sub(r"norm = \.\.\.\s*#.*", "norm = cnt / np.array([w, h])                                 # (K,2)  🧩", src)
            
            lines = src.split('\n')
            cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("d:/AI VIN/lab_2d_perception_student_completed.ipynb", "w", encoding="utf8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
