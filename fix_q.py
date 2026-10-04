import json
import re

with open("d:/AI VIN/lab_2d_perception_student_completed.ipynb", "r", encoding="utf8") as f:
    nb = json.load(f)

replacements = {
    'Q1 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q1 = """\n(a) 640/32=20, 640/16=40, 640/8=80. 80*60 + 40*30 + 20*15 = 6300. Ít hơn 8400 vì ảnh bus không vuông (640x480 thay vì 640x640), số lượng giảm theo diện tích.\n(b) Nhãn gốc là xywh, nhưng bị đọc nhầm là cxcywh (tâm). Do đó, góc trên trái bị dịch đi một nửa kích thước box.\n"""',
    'Q2 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q2 = """\nNMS-free giúp giảm đáng kể latency hậu kỳ. Khi conf thấp và cảnh đông người, số lượng box cần NMS lớn làm tăng latency. NMS-free xử lý được trên CPU/NPU nhanh vì không phụ thuộc cấu trúc đồ thị tuần tự phức tạp của NMS.\n"""',
    'Q3 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q3 = """\n(a) Với ngưỡng 0.7, hai box có IoU 0.75 > 0.7 nên box điểm thấp bị loại -> sót người. Tăng ngưỡng dễ đếm thừa (trùng lặp). Giảm ngưỡng dễ đếm thiếu (loại nhầm).\n(b) Head one-to-many tăng số lượng positive samples lúc train, giúp mô hình hội tụ nhanh và giàu thông tin hơn.\n"""',
    'Q4 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q4 = """\n(a) Đếm thiếu: hai người sát nhau bị dính chung thành 1 vùng semantic. Đếm thừa: một người bị vật che chia đôi người thành hai mảnh rời. Panoptic segmentation kết hợp instance và semantic, tách bạch từng người riêng.\n(b) Semantic gán nhầm vào train (tàu hoả). Vì xe buýt và tàu hoả COCO có hình dạng khá giống nhau (dạng khối hộp) khi chụp mờ hoặc bị che một phần.\n"""',
    'Q5 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q5 = """\n(a) YOLO26-seg cho biên mịn ở object lớn nhờ resolution mask toàn cục. Mask R-CNN cắt 28x28 cho mọi object, upsample object lớn sẽ bị răng cưa.\n(b) Bounding box YOLO có thể lệch nhỏ với mask RCNN, gây lỗi khớp nhãn.\n"""',
    'Q6 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q6 = """\n(a) Điểm A (trên nón) ra cái nón. Prompt bằng box an toàn hơn vì khoanh vùng giới hạn diện tích (scale) của object, tránh việc SAM nhầm lấy mỗi nón thay vì cả người hoặc ngược lại.\n(b) SAM thích hợp khi không cần real-time, hoặc để tự tạo nhãn. YOLO26-seg triển khai khi cần real-time trên biên (edge).\n"""',
    'Q7 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q7 = """\nNó vẽ ở mép ảnh hoặc (0,0). Argmax tìm điểm sáng nhất trên heatmap, nếu không có khớp nào, nó lấy nhiễu ở biên. Vì vậy cần độ tin cậy (confidence) để lọc các keypoint không thực sự xuất hiện.\n"""',
    'Q8 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q8 = """\nMắt cần độ chính xác cực cao (sigma nhỏ) vì sai lệch vài pixel là trượt ra khỏi vùng mắt. Khi người ở xa nhỏ (80 px), độ tin cậy từ mắt rất khó đo đạc nên hệ thống cần rely vào các khớp như vai/hông (sigma lớn hơn).\n"""',
    'Q9 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q9 = """\n(a) Báo nhầm: cúi nhặt đồ hoặc thắt dây giày. Bỏ sót: ngã nhào hướng thẳng vào ống kính (góc 2D không đổi nhiều).\n(b) Box phát hiện người bị méo aspect ratio, mô hình thiếu aug rotation lúc train.\n(c) Cần theo dõi temporal (chuỗi frame) và optical flow để bắt chuyển động.\n"""',
    'Q10 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q10 = """\nGiữ nguyên thì mAP val lật gương sẽ giảm vì điểm bên phải của hổ sẽ thành điểm bên trái trên ảnh lật, cần flip_idx để swap tên khớp. Khi triển khai, lỗi này khiến thuật toán cảnh báo nhầm chân.\n"""',
    'Q11 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q11 = """\n1. Lỗi nhầm chân trái/phải (đảo bên): Khắc phục bằng data aug lật gương (flip_idx) và 3D info. \n2. Bị che khuất (occlusion): Khắc phục bằng cách gán cờ vis=0 cho keypoint bị che.\n"""',
    'Q12 = """\n(viết câu trả lời của bạn ở đây)\n"""': 'Q12 = """\nKhông hợp lý vì độ lệch tự nhiên của đuôi lớn hơn mũi rất nhiều. Để tính sigma custom, cần yêu cầu nhiều người gán nhãn, tính phương sai (variance) vị trí mỗi khớp, rồi thiết lập sigma tỉ lệ với variance đó.\n"""'
}

for cell in nb["cells"]:
    if cell["cell_type"] == "code":
        src = "".join(cell["source"])
        for k, v in replacements.items():
            if k in src:
                src = src.replace(k, v)
        
        lines = src.split('\n')
        cell['source'] = [line + '\n' for line in lines[:-1]] + [lines[-1]] if lines else []

with open("d:/AI VIN/lab_2d_perception_student_completed.ipynb", "w", encoding="utf8") as f:
    json.dump(nb, f, indent=1, ensure_ascii=False)
