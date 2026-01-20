import numpy as np
import torch
import ultralytics.utils.ops as ops
from ultralytics import YOLO
import cv2

img_path = r"D:\other\20260104\AutoSize\data\120.jpg"
output_path=r"D:\other\20260104\AutoSize\train_model\120.predict.jpg"
trained_model=r'D:\other\20260104\AutoSize\train_model\trained_models29\weights\best.pt'
margin_set=0.1                    # 扩大检测框
ref_obj_name = "10cm_plate1"       # 参考物体编号
ref_real_area = 16.80          # 参考物体实际面积
area_decimal = 2              # 面积保留小数位数（0=整数，2=两位小数）



def crop_mask(masks, boxes, margin=margin_set):
    if masks is None or boxes is None:
        return masks
    n, h, w = masks.shape
    device = masks.device
    boxes = boxes.to(device)
    x1_f, y1_f, x2_f, y2_f = boxes.T
    bw = (x2_f - x1_f)
    bh = (y2_f - y1_f)
    x1_f = (x1_f - margin * bw).clamp(0, w - 1)
    y1_f = (y1_f - margin * bh).clamp(0, h - 1)
    x2_f = (x2_f + margin * bw).clamp(0, w - 1)
    y2_f = (y2_f + margin * bh).clamp(0, h - 1)
    x1_i = x1_f.to(torch.int64)
    y1_i = y1_f.to(torch.int64)
    x2_i = (x2_f.to(torch.int64)).clamp(max=w)
    y2_i = (y2_f.to(torch.int64)).clamp(max=h)
    out = torch.zeros_like(masks)
    for i in range(n):
        x1i = int(x1_i[i].item())
        x2i = int(x2_i[i].item())
        y1i = int(y1_i[i].item())
        y2i = int(y2_i[i].item())
        if x2i <= x1i or y2i <= y1i:
            continue
        out[i, y1i:y2i, x1i:x2i] = masks[i, y1i:y2i, x1i:x2i]
    return out
ops.crop_mask = crop_mask
model = YOLO(trained_model)
results = model.predict(
    source=img_path,
    device="cpu",
    retina_masks=True,
    conf=0.5,
    iou=0.5)
result = results[0].cpu()
annotated_img = result.plot(boxes=False)
dic_pixel_count = {}
dic_real_area = {}
obj_name_list = []
sorted_boxes = []

if len(result.boxes) == 0:
    print("未检测到任何目标！")
    cv2.imwrite(output_path, annotated_img)
else:
    class_ids = result.boxes.cls.int().numpy()
    class_names = result.names
    boxes_xyxy = result.boxes.xyxy.cpu().numpy()
    masks = result.masks.data
    per_target_pixel_nums = torch.sum(masks, dim=[1,2]).numpy()
    cx = (boxes_xyxy[:, 0] + boxes_xyxy[:, 2]) / 2.0
    cy = (boxes_xyxy[:, 1] + boxes_xyxy[:, 3]) / 2.0
    y1_list = boxes_xyxy[:, 1]
    y2_list = boxes_xyxy[:, 3]
    all_targets = []
    for i in range(len(class_ids)):
        all_targets.append({
            "cls_id": class_ids[i],
            "cls_name": class_names[class_ids[i]],
            "pixel_num": per_target_pixel_nums[i],
            "box": boxes_xyxy[i],
            "cx": cx[i],
            "cy": cy[i],
            "y1": y1_list[i],
            "y2": y2_list[i]
        })
    rows = []
    for target in all_targets:
        target_y1, target_y2 = target["y1"], target["y2"]
        added = False
        for row in rows:
            row_y1 = row[0]["y1"]
            row_y2 = row[0]["y2"]
            if target_y1 < row_y2 and row_y1 < target_y2:
                row.append(target)
                added = True
                break
        if not added:
            rows.append([target])
    for row in rows:
        row.sort(key=lambda t: t["cx"])
    rows_with_avg_cy = []
    for row in rows:
        row_avg_cy = np.mean([t["cy"] for t in row])
        rows_with_avg_cy.append( (row_avg_cy, row) )
    rows_with_avg_cy.sort(key=lambda x: x[0])
    sorted_rows = [row for (avg_cy, row) in rows_with_avg_cy]
    final_sorted_targets = []
    for row in sorted_rows:
        final_sorted_targets.extend(row)
    class_counter = {}
    for row_idx, row in enumerate(sorted_rows):
        row_avg_cy = np.mean([t["cy"] for t in row])
    for target in final_sorted_targets:
        cls_name = target["cls_name"]
        class_counter[cls_name] = class_counter.get(cls_name, 0) + 1
        obj_name = f"{cls_name}{class_counter[cls_name]}"
        obj_name_list.append(obj_name)
        dic_pixel_count[obj_name] = str(int(target["pixel_num"]))
        sorted_boxes.append(target["box"])
    if ref_obj_name not in dic_pixel_count:
        print(f"⚠️  警告：参考物体【{ref_obj_name}】不存在！")
        print(f"当前有效编号：{list(dic_pixel_count.keys())}")
    else:
        ref_pixel_num = int(dic_pixel_count[ref_obj_name])
        pixel2area = ref_real_area / ref_pixel_num

        for obj_name, pixel_str in dic_pixel_count.items():
            pixel_num = int(pixel_str)
            real_area = round(pixel_num * pixel2area, area_decimal)
            dic_real_area[obj_name] = real_area
    print("实际面积：")
    print(dic_real_area)
    sorted_boxes = torch.tensor(np.array(sorted_boxes))
    margin = margin_set
    H, W = annotated_img.shape[:2]
    x1, y1, x2, y2 = sorted_boxes[:,0], sorted_boxes[:,1], sorted_boxes[:,2], sorted_boxes[:,3]
    bw = x2 - x1
    bh = y2 - y1
    x1e = (x1 - margin * bw).clamp(0, W-1).int()
    y1e = (y1 - margin * bh).clamp(0, H-1).int()
    x2e = (x2 + margin * bw).clamp(0, W-1).int()
    y2e = (y2 + margin * bh).clamp(0, H-1).int()
    for i in range(len(sorted_boxes)):
        cv2.rectangle(
            annotated_img,
            (x1e[i].item(), y1e[i].item()),
            (x2e[i].item(), y2e[i].item()),
            (0, 255, 0), 2
        )
        current_obj = obj_name_list[i]
        if current_obj in dic_real_area:
            target_label = f"{current_obj}: {dic_real_area[current_obj]}"
        else:
            target_label = current_obj
        label_pos = (x1e[i].item(), y1e[i].item() - 10)
        if label_pos[1] < 10:
            label_pos = (x1e[i].item(), y1e[i].item() + 20)
        cv2.putText(annotated_img, target_label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 3)
        cv2.putText(annotated_img, target_label, label_pos,
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
    cv2.imwrite(output_path, annotated_img)
torch.cuda.empty_cache()
