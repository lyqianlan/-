import os
import torch
from torch.utils.data import DataLoader
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from collections import Counter

# 从训练脚本中导入相同的类和函数
from train import (
    build_resnet50_fpn_detector,
    COCOVehicleDataset,
    collate_fn
)


@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt, debug=False):
    model.eval()
    results = []
    # 与训练代码完全一致的映射：模型标签 1~5 -> COCO原始ID
    mapping = {1: 3, 2: 4, 3: 6, 4: 7, 5: 8}
    label_counter = Counter()

    for batch_idx, batch in enumerate(data_loader):
        images, targets, img_ids = batch
        images = [img.to(device) for img in images]

        outputs = model(images)

        for img_idx, output in enumerate(outputs):
            img_id = img_ids[img_idx]
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            label_counter.update(labels)

            for box, score, label in zip(boxes, scores, labels):
                # 跳过背景（label=0）和未定义标签
                if label == 0 or label not in mapping:
                    continue
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                if w <= 0 or h <= 0:
                    continue
                results.append({
                    'image_id': int(img_id),
                    'category_id': mapping[label],
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })

    if debug:
        print("\n=== Debug Info ===")
        print(f"Total predictions: {len(results)}")
        if results:
            print(f"Sample predictions (first 3): {results[:3]}")
        print(f"Label distribution in predictions: {label_counter}")

    if not results:
        print("No predictions, skip evaluation.")
        return None

    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return {
        'AP': stats[0],
        'AP50': stats[1],
        'AP75': stats[2],
        'AP_S': stats[3],
        'AP_M': stats[4],
        'AP_L': stats[5]
    }


def main():
    # 配置路径（请根据实际情况修改）
    data_root = './coco2017'
    val_img_dir = os.path.join(data_root, 'val2017')
    val_ann_file = os.path.join(data_root, 'annotations', 'instances_val2017.json')
    model_path = 'best_vehicle_detector.pth'   # 或 'final_vehicle_detector.pth'
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载验证集（与训练代码完全相同）
    val_dataset = COCOVehicleDataset(
        root=val_img_dir,
        ann_file=val_ann_file,
        is_train=False
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=2,
        shuffle=False,
        num_workers=4,
        collate_fn=collate_fn,
        pin_memory=True
    )
    coco_gt = COCO(val_ann_file)

    # 构建模型并加载权重
    num_classes = len(val_dataset.VALID_IDS) + 1   # 6（背景+5类）
    model = build_resnet50_fpn_detector(num_classes=num_classes, pretrained=False)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)

    print("Evaluating on validation set...")
    eval_results = evaluate(model, val_loader, device, coco_gt, debug=True)

    if eval_results:
        print("\n========== Evaluation Results ==========")
        print(f"AP (0.5:0.95): {eval_results['AP']:.4f}")
        print(f"AP50 (0.5):     {eval_results['AP50']:.4f}")
        print(f"AP75 (0.75):    {eval_results['AP75']:.4f}")
        print(f"AP small:       {eval_results['AP_S']:.4f}")
        print(f"AP medium:      {eval_results['AP_M']:.4f}")
        print(f"AP large:       {eval_results['AP_L']:.4f}")
    else:
        print("Evaluation failed.")


if __name__ == '__main__':
    main()