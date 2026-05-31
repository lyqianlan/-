
import os
import torch
import torch.nn as nn
import torchvision
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torch.utils.data import DataLoader, Dataset
from torch.cuda.amp import autocast, GradScaler
import numpy as np
import cv2
import pickle
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

# ----------------------------- 1. 模型构建 -----------------------------
def build_resnet50_fpn_detector(num_classes=6, pretrained=True):
    """
    构建 Faster R-CNN 检测器，骨干网络为 ResNet50+FPN
    Args:
        num_classes: 类别数（包含背景）
        pretrained: 是否使用 ImageNet 预训练权重
    Returns:
        model: torch.nn.Module
    """
    backbone = torchvision.models.detection.backbone_utils.resnet_fpn_backbone(
        backbone_name='resnet50',
        pretrained=pretrained,
        trainable_layers=3
    )
    # 锚框生成器：适配 FPN 的 5 个层级 (P2-P6)
    anchor_sizes = ((32,), (64,), (128,), (256,), (512,))
    aspect_ratios = ((0.5, 1.0, 2.0),) * len(anchor_sizes)
    anchor_generator = AnchorGenerator(sizes=anchor_sizes, aspect_ratios=aspect_ratios)

    roi_pooler = torchvision.ops.MultiScaleRoIAlign(
        featmap_names=['0', '1', '2', '3'],
        output_size=7,
        sampling_ratio=2
    )

    model = FasterRCNN(
        backbone=backbone,
        num_classes=num_classes,
        rpn_anchor_generator=anchor_generator,
        box_roi_pool=roi_pooler,
        min_size=800,
        max_size=1333
    )
    return model


# ----------------------------- 2. COCO 车辆数据集 -----------------------------
class COCOVehicleDataset(Dataset):
    """
    从 COCO 中提取车辆类别 (car, truck, bus, motorcycle, train)
    """
    VEHICLE_CATEGORIES = {
        3: 'car',      # 类别ID映射
        4: 'motorcycle',
        6: 'bus',
        7: 'train',
        8: 'truck'
    }
    VALID_IDS = list(VEHICLE_CATEGORIES.keys())   # [3,4,6,7,8]

    def __init__(self, root, ann_file, transforms=None, is_train=True):
        self.root = root
        self.coco = COCO(ann_file)
        self.transforms = transforms
        self.is_train = is_train

        # 获取所有包含车辆类别的图像ID
        self.img_ids = set()
        for cat_id in self.VALID_IDS:
            self.img_ids.update(self.coco.getImgIds(catIds=[cat_id]))
        self.img_ids = sorted(list(self.img_ids))

        # 类别映射: COCO 原始ID -> 连续ID (0~4)
        self.contiguous_id = {orig: idx for idx, orig in enumerate(self.VALID_IDS)}
        self.num_classes = len(self.VALID_IDS) + 1   # +背景

    def __len__(self):
        return len(self.img_ids)

    def __getitem__(self, idx):
        img_id = self.img_ids[idx]
        img_info = self.coco.loadImgs([img_id])[0]
        img_path = os.path.join(self.root, img_info['file_name'])
        image = cv2.imread(img_path)
        if image is None:
            raise FileNotFoundError(f"Image not found: {img_path}")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # 获取该图像下的车辆标注
        ann_ids = self.coco.getAnnIds(imgIds=[img_id], catIds=self.VALID_IDS)
        annotations = self.coco.loadAnns(ann_ids)

        boxes = []
        labels = []
        for ann in annotations:
            x, y, w, h = ann['bbox']
            # 转换 bbox 为 [x1, y1, x2, y2]
            boxes.append([x, y, x + w, y + h])
            labels.append(self.contiguous_id[ann['category_id']])

        if len(boxes) == 0:
            # 如果没有有效标注（理论上数据集保证有），返回空张量
            boxes = torch.zeros((0, 4), dtype=torch.float32)
            labels = torch.zeros((0,), dtype=torch.int64)
        else:
            boxes = torch.as_tensor(boxes, dtype=torch.float32)
            labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {'boxes': boxes, 'labels': labels}

        # 简单的归一化 (不做数据增强，若需要可自行添加)
        image = torch.from_numpy(image).permute(2, 0, 1).float() / 255.0

        if self.is_train:
            return image, target
        else:
            # 验证时额外返回 img_id 用于 COCO 评估
            return image, target, img_id


# ----------------------------- 3. DataLoader 的 collate_fn （全局函数解决 pickle 问题）-----------------------------
def collate_fn(batch):
    """将 batch 中的图像和标注分别打包成元组"""
    if len(batch[0]) == 2:   # train mode
        images, targets = zip(*batch)
        return list(images), list(targets)
    else:                    # val mode
        images, targets, img_ids = zip(*batch)
        return list(images), list(targets), list(img_ids)


# ----------------------------- 4. 训练函数 (支持 AMP) -----------------------------
def train_one_epoch(model, optimizer, data_loader, device, epoch, scaler):
    model.train()
    total_loss = 0.0
    num_batches = len(data_loader)

    for batch_idx, batch in enumerate(data_loader):
        images, targets = batch
        images = [img.to(device) for img in images]
        targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

        with autocast():
            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())

        optimizer.zero_grad()
        scaler.scale(losses).backward()
        scaler.step(optimizer)
        scaler.update()

        total_loss += losses.item()

        if batch_idx % 50 == 0:
            print(f'Epoch {epoch} | Batch {batch_idx}/{num_batches} | Loss: {losses.item():.4f}')

    avg_loss = total_loss / num_batches
    return avg_loss


# ----------------------------- 5. 评估函数 (COCO 官方指标) -----------------------------
@torch.no_grad()
def evaluate(model, data_loader, device, coco_gt):
    model.eval()
    results = []

    for batch_idx, batch in enumerate(data_loader):
        images, targets, img_ids = batch
        images = [img.to(device) for img in images]

        outputs = model(images)

        for img_idx, output in enumerate(outputs):
            img_id = img_ids[img_idx]
            boxes = output['boxes'].cpu().numpy()
            scores = output['scores'].cpu().numpy()
            labels = output['labels'].cpu().numpy()

            # 将连续标签映射回 COCO 原始类别 ID
            # 注意：我们的标签 0->car, 1->motorcycle, 2->bus, 3->train, 4->truck
            # 需要映射回 COCO ID: [3,4,6,7,8]
            mapping = {0: 3, 1: 4, 2: 6, 3: 7, 4: 8}
            for box, score, label in zip(boxes, scores, labels):
                if label == 0:   # 背景类跳过
                    continue
                x1, y1, x2, y2 = box
                w = x2 - x1
                h = y2 - y1
                results.append({
                    'image_id': int(img_id),
                    'category_id': mapping[int(label)],
                    'bbox': [float(x1), float(y1), float(w), float(h)],
                    'score': float(score)
                })

    if len(results) == 0:
        print("No predictions, skip evaluation.")
        return None

    # 转为 COCO 评估格式
    coco_dt = coco_gt.loadRes(results)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    stats = coco_eval.stats
    return {
        'AP': stats[0],      # AP@0.5:0.95
        'AP50': stats[1],    # AP@0.5
        'AP75': stats[2],    # AP@0.75
        'AP_S': stats[3],    # small
        'AP_M': stats[4],    # medium
        'AP_L': stats[5]     # large
    }


# ----------------------------- 6. 主程序 -----------------------------
def main():
    # 配置参数
    data_root = './coco2017'            # 数据集根目录，需包含 train2017, val2017, annotations
    train_img_dir = os.path.join(data_root, 'train2017')
    val_img_dir = os.path.join(data_root, 'val2017')
    train_ann_file = os.path.join(data_root, 'annotations', 'instances_train2017.json')
    val_ann_file = os.path.join(data_root, 'annotations', 'instances_val2017.json')

    num_epochs = 12
    batch_size = 4          # 根据显存调整，AMP 下可尝试更大
    num_workers = 4
    lr = 0.02
    momentum = 0.9
    weight_decay = 1e-4
    milestones = [8, 11]
    gamma = 0.1

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # 加载数据集
    train_dataset = COCOVehicleDataset(root=train_img_dir, ann_file=train_ann_file, is_train=True)
    val_dataset = COCOVehicleDataset(root=val_img_dir, ann_file=val_ann_file, is_train=False)

    # 注意 collate_fn 使用全局函数
    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset, batch_size=2, shuffle=False,
        num_workers=num_workers, collate_fn=collate_fn, pin_memory=True
    )

    # 加载 COCO 真值用于评估
    coco_gt = COCO(val_ann_file)

    # 构建模型
    model = build_resnet50_fpn_detector(num_classes=len(train_dataset.VALID_IDS) + 1, pretrained=True)
    model.to(device)

    # 优化器和学习率调度器
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.SGD(params, lr=lr, momentum=momentum, weight_decay=weight_decay)
    lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=gamma)

    # AMP 梯度缩放器
    scaler = GradScaler()

    # 训练与验证循环
    best_ap = 0.0
    for epoch in range(1, num_epochs + 1):
        print(f"\n========== Epoch {epoch}/{num_epochs} ==========")
        avg_loss = train_one_epoch(model, optimizer, train_loader, device, epoch, scaler)
        lr_scheduler.step()
        print(f"Epoch {epoch} finished, Average Loss: {avg_loss:.4f}")

        # 每两个 epoch 评估一次
        if epoch % 2 == 0:
            print("Evaluating on validation set...")
            eval_results = evaluate(model, val_loader, device, coco_gt)
            if eval_results:
                print(f"AP: {eval_results['AP']:.2f}%, AP50: {eval_results['AP50']:.2f}%, AP_S: {eval_results['AP_S']:.2f}%")
                if eval_results['AP'] > best_ap:
                    best_ap = eval_results['AP']
                    torch.save(model.state_dict(), 'best_vehicle_detector.pth')
                    print("Best model saved.")

    # 保存最终模型
    torch.save(model.state_dict(), 'final_vehicle_detector.pth')
    print("\nTraining completed. Final model saved.")


if __name__ == '__main__':
    main()