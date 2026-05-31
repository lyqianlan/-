import cv2
import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from train import build_resnet50_fpn_detector, COCOVehicleDataset


def visualize_predictions(model, dataset, device, num_samples=6, score_thresh=0.5):
    model.eval()
    indices = np.random.choice(len(dataset), num_samples, replace=False)

    # 类别名称映射
    class_names = {1: 'car', 2: 'motorcycle', 3: 'bus', 4: 'train', 5: 'truck'}

    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    axes = axes.flatten()

    for i, idx in enumerate(indices):
        image, target, img_id = dataset[idx]  # image shape (3, H, W), range [0,1]
        img_np = image.permute(1, 2, 0).cpu().numpy()  # (H, W, 3) RGB, 范围[0,1]

        # 模型预测
        with torch.no_grad():
            pred = model([image.to(device)])[0]

        # 筛选高置信度框
        boxes = pred['boxes'][pred['scores'] > score_thresh].cpu().numpy()
        labels = pred['labels'][pred['scores'] > score_thresh].cpu().numpy()
        scores = pred['scores'][pred['scores'] > score_thresh].cpu().numpy()

        # 显示图像
        ax = axes[i]
        ax.imshow(img_np)

        # 绘制矩形框和标签
        for box, label, score in zip(boxes, labels, scores):
            x1, y1, x2, y2 = box
            w, h = x2 - x1, y2 - y1
            # 创建矩形 patch
            rect = patches.Rectangle((x1, y1), w, h, linewidth=2, edgecolor='lime', facecolor='none')
            ax.add_patch(rect)
            label_name = class_names.get(label, 'vehicle')
            ax.text(x1, y1 - 5, f'{label_name} {score:.2f}', color='red', fontsize=8,
                    bbox=dict(facecolor='white', alpha=0.7, edgecolor='none'))

        ax.set_title(f'Image {img_id}')
        ax.axis('off')

    plt.tight_layout()
    plt.savefig('prediction_samples.png', dpi=150)
    plt.show()


def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    val_dataset = COCOVehicleDataset(
        root='/shared-public/coco2017/val2017',
        ann_file='/shared-public/coco2017/annotations/instances_val2017.json',
        is_train=False
    )
    model = build_resnet50_fpn_detector(num_classes=6, pretrained=False)
    model.load_state_dict(torch.load('best_vehicle_detector.pth', map_location=device))
    model.to(device)
    visualize_predictions(model, val_dataset, device, num_samples=6, score_thresh=0.6)


if __name__ == '__main__':
    main()