"""
Score a checkpoint on the test strips (defect precision, recall, F1, IoU).
Run from the repository root:
  python evaluate_metrics.py --model pointnet2_sem_seg_color --checkpoint log/sem_seg/my_run/checkpoints/best_model.pth --repeats 5
Add --rotate_deg 90 to rotate every test strip about the vertical axis.
"""
import os
import argparse
import torch
import numpy as np
from tqdm import tqdm
import importlib
import sys
sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), 'models'))

from data_utils.ColorRoadDatasetLoader import ColorRoadDataset


def parse_args():
    parser = argparse.ArgumentParser('Evaluation')
    parser.add_argument('--model', type=str, default='pointnet2_sem_seg_color', help='model name')
    parser.add_argument('--batch_size', type=int, default=16, help='batch size')
    parser.add_argument('--checkpoint', type=str, required=True, help='path to best_model.pth')
    parser.add_argument('--data_root', type=str, default='data/', help='root directory of the dataset')
    parser.add_argument('--npoint', type=int, default=4096, help='points per strip (must match training) [default: 4096]')
    parser.add_argument('--seed', type=int, default=0, help='random seed [default: 0]')
    parser.add_argument('--repeats', type=int, default=1,
                        help='evaluate this many times with different seeds and report mean +- std '
                             '(useful for random centroid sampling) [default: 1]')
    parser.add_argument('--rotate_deg', type=float, default=0.0,
                        help='rotate every test strip by this many degrees about the vertical (z) axis, '
                             'around its own x/y centre, before it goes into the model. 0 = normal evaluation '
                             '(normal evaluation). Use 90 to test road-direction dependence '
                             '[default: 0]')
    return parser.parse_args()

def rotate_strips(points, deg):
    if deg == 0:
        return points
    a = np.deg2rad(deg)
    c, s = float(np.cos(a)), float(np.sin(a))
    out = points.clone()
    cx = points[:, :, 0].mean(dim=1, keepdim=True)
    cy = points[:, :, 1].mean(dim=1, keepdim=True)
    dx = points[:, :, 0] - cx
    dy = points[:, :, 1] - cy
    out[:, :, 0] = cx + c * dx - s * dy
    out[:, :, 1] = cy + s * dx + c * dy
    return out

def calculate_metrics(all_preds, all_labels, num_classes=2):
    cm = np.zeros((num_classes, num_classes), dtype=np.int64)
    cm += np.bincount(num_classes * all_labels.astype(int) + all_preds.astype(int), minlength=num_classes**2).reshape(num_classes, num_classes)

    tp = np.diag(cm)
    fp = cm.sum(axis=0) - tp
    fn = cm.sum(axis=1) - tp

    total_points = cm.sum()
    accuracy = tp.sum() / total_points if total_points > 0 else 0.0

    with np.errstate(divide='ignore', invalid='ignore'):
        recall_per_class = np.nan_to_num(tp / (tp + fn))
        precision_per_class = np.nan_to_num(tp / (tp + fp))
        f1_per_class = np.nan_to_num(2 * (precision_per_class * recall_per_class) / (precision_per_class + recall_per_class))
        iou_per_class = np.nan_to_num(tp / (tp + fp + fn))

    return {
        "Accuracy": accuracy,
        "mIoU": np.mean(iou_per_class),
        "mF1": np.mean(f1_per_class),
        "mRecall": np.mean(recall_per_class),
        "IoU_per_class": iou_per_class,
        "F1_per_class": f1_per_class,
        "Precision_per_class": precision_per_class,
        "Recall_per_class": recall_per_class,
        "confusion": cm,
    }

def run_pass(classifier, test_loader, device, rotate_deg=0.0):
    all_preds = []
    all_labels = []
    with torch.no_grad():
        for points, target in tqdm(test_loader, total=len(test_loader)):
            points, target = points.to(device), target.to(device)

            points = rotate_strips(points.float(), rotate_deg).transpose(2, 1)

            seg_pred, _ = classifier(points)

            preds = seg_pred.argmax(dim=2)

            all_preds.append(preds.cpu().numpy().reshape(-1))
            all_labels.append(target.cpu().numpy().reshape(-1))
    return np.concatenate(all_preds), np.concatenate(all_labels)

def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    print("--- Loading Validation/Test Dataset ---")
    TEST_DATASET = ColorRoadDataset(root=args.data_root, split='test', num_points=args.npoint)
    test_loader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=args.batch_size, shuffle=False, num_workers=2)
    print(f"Test strips: {len(TEST_DATASET)} | rotation applied to every strip: {args.rotate_deg} degrees")

    print(f"--- Loading Checkpoint from {args.checkpoint} ---")
    checkpoint = torch.load(args.checkpoint, map_location=device, weights_only=False)
    model_cfg = checkpoint.get('model_cfg', {}) if isinstance(checkpoint, dict) else {}
    print(f"Model configuration in checkpoint: {model_cfg if model_cfg else 'none (original model)'}")
    if isinstance(checkpoint, dict) and 'epoch' in checkpoint:
        print(f"Checkpoint epoch: {checkpoint['epoch'] + 1}"
              + (f" | mIoU recorded during training: {float(checkpoint['class_avg_iou']):.4f}"
                 if 'class_avg_iou' in checkpoint else ''))

    MODEL = importlib.import_module(args.model)
    classifier = MODEL.get_model(2, **model_cfg).to(device)

    if 'model_state_dict' in checkpoint:
        classifier.load_state_dict(checkpoint['model_state_dict'])
    else:
        classifier.load_state_dict(checkpoint)
    classifier.eval()

    results = []
    for r in range(args.repeats):
        torch.manual_seed(args.seed + r)
        np.random.seed(args.seed + r)
        print(f"--- Running Evaluation Pass {r + 1}/{args.repeats} (seed {args.seed + r}) ---")
        all_preds, all_labels = run_pass(classifier, test_loader, device, args.rotate_deg)
        results.append(calculate_metrics(all_preds, all_labels, num_classes=2))

    metrics = results[0]
    print("\n================ FINAL EVALUATION RESULTS ================")
    print(f"Overall Accuracy:                {metrics['Accuracy']:.4f}")
    print(f"Mean Recall (mRec):              {metrics['mRecall']:.4f}")
    print(f"Mean F1-Score (mF1):             {metrics['mF1']:.4f}")
    print(f"Mean Intersection over Union (mIoU): {metrics['mIoU']:.4f}")
    print("----------------------------------------------------------")
    print(f"IoU per class (Normal, Defect):   {metrics['IoU_per_class']}")
    print(f"F1 per class (Normal, Defect):    {metrics['F1_per_class']}")
    print("---------------- DEFECT CLASS (for the table) -------------")
    print(f"Defect Precision: {metrics['Precision_per_class'][1]:.4f}")
    print(f"Defect Recall:    {metrics['Recall_per_class'][1]:.4f}")
    print(f"Defect F1:        {metrics['F1_per_class'][1]:.4f}")
    print(f"Defect IoU:       {metrics['IoU_per_class'][1]:.4f}")
    print(f"Confusion matrix (rows = true, cols = predicted; Normal, Defect):\n{metrics['confusion']}")
    if args.repeats > 1:
        print("---------------- MEAN +- STD over %d passes -------------" % args.repeats)
        for name, getter in [("Accuracy", lambda m: m['Accuracy']),
                             ("mIoU", lambda m: m['mIoU']),
                             ("Defect Precision", lambda m: m['Precision_per_class'][1]),
                             ("Defect Recall", lambda m: m['Recall_per_class'][1]),
                             ("Defect F1", lambda m: m['F1_per_class'][1]),
                             ("Defect IoU", lambda m: m['IoU_per_class'][1])]:
            vals = np.array([getter(m) for m in results])
            print(f"{name:17s} {vals.mean():.4f} +- {vals.std():.4f}")
    print("==========================================================\n")

if __name__ == '__main__':
    args = parse_args()
    main(args)
