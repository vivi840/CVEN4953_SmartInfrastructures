"""
Dataset for the RSRD segmentation strips (x, y, z, r, g, b + defect label per point).
Not run directly: imported by color_train_semseg.py and evaluate_metrics.py.
Expected layout: <data_root>/rsrd_segmentation/{train,test}/*.pcd (binary PCD, 17 bytes per point).
"""
import os
import numpy as np
from torch.utils.data import Dataset


class ColorRoadDataset(Dataset):
    def __init__(self, root, split='train', num_points=4096):
        self.root = root
        self.split = split
        self.num_points = num_points
        self.data_dir = os.path.join(self.root, 'rsrd_segmentation', self.split)
        self.file_list = [os.path.join(self.data_dir, f) for f in os.listdir(self.data_dir) if f.endswith('.pcd')]

        if len(self.file_list) == 0:
            raise RuntimeError(f"No .pcd files found in {self.data_dir}")

        print(f"Loaded {len(self.file_list)} color-enabled road scans for {self.split} split.")

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        pcd_path = self.file_list[index]

        with open(pcd_path, 'rb') as f:
            header_bytes = b""
            while b"DATA binary" not in header_bytes:
                header_bytes += f.readline()
            header_length = len(header_bytes)
        num_points_in_file = int(next(l.split()[1] for l in header_bytes.decode('ascii', 'ignore').splitlines()
                                      if l.startswith('POINTS')))

        point_stride = 17
        raw_bytes = np.fromfile(pcd_path, dtype=np.uint8, offset=header_length)
        raw_bytes = raw_bytes[:num_points_in_file * point_stride].reshape(num_points_in_file, point_stride)

        points = np.ascontiguousarray(raw_bytes[:, 0:12]).view(np.float32).reshape(-1, 3).copy()
        labels = np.ascontiguousarray(raw_bytes[:, 12:14]).view(np.uint16).astype(np.int64).flatten()
        color_bytes = raw_bytes[:, 14:17].astype(np.float32) / 255.0

        points = np.hstack((points, color_bytes))

        choice = np.random.choice(num_points_in_file, self.num_points, replace=num_points_in_file < self.num_points)
        points = points[choice, :]
        labels = labels[choice]

        points[:, 0:3] -= np.mean(points[:, 0:3], axis=0)

        return points, labels
