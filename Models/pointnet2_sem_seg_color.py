"""
PointNet++ segmentation network (colour input, optional cubic grouping / random sampling).
Not run directly: select it with --model pointnet2_sem_seg_color in color_train_semseg.py and evaluate_metrics.py.

Adapted from: Yan, X 2021, Pointnet_Pointnet2_pytorch, source code, GitHub, viewed 01 June 2026,
https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet2_utils import PointNetSetAbstraction, PointNetFeaturePropagation

DEFAULT_CFG = dict(
    in_feat=6,
    grouping='ball',
    sampling='fps',
    centroids=(1024, 256, 64, 16),
    nsample=32,
    radii=(0.1, 0.2, 0.4, 0.8),
    cube_w=(0.01, 0.02, 0.03, 0.04),
    cube_l=(0.02, 0.04, 0.08, 0.16),
    cube_norm=3.0,
    lat_axis=0,
    long_axis=1,
)


class get_model(nn.Module):
    def __init__(self, num_classes, **kwargs):
        super(get_model, self).__init__()
        unknown = set(kwargs) - set(DEFAULT_CFG)
        if unknown:
            raise TypeError('unknown model option(s): %s' % sorted(unknown))
        cfg = dict(DEFAULT_CFG)
        cfg.update({k: (tuple(v) if isinstance(v, (list, tuple)) else v) for k, v in kwargs.items()})
        self.cfg = cfg
        self.in_feat = cfg['in_feat']

        npts, rad, ns = cfg['centroids'], cfg['radii'], cfg['nsample']
        mlps = ([32, 32, 64], [64, 64, 128], [128, 128, 256], [256, 256, 512])
        in_ch = (self.in_feat + 3, 64 + 3, 128 + 3, 256 + 3)
        cubes = [None] * 4
        if cfg['grouping'] == 'cube':
            cubes = [(w * cfg['cube_norm'], l * cfg['cube_norm'])
                     for w, l in zip(cfg['cube_w'], cfg['cube_l'])]

        def sa(i):
            return PointNetSetAbstraction(npts[i], rad[i], ns, in_ch[i], list(mlps[i]), False,
                                          grouping=cfg['grouping'], cube=cubes[i],
                                          sampling=cfg['sampling'],
                                          lat_axis=cfg['lat_axis'], long_axis=cfg['long_axis'])

        self.sa1 = sa(0)
        self.sa2 = sa(1)
        self.sa3 = sa(2)
        self.sa4 = sa(3)
        self.fp4 = PointNetFeaturePropagation(768, [256, 256])
        self.fp3 = PointNetFeaturePropagation(384, [256, 256])
        self.fp2 = PointNetFeaturePropagation(320, [256, 128])
        self.fp1 = PointNetFeaturePropagation(128, [128, 128, 128])
        self.conv1 = nn.Conv1d(128, 128, 1)
        self.bn1 = nn.BatchNorm1d(128)
        self.drop1 = nn.Dropout(0.5)
        self.conv2 = nn.Conv1d(128, num_classes, 1)

    def forward(self, xyz):
        xyz = xyz[:, :self.in_feat, :]
        l0_points = xyz
        l0_xyz = xyz[:,:3,:]

        l1_xyz, l1_points = self.sa1(l0_xyz, l0_points)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        l4_xyz, l4_points = self.sa4(l3_xyz, l3_points)

        l3_points = self.fp4(l3_xyz, l4_xyz, l3_points, l4_points)
        l2_points = self.fp3(l2_xyz, l3_xyz, l2_points, l3_points)
        l1_points = self.fp2(l1_xyz, l2_xyz, l1_points, l2_points)
        l0_points = self.fp1(l0_xyz, l1_xyz, None, l1_points)

        x = self.drop1(F.relu(self.bn1(self.conv1(l0_points))))
        x = self.conv2(x)
        x = F.log_softmax(x, dim=1)
        x = x.permute(0, 2, 1)
        return x, l4_points


def load_model(ckpt_path, num_classes=2, device='cuda'):
    try:
        ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    except TypeError:
        ckpt = torch.load(ckpt_path, map_location=device)
    cfg = ckpt.get('model_cfg', {})
    model = get_model(num_classes, **cfg).to(device)
    model.load_state_dict(ckpt['model_state_dict'])
    return model.eval(), model.cfg


class get_loss(nn.Module):
    def __init__(self, gamma=2.0):
        super(get_loss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target, trans_feat, weight):
        if pred.dim() == 3:
            pred = pred.view(-1, pred.size(-1))
            target = target.view(-1)

        nll_loss_unweighted = F.nll_loss(pred, target, reduction='none')
        pt = torch.exp(-nll_loss_unweighted)

        nll_loss_weighted = F.nll_loss(pred, target, weight=weight, reduction='none')

        focal_loss = ((1.0 - pt) ** self.gamma) * nll_loss_weighted

        return focal_loss.mean()

if __name__ == '__main__':
    model = get_model(2)
    xyz = torch.rand(6, 6, 2048)
    (model(xyz))
