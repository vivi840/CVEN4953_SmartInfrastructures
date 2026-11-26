Code for an Honours thesis on segmenting road defects (potholes, speed bumps) in Mobile Laser Scanning point clouds with a modified PointNet++. It is trained and tested on the RSRD dataset. The goal of this repository is that later students can reproduce the results and build on them.

The PointNet++ implementation is adapted from Yan, X 2021, Pointnet_Pointnet2_pytorch, source code, GitHub, viewed 01 June 2026, https://github.com/yanx27/Pointnet_Pointnet2_pytorch

File Purpose
color_train_semseg.py:	Trains the model. Switches select the loss (focal, ce, dice, poly), grouping (ball, cube), centroid sampling (fps, random), colour on/off and rotation augmentation.
evaluate_metrics.py:	Scores a checkpoint on the test strips (defect precision, recall, F1, IoU). --repeats 5 gives mean ± SD over five seeded passes. --rotate_deg rotates the test strips about the vertical axis.
pipeline.py	Scene-level pipeline: slides windows over a fused scene, predicts with a checkpoint, applies kerb suppression and scores the scene.
models/pointnet2_sem_seg_color.py:	The segmentation network. Its configuration is stored in every checkpoint, so evaluation rebuilds the right model automatically.
models/pointnet2_utils.py:	PointNet++ layers, plus random centroid sampling and cubic grouping.

DATA
Source: the RSRD road surface reconstruction dataset (Zhao et al. 2024), https://ztsrxh.github.io/rsrd/; paper: https://arxiv.org/abs/2310.02262. Download it from the project page.
Strip data: used unchanged from the dataset (the 4096-point strips used for training and testing). Put it in data/, laid out as expected by data_utils/ColorRoadDatasetLoader.py.
Scene data: the full scenes were fused (point cloud and camera colour) by following the instructions at the RSRD development toolkit, https://github.com/ztsrxh/RSRD_dev_toolkit. Except substituting RSRD-seg uncropped scenes in place of RSRD-dense, via matching file names. 

References
Yan, X 2021, Pointnet_Pointnet2_pytorch, source code, GitHub, viewed 01 June 2026, https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
Zhao, T, Guo, P, He, J & Wei, Y 2024, 'A hierarchical scheme of road unevenness perception with LiDAR for autonomous driving comfort', IEEE Transactions on Intelligent Vehicles, vol. 9, no. 1, pp. 2439–2448, doi:10.1109/TIV.2023.3337236.
