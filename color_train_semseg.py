"""
Train a PointNet++ road-defect segmentation model (switches: loss, grouping, sampling, rotation).
Run from the repository root, with the dataset in data/:
  python color_train_semseg.py --model pointnet2_sem_seg_color --epoch 32 --loss focal --rotate_aug on --log_dir my_run
Options: --loss focal|ce|dice|poly, --grouping ball|cube, --sampling fps|random, --no_color, --rotate_aug on|off|auto.
Checkpoints are saved to log/sem_seg/<log_dir>/checkpoints/.

Adapted from: Yan, X 2021, Pointnet_Pointnet2_pytorch, source code, GitHub, viewed 01 June 2026,
https://github.com/yanx27/Pointnet_Pointnet2_pytorch.
"""
import argparse
import os
from data_utils.S3DISDataLoader import S3DISDataset
import torch
import torch.nn as nn
import torch.nn.functional as F
import datetime
import logging
from data_utils.ColorRoadDatasetLoader import ColorRoadDataset
import sys
import importlib
import shutil
from tqdm import tqdm
import provider
import numpy as np
import time
from pathlib import Path

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

classes = ['normal_pavement', 'pavement_defects']
class2label = {cls: i for i, cls in enumerate(classes)}
seg_classes = class2label
seg_label_to_cat = {}
for i, cat in enumerate(seg_classes.keys()):
    seg_label_to_cat[i] = cat

def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pointnet_sem_seg', help='model name [default: pointnet_sem_seg]')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
    parser.add_argument('--epoch', default=32, type=int, help='Epoch to run [default: 32]')
    parser.add_argument('--learning_rate', default=0.001, type=float, help='Initial learning rate [default: 0.001]')
    parser.add_argument('--gpu', type=str, default='0', help='GPU to use [default: GPU 0]')
    parser.add_argument('--optimizer', type=str, default='Adam', help='Adam or SGD [default: Adam]')
    parser.add_argument('--log_dir', type=str, default=None, help='Log path [default: None]')
    parser.add_argument('--decay_rate', type=float, default=1e-4, help='weight decay [default: 1e-4]')
    parser.add_argument('--npoint', type=int, default=4096, help='Point Number [default: 4096]')
    parser.add_argument('--step_size', type=int, default=10, help='Decay step for lr decay [default: every 10 epochs]')
    parser.add_argument('--lr_decay', type=float, default=0.7, help='Decay rate for lr decay [default: 0.7]')
    parser.add_argument('--test_area', type=int, default=5, help='Which area to use for test, option: 1-6 [default: 5]')

    parser.add_argument('--no_color', action='store_true', help='use xyz only (drop the RGB channels)')
    parser.add_argument('--loss', type=str, default='focal', choices=['focal', 'ce', 'dice', 'poly'],
                        help="'focal' = class-weighted focal loss [default]; 'ce' = plain cross-entropy (Zhao et al.); 'dice' = cross-entropy + soft Dice; 'poly' = Poly-1 (cross-entropy + eps*(1-p_t))")
    parser.add_argument('--grouping', type=str, default='ball', choices=['ball', 'cube'],
                        help="'ball' = ball query [default]; 'cube' = Zhao et al. cubic grouping")
    parser.add_argument('--sampling', type=str, default='fps', choices=['fps', 'random'],
                        help="'fps' = farthest point sampling [default]; 'random' = random centroids (Zhao et al.)")
    parser.add_argument('--centroids', type=str, default='1024,256,64,16',
                        help='centroids per level [default: 1024,256,64,16]; Zhao et al. use 1024,512,256,128')
    parser.add_argument('--cube_norm', type=float, default=3.0,
                        help='metres per normalised unit used to convert Zhao et al. cube sizes to metres [default: 3.0]')
    parser.add_argument('--lat_axis', type=int, default=0, help='coordinate index ACROSS the road [default: 0 = x]')
    parser.add_argument('--long_axis', type=int, default=1, help='coordinate index ALONG the road [default: 1 = y]')
    parser.add_argument('--rotate_aug', type=str, default='auto', choices=['auto', 'on', 'off'],
                        help="random rotation about z during training. 'auto' = on for ball query, off for cubic "
                             "grouping (a rotated strip would no longer line up with the cube axes) [default: auto]")

    return parser.parse_args()

class VectorizedFocalLoss(nn.Module):
    def __init__(self, gamma=2):
        super(VectorizedFocalLoss, self).__init__()
        self.gamma = gamma

    def forward(self, pred, target, trans_feat, weight=None):

        log_pt = pred.gather(1, target.unsqueeze(1)).squeeze(1)
        pt = torch.exp(log_pt)

        focal_weight = (1 - pt) ** self.gamma

        if weight is not None:
            class_weight = weight[target]
            loss = -focal_weight * class_weight * log_pt
        else:
            loss = -focal_weight * log_pt

        return loss.mean()


class DiceCELoss(nn.Module):
    def forward(self, pred, target, trans_feat=None, weight=None):
        pred = pred.reshape(-1, pred.size(-1))
        target = target.reshape(-1)
        ce = F.nll_loss(pred, target)
        p = pred.exp()[:, 1]
        t = (target == 1).float()
        dice = 1 - (2 * (p * t).sum() + 1) / (p.sum() + t.sum() + 1)
        return ce + dice


class PolyCELoss(nn.Module):
    def __init__(self, eps=1.0):
        super().__init__()
        self.eps = eps
    def forward(self, pred, target, trans_feat=None, weight=None):
        pred = pred.reshape(-1, pred.size(-1))
        target = target.reshape(-1)
        nll = F.nll_loss(pred, target, reduction='none')
        pt = torch.exp(-nll)
        return (nll + self.eps * (1 - pt)).mean()


class PlainCrossEntropy(nn.Module):
    def forward(self, pred, target, trans_feat, weight=None):
        return F.nll_loss(pred, target)


def main(args):
    def log_string(str):
        logger.info(str)
        print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    experiment_dir = Path('./log/')
    experiment_dir.mkdir(exist_ok=True)
    experiment_dir = experiment_dir.joinpath('sem_seg')
    experiment_dir.mkdir(exist_ok=True)
    if args.log_dir is None:
        experiment_dir = experiment_dir.joinpath(timestr)
    else:
        experiment_dir = experiment_dir.joinpath(args.log_dir)
    experiment_dir.mkdir(exist_ok=True)
    checkpoints_dir = experiment_dir.joinpath('checkpoints/')
    checkpoints_dir.mkdir(exist_ok=True)
    log_dir = experiment_dir.joinpath('logs/')
    log_dir.mkdir(exist_ok=True)

    '''LOG'''
    args = parse_args()
    logger = logging.getLogger("Model")
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    log_string('PARAMETER ...')
    log_string(args)

    root = 'data'
    NUM_CLASSES = 2
    NUM_POINT = args.npoint
    BATCH_SIZE = args.batch_size

    '''MODEL CONFIGURATION (saved inside every checkpoint)'''
    model_cfg = dict(
        in_feat=3 if args.no_color else 6,
        grouping=args.grouping,
        sampling=args.sampling,
        centroids=tuple(int(c) for c in args.centroids.split(',')),
        cube_norm=args.cube_norm,
        lat_axis=args.lat_axis,
        long_axis=args.long_axis,
    )
    use_rotation = (args.rotate_aug == 'on') or (args.rotate_aug == 'auto' and args.grouping == 'ball')
    log_string('MODEL CONFIG: %s' % model_cfg)
    log_string('LOSS: %s | rotation augmentation: %s' % (args.loss, use_rotation))
    if args.grouping == 'cube' and use_rotation:
        log_string('WARNING: cubic grouping with rotation augmentation - strips will not line up with the cube axes')

    print("start loading training data ...")
    TRAIN_DATASET = ColorRoadDataset(root=root, split='train', num_points=NUM_POINT)
    print("!!! MY DATA SHAPE IS:", TRAIN_DATASET[0][0].shape)
    print("start loading test data ...")
    TEST_DATASET = ColorRoadDataset(root=root, split='test', num_points=NUM_POINT)


    trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=BATCH_SIZE, shuffle=True, num_workers=8,
                                                  pin_memory=True, drop_last=True,
                                                  worker_init_fn=lambda x: np.random.seed(x + int(time.time())))
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=BATCH_SIZE, shuffle=False, num_workers=8,
                                                 pin_memory=True, drop_last=True)
    weights = torch.tensor([0.3, 0.7]).cuda()
    log_string(f"--- ACTIVE LOSS WEIGHTS: {weights.cpu().tolist()} ---")

    log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    log_string("The number of test data is: %d" % len(TEST_DATASET))

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    shutil.copy('models/%s.py' % args.model, str(experiment_dir))
    shutil.copy('models/pointnet2_utils.py', str(experiment_dir))

    classifier = MODEL.get_model(NUM_CLASSES, **model_cfg).cuda()
    if args.loss == 'ce':
        criterion = PlainCrossEntropy().cuda()
    elif args.loss == 'dice':
        criterion = DiceCELoss().cuda()
    elif args.loss == 'poly':
        criterion = PolyCELoss().cuda()
    else:
        criterion = VectorizedFocalLoss(gamma=2).cuda()

    classifier.apply(inplace_relu)

    def weights_init(m):
        classname = m.__class__.__name__
        if classname.find('Conv2d') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)
        elif classname.find('Linear') != -1:
            torch.nn.init.xavier_normal_(m.weight.data)
            torch.nn.init.constant_(m.bias.data, 0.0)

    try:
        checkpoint = torch.load(str(experiment_dir) + '/checkpoints/best_model.pth')
        start_epoch = checkpoint['epoch']
        classifier.load_state_dict(checkpoint['model_state_dict'])
        log_string('Use pretrain model')
    except:
        log_string('No existing model, starting training from scratch...')
        start_epoch = 0
        classifier = classifier.apply(weights_init)

    if args.optimizer == 'Adam':
        optimizer = torch.optim.Adam(
            classifier.parameters(),
            lr=args.learning_rate,
            betas=(0.9, 0.999),
            eps=1e-08,
            weight_decay=args.decay_rate
        )
    else:
        optimizer = torch.optim.SGD(classifier.parameters(), lr=args.learning_rate, momentum=0.9)

    def bn_momentum_adjust(m, momentum):
        if isinstance(m, torch.nn.BatchNorm2d) or isinstance(m, torch.nn.BatchNorm1d):
            m.momentum = momentum

    LEARNING_RATE_CLIP = 1e-5
    MOMENTUM_ORIGINAL = 0.1
    MOMENTUM_DECCAY = 0.5
    MOMENTUM_DECCAY_STEP = args.step_size

    global_epoch = 0
    best_iou = 0

    for epoch in range(start_epoch, args.epoch):
        '''Train on chopped scenes'''
        log_string('**** Epoch %d (%d/%s) ****' % (global_epoch + 1, epoch + 1, args.epoch))
        lr = max(args.learning_rate * (args.lr_decay ** (epoch // args.step_size)), LEARNING_RATE_CLIP)
        log_string('Learning rate:%f' % lr)
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr
        momentum = MOMENTUM_ORIGINAL * (MOMENTUM_DECCAY ** (epoch // MOMENTUM_DECCAY_STEP))
        if momentum < 0.01:
            momentum = 0.01
        print('BN momentum updated to: %f' % momentum)
        classifier = classifier.apply(lambda x: bn_momentum_adjust(x, momentum))
        num_batches = len(trainDataLoader)
        total_correct = 0
        total_seen = 0
        loss_sum = 0
        classifier = classifier.train()

        for i, (points, target) in tqdm(enumerate(trainDataLoader), total=len(trainDataLoader), smoothing=0.9):
            optimizer.zero_grad()

            points = points.data.numpy()
            if use_rotation:
                points[:, :, :3] = provider.rotate_point_cloud_z(points[:, :, :3])
            points = torch.Tensor(points)
            points, target = points.float().cuda(), target.long().cuda()
            points = points.transpose(2, 1)

            seg_pred, trans_feat = classifier(points)
            seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

            batch_label = target.view(-1, 1)[:, 0].cpu().data.numpy()
            target = target.view(-1, 1)[:, 0]
            loss = criterion(seg_pred, target, trans_feat, weights)
            loss.backward()
            optimizer.step()

            pred_choice = seg_pred.cpu().data.max(1)[1].numpy()
            correct = np.sum(pred_choice == batch_label)
            total_correct += correct
            total_seen += (BATCH_SIZE * NUM_POINT)
            loss_sum += loss
        log_string('Training mean loss: %f' % (loss_sum / num_batches))
        log_string('Training accuracy: %f' % (total_correct / float(total_seen)))

        if epoch % 5 == 0:
            logger.info('Save model...')
            savepath = str(checkpoints_dir) + '/model.pth'
            log_string('Saving at %s' % savepath)
            state = {
                'epoch': epoch,
                'model_cfg': model_cfg,
                'model_state_dict': classifier.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
            }
            torch.save(state, savepath)
            log_string('Saving model....')

        '''Evaluate on chopped scenes'''
        with torch.no_grad():
            num_batches = len(testDataLoader)
            total_correct = 0
            total_seen = 0
            loss_sum = 0
            labelweights = np.zeros(NUM_CLASSES)
            total_seen_class = [0 for _ in range(NUM_CLASSES)]
            total_correct_class = [0 for _ in range(NUM_CLASSES)]
            total_iou_deno_class = [0 for _ in range(NUM_CLASSES)]
            classifier = classifier.eval()

            log_string('---- EPOCH %03d EVALUATION ----' % (global_epoch + 1))
            for i, (points, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
                points = points.data.numpy()
                points = torch.Tensor(points)
                points, target = points.float().cuda(), target.long().cuda()
                points = points.transpose(2, 1)

                seg_pred, trans_feat = classifier(points)
                pred_val = seg_pred.contiguous().cpu().data.numpy()
                seg_pred = seg_pred.contiguous().view(-1, NUM_CLASSES)

                batch_label = target.cpu().data.numpy()
                target = target.view(-1, 1)[:, 0]
                loss = criterion(seg_pred, target, trans_feat, weights)
                loss_sum += loss
                pred_val = np.argmax(pred_val, 2)
                correct = np.sum((pred_val == batch_label))
                total_correct += correct
                total_seen += (BATCH_SIZE * NUM_POINT)
                tmp, _ = np.histogram(batch_label, range(NUM_CLASSES + 1))
                labelweights += tmp

                for l in range(NUM_CLASSES):
                    total_seen_class[l] += np.sum((batch_label == l))
                    total_correct_class[l] += np.sum((pred_val == l) & (batch_label == l))
                    total_iou_deno_class[l] += np.sum(((pred_val == l) | (batch_label == l)))

            labelweights = labelweights.astype(np.float32) / np.sum(labelweights.astype(np.float32))
            mIoU = np.mean(np.array(total_correct_class) / (np.array(total_iou_deno_class, dtype=float) + 1e-6))
            log_string('eval mean loss: %f' % (loss_sum / float(num_batches)))
            log_string('eval point avg class IoU: %f' % (mIoU))
            log_string('eval point accuracy: %f' % (total_correct / float(total_seen)))
            log_string('eval point avg class acc: %f' % (
                np.mean(np.array(total_correct_class) / (np.array(total_seen_class, dtype=float) + 1e-6))))

            iou_per_class_str = '------- IoU --------\n'
            for l in range(NUM_CLASSES):
                iou_per_class_str += 'class %s weight: %.3f, IoU: %.3f \n' % (
                    seg_label_to_cat[l] + ' ' * (14 - len(seg_label_to_cat[l])), labelweights[l],
                    total_correct_class[l] / float(total_iou_deno_class[l]))

            log_string(iou_per_class_str)
            log_string('Eval mean loss: %f' % (loss_sum / num_batches))
            log_string('Eval accuracy: %f' % (total_correct / float(total_seen)))

            if mIoU >= best_iou:
                best_iou = mIoU
                logger.info('Save model...')
                savepath = str(checkpoints_dir) + '/best_model.pth'
                log_string('Saving at %s' % savepath)
                state = {
                    'epoch': epoch,
                    'class_avg_iou': mIoU,
                    'model_cfg': model_cfg,
                    'model_state_dict': classifier.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict(),
                }
                torch.save(state, savepath)
                log_string('Saving model....')
            log_string('Best mIoU: %f' % best_iou)
        global_epoch += 1


if __name__ == '__main__':
    args = parse_args()
    main(args)
