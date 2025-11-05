import os
import os.path as osp

import torch
import torch.nn as nn
from torch.utils import data
from torch.autograd import Variable
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F

import numpy as np
import random
import wandb
from tqdm import tqdm
from datetime import datetime

from model.refinenetlw import rf_lw101
from model.fogpassfilter import FogPassFilter_conv1, FogPassFilter_res1
from utils.losses import CrossEntropy2d
from utils.mtl_loss import HydraNetMTLoss
from dataset.paired_cityscapes import Pairedcityscapes
from dataset.Foggy_Zurich import foggyzurichDataSet
from configs.train_config import get_arguments
from utils.optimisers import get_optimisers, get_lr_schedulers
from pytorch_metric_learning import losses
from pytorch_metric_learning.distances import CosineSimilarity
from pytorch_metric_learning.reducers import MeanReducer


IMG_MEAN = np.array((104.00698793, 116.66876762, 122.67891434), dtype=np.float32)
RESTORE_FROM = 'without_pretraining'
RESTORE_FROM_fogpass = 'without_pretraining'


def loss_calc(pred, label, gpu):
    label = Variable(label.long()).cuda(gpu)
    criterion = CrossEntropy2d().cuda(gpu)
    return criterion(pred, label)


def gram_matrix(tensor):
    d, h, w = tensor.size()
    tensor = tensor.view(d, h * w)
    gram = torch.mm(tensor, tensor.t())
    return gram


def setup_optimisers_and_schedulers(args, model):
    optimisers = get_optimisers(
        model=model,
        enc_optim_type="sgd",
        enc_lr=6e-4,
        enc_weight_decay=1e-5,
        enc_momentum=0.9,
        dec_optim_type="sgd",
        dec_lr=6e-3,
        dec_weight_decay=1e-5,
        dec_momentum=0.9,
    )
    schedulers = get_lr_schedulers(
        enc_optim=optimisers[0],
        dec_optim=optimisers[1],
        enc_lr_gamma=0.5,
        dec_lr_gamma=0.5,
        enc_scheduler_type="multistep",
        dec_scheduler_type="multistep",
        epochs_per_stage=(100, 100, 100),
    )
    return optimisers, schedulers


def make_list(x):
    if isinstance(x, list):
        return x
    elif isinstance(x, tuple):
        return list(x)
    else:
        return [x]


class EdgeHead(nn.Module):
    """
    Lightweight FPN-style edge head using backbone features (l1..l4) at strides 4,8,16,32.
    Produces 1-channel edge logits at stride 4 (same as layer1 feature resolution).
    """

    def __init__(self):
        super().__init__()
        # Channel dims for rf_lw101 Bottleneck backbone: l1=256, l2=512, l3=1024, l4=2048
        self.red_l1 = nn.Conv2d(256, 64, 1, bias=False)
        self.red_l2 = nn.Conv2d(512, 64, 1, bias=False)
        self.red_l3 = nn.Conv2d(1024, 64, 1, bias=False)
        self.red_l4 = nn.Conv2d(2048, 64, 1, bias=False)
        self.refine = nn.Sequential(
            nn.Conv2d(64, 64, 3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 1, 1)
        )

    def forward(self, l1, l2, l3, l4):
        # reduce
        p1 = self.red_l1(l1)
        p2 = self.red_l2(l2)
        p3 = self.red_l3(l3)
        p4 = self.red_l4(l4)
        # upsample to l1 size and fuse
        size = l1.shape[-2:]
        p2 = F.interpolate(p2, size=size, mode='bilinear', align_corners=True)
        p3 = F.interpolate(p3, size=size, mode='bilinear', align_corners=True)
        p4 = F.interpolate(p4, size=size, mode='bilinear', align_corners=True)
        f = p1 + p2 + p3 + p4
        return self.refine(f)


def targets_to_edge(seg_targets: torch.Tensor, thickness: int = 1) -> torch.Tensor:
    """
    Derive binary edge targets (0/1) from segmentation labels using simple morphological gradient.
    seg_targets: [N, H, W] (int)
    returns: [N, 1, H, W] (float)
    """
    # compute edges by comparing with shifted neighbors
    with torch.no_grad():
        n, h, w = seg_targets.shape
        dev = seg_targets.device
        edges = torch.zeros((n, 1, h, w), dtype=torch.float32, device=dev)
        # right and down shifts
        right = torch.zeros_like(seg_targets)
        right[:, :, :-1] = seg_targets[:, :, 1:]
        down = torch.zeros_like(seg_targets)
        down[:, :-1, :] = seg_targets[:, 1:, :]
        e = (seg_targets != right) | (seg_targets != down)
        e = e.float().unsqueeze(1)
        if thickness > 1:
            # thicken edges using max-pooling
            k = 2 * thickness + 1
            pad = thickness
            e = F.max_pool2d(e, kernel_size=k, stride=1, padding=pad)
        edges = torch.clamp(e, 0, 1)
        return edges


def main():
    args = get_arguments()
    # defaults for new flags if not present in existing configs
    if not hasattr(args, 'enable_edges'):
        args.enable_edges = True
    if not hasattr(args, 'lambda_fog'):
        args.lambda_fog = getattr(args, 'lambda_fsm', 1.0)
    if not hasattr(args, 'edge_from_label'):
        args.edge_from_label = True  # derive edges from seg labels by default

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(args.random_seed)
    np.random.seed(args.random_seed)
    torch.manual_seed(args.random_seed)
    torch.cuda.manual_seed_all(args.random_seed)
    torch.cuda.manual_seed(args.random_seed)

    now = datetime.now().strftime('%m-%d-%H-%M')
    run_name = f'{args.file_name}-improve-{now}'

    wandb.init(project='FIFO', name=f'{run_name}')
    wandb.config.update(vars(args))

    w, h = map(int, args.input_size.split(','))
    input_size = (w, h)

    w_r, h_r = map(int, args.input_size_rf.split(','))
    input_size_rf = (w_r, h_r)

    cudnn.enabled = True
    gpu = args.gpu

    # Backbone
    if args.restore_from == RESTORE_FROM:
        start_iter = 0
        model = rf_lw101(num_classes=args.num_classes)
    else:
        restore = torch.load(args.restore_from)
        model = rf_lw101(num_classes=args.num_classes)
        model.load_state_dict(restore['state_dict'])
        start_iter = 0

    model.train()
    model.cuda(args.gpu)

    # Edge head and its optimizer
    edge_head = EdgeHead().cuda(args.gpu)
    edge_head_opt = torch.optim.SGD(edge_head.parameters(), lr=6e-3, momentum=0.9, weight_decay=1e-5)

    # Fog-pass filters
    lr_fpf1 = 1e-3
    lr_fpf2 = 1e-3
    if args.modeltrain == 'train':
        lr_fpf1 = 5e-4

    FogPassFilter1 = FogPassFilter_conv1(2080)
    FogPassFilter1_optimizer = torch.optim.Adamax([p for p in FogPassFilter1.parameters() if p.requires_grad], lr=lr_fpf1)
    FogPassFilter1.cuda(args.gpu)
    FogPassFilter2 = FogPassFilter_res1(32896)
    FogPassFilter2_optimizer = torch.optim.Adamax([p for p in FogPassFilter2.parameters() if p.requires_grad], lr=lr_fpf2)
    FogPassFilter2.cuda(args.gpu)

    if args.restore_from_fogpass != RESTORE_FROM_fogpass:
        restore = torch.load(args.restore_from_fogpass)
        FogPassFilter1.load_state_dict(restore['fogpass1_state_dict'])
        FogPassFilter2.load_state_dict(restore['fogpass2_state_dict'])

    fogpassfilter_loss = losses.ContrastiveLoss(
        pos_margin=0.1,
        neg_margin=0.1,
        distance=CosineSimilarity(),
        reducer=MeanReducer()
    )

    cudnn.benchmark = True

    if not os.path.exists(args.snapshot_dir):
        os.makedirs(args.snapshot_dir)

    cwsf_pair_loader = data.DataLoader(
        Pairedcityscapes(
            args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN, set=args.set
        ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )

    rf_loader = data.DataLoader(
        foggyzurichDataSet(
            args.data_dir_rf, args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN, set=args.set
        ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )

    cwsf_pair_loader_fogpass = data.DataLoader(
        Pairedcityscapes(
            args.data_dir, args.data_dir_cwsf, args.data_list, args.data_list_cwsf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN, set=args.set
        ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )

    rf_loader_fogpass = data.DataLoader(
        foggyzurichDataSet(
            args.data_dir_rf, args.data_list_rf,
            max_iters=args.num_steps * args.iter_size * args.batch_size,
            mean=IMG_MEAN, set=args.set
        ),
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
        pin_memory=True
    )

    rf_loader_iter = enumerate(rf_loader)
    cwsf_pair_loader_iter = enumerate(cwsf_pair_loader)
    cwsf_pair_loader_iter_fogpass = enumerate(cwsf_pair_loader_fogpass)
    rf_loader_iter_fogpass = enumerate(rf_loader_fogpass)

    optimisers, schedulers = setup_optimisers_and_schedulers(args, model=model)
    opts = make_list(optimisers) + [edge_head_opt]
    kl_loss = torch.nn.KLDivLoss(reduction='batchmean').cuda(args.gpu)
    m = nn.Softmax(dim=1).cuda(args.gpu)
    log_m = nn.LogSoftmax(dim=1).cuda(args.gpu)

    # MTL loss
    mtl_loss_fn = HydraNetMTLoss(
        seg_criterion=CrossEntropy2d().cuda(args.gpu),
        edge_criterion=nn.BCEWithLogitsLoss().cuda(args.gpu)
    ).cuda(args.gpu)

    for i_iter in tqdm(range(start_iter, args.num_steps)):
        loss_seg_cw_value = 0
        loss_seg_sf_value = 0
        loss_fsm_value = 0
        loss_con_value = 0
        loss_edge_value = 0

        for opt in opts:
            opt.zero_grad()

        for sub_i in range(args.iter_size):
            # ===== Stage 1: Train FPF (freeze segmentation + edge head) =====
            model.eval()
            edge_head.eval()
            for p in model.parameters():
                p.requires_grad = False
            for p in edge_head.parameters():
                p.requires_grad = False
            for p in FogPassFilter1.parameters():
                p.requires_grad = True
            for p in FogPassFilter2.parameters():
                p.requires_grad = True

            _, batch = cwsf_pair_loader_iter_fogpass.__next__()
            sf_image, cw_image, label, size, sf_name, cw_name = batch
            interp = nn.Upsample(size=(size[0][0], size[0][1]), mode='bilinear')

            _, batch_rf = rf_loader_iter_fogpass.__next__()
            rf_img, rf_size, rf_name = batch_rf
            img_rf = Variable(rf_img).cuda(args.gpu)
            feature_rf0, feature_rf1, feature_rf2, feature_rf3, feature_rf4, feature_rf5 = model(img_rf)

            images = Variable(sf_image).cuda(args.gpu)
            feature_sf0, feature_sf1, feature_sf2, feature_sf3, feature_sf4, feature_sf5 = model(images)

            images_cw = Variable(cw_image).cuda(args.gpu)
            feature_cw0, feature_cw1, feature_cw2, feature_cw3, feature_cw4, feature_cw5 = model(images_cw)

            fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
            sf_features = {'layer0': feature_sf0, 'layer1': feature_sf1}
            cw_features = {'layer0': feature_cw0, 'layer1': feature_cw1}
            rf_features = {'layer0': feature_rf0, 'layer1': feature_rf1}

            total_fpf_loss = 0

            for idx, layer in enumerate(fsm_weights):
                cw_feature = cw_features[layer]
                sf_feature = sf_features[layer]
                rf_feature = rf_features[layer]

                if idx == 0:
                    fogpassfilter = FogPassFilter1
                    fogpassfilter_optimizer = FogPassFilter1_optimizer
                elif idx == 1:
                    fogpassfilter = FogPassFilter2
                    fogpassfilter_optimizer = FogPassFilter2_optimizer

                fogpassfilter.train()
                fogpassfilter_optimizer.zero_grad()

                sf_gram = [0] * args.batch_size
                cw_gram = [0] * args.batch_size
                rf_gram = [0] * args.batch_size
                vector_sf_gram = [0] * args.batch_size
                vector_cw_gram = [0] * args.batch_size
                vector_rf_gram = [0] * args.batch_size
                fog_factor_sf = [0] * args.batch_size
                fog_factor_cw = [0] * args.batch_size
                fog_factor_rf = [0] * args.batch_size

                for batch_idx in range(args.batch_size):
                    sf_gram[batch_idx] = gram_matrix(sf_feature[batch_idx])
                    cw_gram[batch_idx] = gram_matrix(cw_feature[batch_idx])
                    rf_gram[batch_idx] = gram_matrix(rf_feature[batch_idx])

                    ones_sf = torch.triu(torch.ones(sf_gram[batch_idx].size()[0], sf_gram[batch_idx].size()[1])).to(sf_gram[batch_idx].device)
                    ones_cw = torch.triu(torch.ones(cw_gram[batch_idx].size()[0], cw_gram[batch_idx].size()[1])).to(cw_gram[batch_idx].device)
                    ones_rf = torch.triu(torch.ones(rf_gram[batch_idx].size()[0], rf_gram[batch_idx].size()[1])).to(rf_gram[batch_idx].device)

                    vector_sf_gram[batch_idx] = Variable(sf_gram[batch_idx][ones_sf == 1], requires_grad=True)
                    vector_cw_gram[batch_idx] = Variable(cw_gram[batch_idx][ones_cw == 1], requires_grad=True)
                    vector_rf_gram[batch_idx] = Variable(rf_gram[batch_idx][ones_rf == 1], requires_grad=True)

                    fog_factor_sf[batch_idx] = fogpassfilter(vector_sf_gram[batch_idx])
                    fog_factor_cw[batch_idx] = fogpassfilter(vector_cw_gram[batch_idx])
                    fog_factor_rf[batch_idx] = fogpassfilter(vector_rf_gram[batch_idx])

                fog_factor_embeddings = torch.cat((
                    torch.unsqueeze(fog_factor_sf[0], 0), torch.unsqueeze(fog_factor_cw[0], 0), torch.unsqueeze(fog_factor_rf[0], 0),
                    torch.unsqueeze(fog_factor_sf[1], 0), torch.unsqueeze(fog_factor_cw[1], 0), torch.unsqueeze(fog_factor_rf[1], 0),
                    torch.unsqueeze(fog_factor_sf[2], 0), torch.unsqueeze(fog_factor_cw[2], 0), torch.unsqueeze(fog_factor_rf[2], 0),
                    torch.unsqueeze(fog_factor_sf[3], 0), torch.unsqueeze(fog_factor_cw[3], 0), torch.unsqueeze(fog_factor_rf[3], 0)
                ), 0)

                fog_factor_embeddings_norm = torch.norm(fog_factor_embeddings, p=2, dim=1).detach()
                size_fog_factor = fog_factor_embeddings.size()
                fog_factor_embeddings = fog_factor_embeddings.div(fog_factor_embeddings_norm.expand(size_fog_factor[1], 12).t())
                fog_factor_labels = torch.LongTensor([0, 1, 2, 0, 1, 2, 0, 1, 2, 0, 1, 2]).to(fog_factor_embeddings.device)
                fog_pass_filter_loss = fogpassfilter_loss(fog_factor_embeddings, fog_factor_labels)

                total_fpf_loss += fog_pass_filter_loss

                wandb.log({f'layer{idx}/fpf loss': fog_pass_filter_loss}, step=i_iter)
                wandb.log({f'layer{idx}/total fpf loss': total_fpf_loss}, step=i_iter)

            total_fpf_loss.backward(retain_graph=False)

            # ===== Stage 2: Train HydraNet (seg + edge) to fool FPF =====
            if args.modeltrain == 'train':
                model.train()
                edge_head.train()
                for p in model.parameters():
                    p.requires_grad = True
                for p in edge_head.parameters():
                    p.requires_grad = True
                for p in FogPassFilter1.parameters():
                    p.requires_grad = False
                for p in FogPassFilter2.parameters():
                    p.requires_grad = False

                _, batch = cwsf_pair_loader_iter.__next__()
                sf_image, cw_image, label, size, sf_name, cw_name = batch

                interp = nn.Upsample(size=(size[0][0], size[0][1]), mode='bilinear')

                loss_mtl = 0.0
                loss_con = 0.0
                loss_fsm = 0.0

                # branch A: cw + sf
                if i_iter % 3 == 0:
                    images_sf = Variable(sf_image).cuda(args.gpu)
                    f_sf0, f_sf1, f_sf2, f_sf3, f_sf4, seg_sf = model(images_sf)
                    seg_sf_up = interp(seg_sf)
                    edge_sf = edge_head(f_sf1, f_sf2, f_sf3, f_sf4)
                    edge_sf_up = F.interpolate(edge_sf, size=seg_sf_up.shape[-2:], mode='bilinear', align_corners=True)

                    images_cw = Variable(cw_image).cuda(args.gpu)
                    f_cw0, f_cw1, f_cw2, f_cw3, f_cw4, seg_cw = model(images_cw)
                    seg_cw_up = interp(seg_cw)
                    edge_cw = edge_head(f_cw1, f_cw2, f_cw3, f_cw4)
                    edge_cw_up = F.interpolate(edge_cw, size=seg_cw_up.shape[-2:], mode='bilinear', align_corners=True)

                    # consistency loss on logits (pre-softmax)
                    loss_con = kl_loss(log_m(seg_sf), m(seg_cw))

                    # MTL losses (derive edges from labels if needed)
                    if args.enable_edges:
                        if args.edge_from_label:
                            edge_t = targets_to_edge(label.cuda(args.gpu))
                        else:
                            edge_t = None
                    else:
                        edge_t = None

                    loss_mtl_sf, _ = mtl_loss_fn(seg_sf_up, edge_sf_up if args.enable_edges else None,
                                                  label.cuda(args.gpu), edge_t)
                    loss_mtl_cw, _ = mtl_loss_fn(seg_cw_up, edge_cw_up if args.enable_edges else None,
                                                  label.cuda(args.gpu), edge_t)
                    loss_mtl = 0.5 * (loss_mtl_sf + loss_mtl_cw)

                    # features for fog loss
                    fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
                    sf_features = {'layer0': f_sf0, 'layer1': f_sf1}
                    cw_features = {'layer0': f_cw0, 'layer1': f_cw1}

                # branch B: sf + rf
                elif i_iter % 3 == 1:
                    _, batch_rf = rf_loader_iter.__next__()
                    rf_img, rf_size, rf_name = batch_rf

                    images_sf = Variable(sf_image).cuda(args.gpu)
                    f_sf0, f_sf1, f_sf2, f_sf3, f_sf4, seg_sf = model(images_sf)
                    seg_sf_up = interp(seg_sf)
                    edge_sf = edge_head(f_sf1, f_sf2, f_sf3, f_sf4)
                    edge_sf_up = F.interpolate(edge_sf, size=seg_sf_up.shape[-2:], mode='bilinear', align_corners=True)

                    img_rf = Variable(rf_img).cuda(args.gpu)
                    f_rf0, f_rf1, f_rf2, f_rf3, f_rf4, seg_rf = model(img_rf)

                    if args.enable_edges:
                        if args.edge_from_label:
                            edge_t = targets_to_edge(label.cuda(args.gpu))
                        else:
                            edge_t = None
                    else:
                        edge_t = None

                    loss_mtl, _ = mtl_loss_fn(seg_sf_up, edge_sf_up if args.enable_edges else None,
                                              label.cuda(args.gpu), edge_t)

                    fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
                    sf_features = {'layer0': f_sf0, 'layer1': f_sf1}
                    rf_features = {'layer0': f_rf0, 'layer1': f_rf1}

                # branch C: cw + rf
                else:
                    _, batch_rf = rf_loader_iter.__next__()
                    rf_img, rf_size, rf_name = batch_rf

                    images_cw = Variable(cw_image).cuda(args.gpu)
                    f_cw0, f_cw1, f_cw2, f_cw3, f_cw4, seg_cw = model(images_cw)
                    seg_cw_up = interp(seg_cw)
                    edge_cw = edge_head(f_cw1, f_cw2, f_cw3, f_cw4)
                    edge_cw_up = F.interpolate(edge_cw, size=seg_cw_up.shape[-2:], mode='bilinear', align_corners=True)

                    img_rf = Variable(rf_img).cuda(args.gpu)
                    f_rf0, f_rf1, f_rf2, f_rf3, f_rf4, seg_rf = model(img_rf)

                    if args.enable_edges:
                        if args.edge_from_label:
                            edge_t = targets_to_edge(label.cuda(args.gpu))
                        else:
                            edge_t = None
                    else:
                        edge_t = None

                    loss_mtl, _ = mtl_loss_fn(seg_cw_up, edge_cw_up if args.enable_edges else None,
                                              label.cuda(args.gpu), edge_t)

                    fsm_weights = {'layer0': 0.5, 'layer1': 0.5}
                    cw_features = {'layer0': f_cw0, 'layer1': f_cw1}
                    rf_features = {'layer0': f_rf0, 'layer1': f_rf1}

                # Fog-invariant loss (reuse original logic)
                loss_fsm = 0
                for idx, layer in enumerate(fsm_weights):
                    if i_iter % 3 == 0:
                        a_feature = cw_features[layer]
                        b_feature = sf_features[layer]
                    if i_iter % 3 == 1:
                        a_feature = rf_features[layer]
                        b_feature = sf_features[layer]
                    if i_iter % 3 == 2:
                        a_feature = rf_features[layer]
                        b_feature = cw_features[layer]

                    layer_fsm_loss = 0
                    na, da, ha, wa = a_feature.size()
                    nb, db, hb, wb = b_feature.size()

                    if idx == 0:
                        fogpassfilter = FogPassFilter1
                    elif idx == 1:
                        fogpassfilter = FogPassFilter2

                    fogpassfilter.eval()

                    for batch_idx in range(min(4, a_feature.size(0))):
                        b_gram = gram_matrix(b_feature[batch_idx])
                        a_gram = gram_matrix(a_feature[batch_idx])

                        if i_iter % 3 == 1 or i_iter % 3 == 2:
                            a_gram = a_gram * (hb * wb) / (ha * wa)

                        mask_b = torch.triu(torch.ones(b_gram.size()[0], b_gram.size()[1])).to(b_gram.device)
                        mask_a = torch.triu(torch.ones(a_gram.size()[0], a_gram.size()[1])).to(a_gram.device)
                        vector_b_gram = b_gram[mask_b == 1].requires_grad_()
                        vector_a_gram = a_gram[mask_a == 1].requires_grad_()

                        fog_factor_b = fogpassfilter(vector_b_gram)
                        fog_factor_a = fogpassfilter(vector_a_gram)
                        half = int(fog_factor_b.shape[0] / 2)

                        layer_fsm_loss += fsm_weights[layer] * torch.mean((fog_factor_b / (hb * wb) - fog_factor_a / (ha * wa)) ** 2) / half / b_feature.size(0)

                    loss_fsm += layer_fsm_loss / 4.0

                total_loss = loss_mtl + args.lambda_fog * loss_fsm + args.lambda_con * loss_con
                total_loss = total_loss / args.iter_size
                total_loss.backward()

                # logging scalars (match style of original)
                if i_iter % 3 == 0:
                    loss_seg_sf_value += (loss_mtl.detach().cpu().item()) / args.iter_size
                    loss_seg_cw_value += (loss_mtl.detach().cpu().item()) / args.iter_size
                else:
                    loss_seg_sf_value += (loss_mtl.detach().cpu().item()) / args.iter_size

                if loss_fsm != 0:
                    loss_fsm_value += loss_fsm.data.cpu().numpy() / args.iter_size
                if loss_con != 0:
                    loss_con_value += loss_con.data.cpu().numpy() / args.iter_size

                wandb.log({"fsm loss": args.lambda_fog * loss_fsm_value}, step=i_iter)
                wandb.log({'mtl_loss': loss_mtl.detach().cpu().item()}, step=i_iter)
                wandb.log({'consistency loss': args.lambda_con * loss_con_value}, step=i_iter)
                wandb.log({'total_loss': total_loss.detach().cpu().item()}, step=i_iter)

                for opt in opts:
                    opt.step()

            # end Stage 2
            FogPassFilter1_optimizer.step()
            FogPassFilter2_optimizer.step()

        # snapshot frequency
        if i_iter < 20000:
            save_pred_every = 5000
            if args.modeltrain == 'train':
                save_pred_every = 2000
        else:
            save_pred_every = args.save_pred_every

        if i_iter >= args.num_steps_stop - 1:
            print('save model (final) ..')
            torch.save(model.state_dict(), osp.join(args.snapshot_dir, args.file_name + '_improve_' + str(args.num_steps_stop) + '.pth'))
            # also save edge head and MTL params
            torch.save({
                'edge_head': edge_head.state_dict(),
                'mtl_log_vars': mtl_loss_fn.log_vars.detach().cpu(),
            }, osp.join(args.snapshot_dir, args.file_name + '_improve_aux_' + str(args.num_steps_stop) + '.pth'))
            break

        if args.modeltrain != 'train':
            if i_iter == 5000:
                torch.save({'state_dict': model.state_dict(),
                            'edge_head': edge_head.state_dict(),
                            'fogpass1_state_dict': FogPassFilter1.state_dict(),
                            'fogpass2_state_dict': FogPassFilter2.state_dict(),
                            'train_iter': i_iter,
                            'args': args
                            }, osp.join(args.snapshot_dir, run_name) + '_fogpassfilter_improve_' + str(i_iter) + '.pth')

        if i_iter % save_pred_every == 0 and i_iter != 0:
            print('taking snapshot (intermediate) ...')
            save_dir = osp.join(f'./result/FIFO_model', args.file_name)
            if not os.path.exists(save_dir):
                os.makedirs(save_dir)

            torch.save({
                'state_dict': model.state_dict(),
                'edge_head': edge_head.state_dict(),
                'fogpass1_state_dict': FogPassFilter1.state_dict(),
                'fogpass2_state_dict': FogPassFilter2.state_dict(),
                'train_iter': i_iter,
                'args': args
            }, osp.join(args.snapshot_dir, run_name) + '_FIFO_improve_' + str(i_iter) + '.pth')


if __name__ == '__main__':
    main()
