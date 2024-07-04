import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from collections import OrderedDict

import torch_scatter
from torch_scatter import scatter_sum

from . import fastba
from . import altcorr
from . import lietorch
from .lietorch import SE3

from .extractor import BasicEncoder, BasicEncoder4
from .blocks import GradientClip, GatedResidual, SoftAgg

from .utils import *
from .ba import BA
from . import projective_ops as pops

autocast = torch.cuda.amp.autocast
import matplotlib.pyplot as plt

from transformers import Dinov2Backbone

DIM = 384

class Update(nn.Module):
    def __init__(self, p):
        super(Update, self).__init__()

        self.c1 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))

        self.c2 = nn.Sequential(
            nn.Linear(DIM, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM))
        
        self.norm = nn.LayerNorm(DIM, eps=1e-3)

        self.agg_kk = SoftAgg(DIM)
        self.agg_ij = SoftAgg(DIM)

        self.gru = nn.Sequential(
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            GatedResidual(DIM),
        )

        self.corr = nn.Sequential(
            nn.Linear(2*49*p*p, DIM),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
            nn.LayerNorm(DIM, eps=1e-3),
            nn.ReLU(inplace=True),
            nn.Linear(DIM, DIM),
        )

        self.d = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip())

        self.w = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Linear(DIM, 2),
            GradientClip(),
            nn.Sigmoid())


    def forward(self, net, inp, corr, flow, ii, jj, kk):
        """ update operator """

        net = net + inp + self.corr(corr)
        net = self.norm(net)

        ix, jx = fastba.neighbors(kk, jj)
        mask_ix = (ix >= 0).float().reshape(1, -1, 1)
        mask_jx = (jx >= 0).float().reshape(1, -1, 1)

        net = net + self.c1(mask_ix * net[:,ix])
        net = net + self.c2(mask_jx * net[:,jx])

        net = net + self.agg_kk(net, kk)
        net = net + self.agg_ij(net, ii*12345 + jj)

        net = self.gru(net)

        return net, (self.d(net), self.w(net), None)
    
class Scorer(nn.Module):
    def __init__(self, bins=512, patches_per_image=80) -> None:
        super().__init__()
        self.enc = nn.Sequential(
            nn.Conv2d(bins, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 1, kernel_size=3, padding=1),
            nn.MaxPool2d(kernel_size=4, stride=4),
            nn.Sigmoid())
        
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.InstanceNorm2d, nn.GroupNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
    
    def forward(self, x):
        b, n, c1, h1, w1 = x.shape # voxels (batch,n_frames,bins,h,w)
        x = x.view(b*n, c1, h1, w1)
        scores = self.enc(x)
        # scores = self.dec(scores)
        scores = F.interpolate(scores, size=(h1, w1), mode='bilinear')
        
        _, c2, h2, w2 = scores.shape
        return scores.view(b, n, h2, w2)

class Patchifier(nn.Module):
    def __init__(self, patch_size=3, patches_per_image=80):
        super(Patchifier, self).__init__()
        self.patch_size = patch_size
        self.fnet = BasicEncoder4(output_dim=128, norm_fn='instance')
        self.inet = BasicEncoder4(output_dim=DIM, norm_fn='none')
        # self.dinov2 = Dinov2Backbone.from_pretrained("facebook/dinov2-small", out_indices=[-1]).cuda().eval()
        # self.scorer = Scorer(512)
        self.xfeat = torch.hub.load('verlab/accelerated_features', 'XFeat', pretrained = True, top_k = patches_per_image)

    def __image_gradient(self, images):
        gray = ((images + 0.5) * (255.0 / 2)).sum(dim=2)
        dx = gray[...,:-1,1:] - gray[...,:-1,:-1]
        dy = gray[...,1:,:-1] - gray[...,:-1,:-1]
        g = torch.sqrt(dx**2 + dy**2)
        g = F.avg_pool2d(g, 4, 4)
        return g
    
    def get_top_n_coordinates(self, score_maps, top_n):
        b, num_maps, height, width = score_maps.shape
        grid_size = 10
        cell_height = height // grid_size
        cell_width = width // grid_size

        # Initialize tensors to store the results
        x_coords = torch.zeros((b, num_maps, top_n), dtype=torch.long)
        y_coords = torch.zeros((b, num_maps, top_n), dtype=torch.long)

        for batch_idx in range(b):
            for map_idx in range(num_maps):
                top_coordinates = []
                for i in range(grid_size):
                    for j in range(grid_size):
                        cell_scores = score_maps[batch_idx, map_idx, i*cell_height:(i+1)*cell_height, j*cell_width:(j+1)*cell_width]
                        max_score_idx = torch.argmax(cell_scores).item()
                        max_score_coords = (max_score_idx // cell_scores.shape[1], max_score_idx % cell_scores.shape[1])
                        global_coords = (i*cell_height + max_score_coords[0], j*cell_width + max_score_coords[1])
                        # Adjust coordinates to ensure they are within the range [1, height-1] and [1, width-1]
                        global_coords = (min(max(global_coords[0], 1), height-2), min(max(global_coords[1], 1), width-2))
                        top_coordinates.append((global_coords, score_maps[batch_idx, map_idx, global_coords[0], global_coords[1]].item()))

                # Sort coordinates by score in descending order
                top_coordinates = sorted(top_coordinates, key=lambda x: x[1], reverse=True)

                # Select top n coordinates
                top_n_coordinates = top_coordinates[:top_n]

                for idx, coord in enumerate(top_n_coordinates):
                    x_coords[batch_idx, map_idx, idx] = coord[0][0]
                    y_coords[batch_idx, map_idx, idx] = coord[0][1]

        return x_coords, y_coords

    def forward(self, images, patches_per_image=80, disps=None, gradient_bias=False, return_color=False):
        """ extract patches from input images """
        fmap = self.fnet(images) / 4.0
        imap = self.inet(images) / 4.0

        b, n, c, h, w = fmap.shape
        P = self.patch_size
        
        # with torch.no_grad():
        #     b1, n1, c1, h1, w1 = images.shape
        #     dinov2_features = self.dinov2(images.reshape(b1*n1, c1, h1, w1)).feature_maps[-1]
        #     dinov2_features = F.interpolate(dinov2_features, size=(fmap.shape[-2], fmap.shape[-1]), mode='bilinear')
        #     dinov2_features = dinov2_features.reshape(b, n, 384, fmap.shape[-2], fmap.shape[-1])
        # select_features = torch.cat([fmap, dinov2_features], dim=2)
        # scores = self.scorer(select_features)

        # bias patch selection towards regions with high gradient
        if gradient_bias:
            g = self.__image_gradient(images)
            x = torch.randint(1, w-1, size=[n, 3*patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, 3*patches_per_image], device="cuda")

            coords = torch.stack([x, y], dim=-1).float()
            g = altcorr.patchify(g[0,:,None], coords, 0).view(n, 3 * patches_per_image)
            
            ix = torch.argsort(g, dim=1)
            x = torch.gather(x, 1, ix[:, -patches_per_image:])
            y = torch.gather(y, 1, ix[:, -patches_per_image:])

        else:
            x = torch.randint(1, w-1, size=[n, patches_per_image], device="cuda")
            y = torch.randint(1, h-1, size=[n, patches_per_image], device="cuda")
            # x, y = self.get_top_n_coordinates(scores, patches_per_image)
            # x = x.squeeze(0)
            # y = y.squeeze(0)
            
            with torch.no_grad():
                b1, n1, c1, h1, w1 = images.shape
                inputs = images.reshape(b1*n1, c1, h1, w1)
                xfeat_features = self.xfeat.detectAndCompute(inputs, top_k=patches_per_image)
                xfeatures = []
                for xfeat in xfeat_features:
                    kp = xfeat['keypoints']
                    if kp.shape[0] < patches_per_image:
                        # fill with random
                        print('filling with random')
                        x = torch.randint(1, w-1, size=[patches_per_image - kp.shape[0]], device="cuda")
                        y = torch.randint(1, h-1, size=[patches_per_image - kp.shape[0]], device="cuda")
                        kp = torch.cat([kp, torch.stack([x, y], dim=-1)], dim=0)
                    if kp.shape[0] > patches_per_image:
                        kp = kp[:patches_per_image]
                    xfeatures.append(kp)
                xfeatures = torch.stack(xfeatures, dim=0)
            # del xfeat_features
            x = xfeatures[:, :, 0]
            y = xfeatures[:, :, 1]
            x = torch.clamp(x, 1, w-2)
            y = torch.clamp(y, 1, h-2)
        
        coords = torch.stack([x, y], dim=-1).float()
        imap = altcorr.patchify(imap[0], coords, 0).view(b, -1, DIM, 1, 1)
        gmap = altcorr.patchify(fmap[0], coords, P//2).view(b, -1, 128, P, P)

        if return_color:
            clr = altcorr.patchify(images[0], 4*(coords + 0.5), 0).view(b, -1, 3)

        if disps is None:
            disps = torch.ones(b, n, h, w, device="cuda")

        grid, _ = coords_grid_with_index(disps, device=fmap.device)
        patches = altcorr.patchify(grid[0], coords, P//2).view(b, -1, 3, P, P)

        index = torch.arange(n, device="cuda").view(n, 1)
        index = index.repeat(1, patches_per_image).reshape(-1)

        if return_color:
            return fmap, gmap, imap, patches, index, clr

        return fmap, gmap, imap, patches, index


class CorrBlock:
    def __init__(self, fmap, gmap, radius=3, dropout=0.2, levels=[1,4]):
        self.dropout = dropout
        self.radius = radius
        self.levels = levels

        self.gmap = gmap
        self.pyramid = pyramidify(fmap, lvls=levels)

    def __call__(self, ii, jj, coords):
        corrs = []
        for i in range(len(self.levels)):
            corrs += [ altcorr.corr(self.gmap, self.pyramid[i], coords / self.levels[i], ii, jj, self.radius, self.dropout) ]
        return torch.stack(corrs, -1).view(1, len(ii), -1)


class VONet(nn.Module):
    def __init__(self, use_viewer=False, patches_per_image=80):
        super(VONet, self).__init__()
        self.P = 3
        self.patchify = Patchifier(self.P, patches_per_image=patches_per_image)
        self.update = Update(self.P)

        self.DIM = DIM
        self.RES = 4


    @autocast(enabled=False)
    def forward(self, images, poses, disps, intrinsics, M=1024, STEPS=12, P=1, structure_only=False, rescale=False):
        """ Estimates SE3 or Sim3 between pair of frames """

        images = 2 * (images / 255.0) - 0.5
        intrinsics = intrinsics / 4.0
        disps = disps[:, :, 1::4, 1::4].float()

        fmap, gmap, imap, patches, ix = self.patchify(images, disps=disps)

        corr_fn = CorrBlock(fmap, gmap)

        b, N, c, h, w = fmap.shape
        p = self.P

        patches_gt = patches.clone()
        Ps = poses

        d = patches[..., 2, p//2, p//2]
        patches = set_depth(patches, torch.rand_like(d))

        kk, jj = flatmeshgrid(torch.where(ix < 8)[0], torch.arange(0,8, device="cuda"))
        ii = ix[kk]

        imap = imap.view(b, -1, DIM)
        net = torch.zeros(b, len(kk), DIM, device="cuda", dtype=torch.float)
        
        Gs = SE3.IdentityLike(poses)

        if structure_only:
            Gs.data[:] = poses.data[:]

        traj = []
        bounds = [-64, -64, w + 64, h + 64]
        
        while len(traj) < STEPS:
            Gs = Gs.detach()
            patches = patches.detach()

            n = ii.max() + 1
            if len(traj) >= 8 and n < images.shape[1]:
                if not structure_only: Gs.data[:,n] = Gs.data[:,n-1]
                kk1, jj1 = flatmeshgrid(torch.where(ix  < n)[0], torch.arange(n, n+1, device="cuda"))
                kk2, jj2 = flatmeshgrid(torch.where(ix == n)[0], torch.arange(0, n+1, device="cuda"))

                ii = torch.cat([ix[kk1], ix[kk2], ii])
                jj = torch.cat([jj1, jj2, jj])
                kk = torch.cat([kk1, kk2, kk])

                net1 = torch.zeros(b, len(kk1) + len(kk2), DIM, device="cuda")
                net = torch.cat([net1, net], dim=1)

                if np.random.rand() < 0.1:
                    k = (ii != (n - 4)) & (jj != (n - 4))
                    ii = ii[k]
                    jj = jj[k]
                    kk = kk[k]
                    net = net[:,k]

                patches[:,ix==n,2] = torch.median(patches[:,(ix == n-1) | (ix == n-2),2])
                n = ii.max() + 1

            coords = pops.transform(Gs, patches, intrinsics, ii, jj, kk)
            coords1 = coords.permute(0, 1, 4, 2, 3).contiguous()

            corr = corr_fn(kk, jj, coords1)
            net, (delta, weight, _) = self.update(net, imap[:,kk], corr, None, ii, jj, kk)

            lmbda = 1e-4
            target = coords[...,p//2,p//2,:] + delta

            ep = 10
            for itr in range(2):
                Gs, patches = BA(Gs, patches, intrinsics, target, weight, lmbda, ii, jj, kk, 
                    bounds, ep=ep, fixedp=1, structure_only=structure_only)

            kl = torch.as_tensor(0)
            dij = (ii - jj).abs()
            k = (dij > 0) & (dij <= 2)

            coords = pops.transform(Gs, patches, intrinsics, ii[k], jj[k], kk[k])
            coords_gt, valid, _ = pops.transform(Ps, patches_gt, intrinsics, ii[k], jj[k], kk[k], jacobian=True)

            traj.append((valid, coords, coords_gt, Gs[:,:n], Ps[:,:n], kl))

        return traj

