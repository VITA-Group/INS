import torch
from torch import nn
from torch.nn import functional as F

from model.vgg import Vgg16

def interpolate(batch, mode='bilinear', size=[224, 224]):
    x = F.interpolate(batch, size)
    return x

def normalize_batch(batch):
    # normalize using imagenet mean and std
    mean = batch.new_tensor([0.485, 0.456, 0.406]).view(-1, 1, 1)
    std = batch.new_tensor([0.229, 0.224, 0.225]).view(-1, 1, 1)
    batch = batch.div_(255.0)
    return (batch - mean) / std

def gram_matrix(y):
    (b, ch, h, w) = y.size()
    features = y.view(b, ch, w * h)
    features_t = features.transpose(1, 2)
    gram = features.bmm(features_t) / (ch * h * w)
    return gram

class IDRLoss(nn.Module):
    def __init__(self, rgb_weight, eikonal_weight, mask_weight, alpha, perceptual_weight, content_weight, style_weight):
        super().__init__()
        self.rgb_weight = float(rgb_weight)
        self.eikonal_weight = float(eikonal_weight)
        self.mask_weight = float(mask_weight)
        self.alpha = alpha
        self.l1_loss = nn.L1Loss(reduction='sum')

        self.perceptual_weight = float(perceptual_weight)
        self.content_weight = float(content_weight)
        self.style_weight = float(style_weight)
        
        self.VGG = Vgg16(requires_grad=False)

    def get_rgb_loss(self,rgb_values, rgb_gt, network_object_mask, object_mask):
        if (network_object_mask & object_mask).sum() == 0:
            return torch.tensor(0.0).cuda().float()

        rgb_values = rgb_values[network_object_mask & object_mask]
        rgb_gt = rgb_gt.reshape(-1, 3)[network_object_mask & object_mask]
        rgb_loss = self.l1_loss(rgb_values, rgb_gt) / float(object_mask.shape[0])
        return rgb_loss

    def get_eikonal_loss(self, grad_theta):
        if grad_theta.shape[0] == 0:
            return torch.tensor(0.0).cuda().float()

        eikonal_loss = ((grad_theta.norm(2, dim=1) - 1) ** 2).mean()
        return eikonal_loss

    def get_mask_loss(self, sdf_output, network_object_mask, object_mask):
        mask = ~(network_object_mask & object_mask)
        if mask.sum() == 0:
            return torch.tensor(0.0).cuda().float()
        sdf_pred = -self.alpha * sdf_output[mask]
        gt = object_mask[mask].float()
        mask_loss = (1 / self.alpha) * F.binary_cross_entropy_with_logits(sdf_pred.squeeze(-1), gt, reduction='sum') / float(object_mask.shape[0])
        return mask_loss

    def get_style_loss(self, img_res, rgb_values, rgb_gt, style_img, object_mask):
        bs = rgb_gt.shape[0]
        rgb_pred = rgb_values.reshape(bs, img_res, img_res, rgb_values.shape[-1]) * 255.0
        target_s = rgb_gt.reshape(bs, img_res, img_res, rgb_gt.shape[-1]) * 255.0
        style_s = style_img * 255.0
        mask = object_mask.float().reshape(bs, img_res, img_res, 1)

        # [B, H, W, C] -> [B, C, H, W]
        rgb_pred = rgb_pred.permute(0, 3, 1, 2)
        target_s = target_s.permute(0, 3, 1, 2)
        style_s = style_s.permute(0, 3, 1, 2)
        mask = mask.permute(0, 3, 1, 2)

        rgb_pred = torch.clamp(rgb_pred, 0., 255., out=None)
        target_s = torch.clamp(target_s, 0., 255., out=None)
        style_s = torch.clamp(style_s, 0., 255., out=None)
        mask = torch.clamp(mask, 0., 1., out=None)
        
        _rgb_pred = interpolate(rgb_pred)
        _target_s = interpolate(target_s)
        _style_s = interpolate(style_s)
        _mask = interpolate(mask)
    
        rgb_pred_features = self.VGG(normalize_batch(_rgb_pred) * _mask)
        rgb_gt_features = self.VGG(normalize_batch(_target_s) * _mask)
        style_features = self.VGG(normalize_batch(_style_s))


        def img2mse(x, y, mask=None):
            if mask is None:
                return torch.mean((x - y) ** 2)
            else:
                return torch.sum((x*mask - y*mask) ** 2) / (torch.sum(mask) + 1e-5)

        # content loss
        content_loss = img2mse(rgb_gt_features['relu2_2'], rgb_pred_features['relu2_2'])

        # style loss
        style_loss = 0.
        gram_style = []
        for k, y in style_features.items():
            gram_style.append(gram_matrix(y))
        
        gram_pred = []
        for k, y in rgb_pred_features.items():
            gram_pred.append(gram_matrix(y))

        for gm_y, gm_s in zip(gram_pred, gram_style):
            style_loss += img2mse(gm_y, gm_s[:1, :, :])

        return content_loss, style_loss


    def forward(self, model_outputs, ground_truth, style_img=None):
        rgb_gt = ground_truth['rgb'].cuda()
        network_object_mask = model_outputs['network_object_mask']
        object_mask = model_outputs['object_mask']

        rgb_loss = self.get_rgb_loss(model_outputs['rgb_values'], rgb_gt, network_object_mask, object_mask)
        mask_loss = self.get_mask_loss(model_outputs['sdf_output'], network_object_mask, object_mask)
        eikonal_loss = self.get_eikonal_loss(model_outputs['grad_theta'])
        content_loss = style_loss = 0.
        if ground_truth['style_img'] is not None:
            img_res = ground_truth['img_res']
            style_img = ground_truth['style_img'].cuda()
            content_loss, style_loss = self.get_style_loss(img_res, model_outputs['rgb_values'], rgb_gt, style_img, object_mask)

        loss = self.rgb_weight * rgb_loss + \
               self.eikonal_weight * eikonal_loss + \
               self.mask_weight * mask_loss + \
               self.perceptual_weight * self.content_weight * content_loss + \
               self.perceptual_weight * self.style_weight * style_loss

        return {
            'loss': loss,
            'rgb_loss': rgb_loss,
            'eikonal_loss': eikonal_loss,
            'mask_loss': mask_loss,
            'content_loss': content_loss,
            'style_loss': style_loss,
        }
