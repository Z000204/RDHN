import torch
import os
import math
import timm

from torch import nn
from torch.optim import lr_scheduler
import settings
import torch.optim as optim
from .vit_model import vit_large_patch16_224_in21k as create_model

vit_model = create_model(num_classes=24, has_logits=False).to(torch.device("cuda:0"))


if settings.weights != "":
    assert os.path.exists(settings.weights), "weights file: '{}' not exist.".format(settings.weights)
    weights_dict = torch.load(settings.weights, map_location=settings.device)
    # 删除不需要的权重
    del_keys = ['head.weight', 'head.bias'] if vit_model.has_logits \
        else ['pre_logits.fc.weight', 'pre_logits.fc.bias', 'head.weight', 'head.bias']
    for k in del_keys:
        del weights_dict[k]
    print(vit_model.load_state_dict(weights_dict, strict=False))

if settings.freezelayers:
    for name, para in vit_model.named_parameters():
        # 除head, pre_logits外，其他权重全部冻结
        if "head" not in name and "pre_logits" not in name:
            para.requires_grad_(False)
        else:
            print("training {}".format(name))

pg = [p for p in vit_model.parameters() if p.requires_grad]
optimizer = optim.SGD(pg, lr=settings.lrr, momentum=0.9, weight_decay=5E-5)
# Scheduler https://arxiv.org/pdf/1812.01187.pdf
lf = lambda x: ((1 + math.cos(x * math.pi / settings.epochs)) / 2) * (1 - settings.lrf) + settings.lrf  # cosine
scheduler = lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

# vitmodel = vit_model.forward_features()

class image_net(nn.Module):
    def __init__(self):
        super(image_net, self).__init__()
        # self.features = vitmodel

        print('sucess init')

    def forward(self, x):
        # f_x = self.features(x)
        f_x = vit_model.forward_features(x)
        # f_x = f_x.view(f_x.size(0), -1)
        # f_x = self.linear_layers(f_x)

        return f_x