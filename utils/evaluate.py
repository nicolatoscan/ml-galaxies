from typing import Dict
import torch

def evaluate(yt: torch.Tensor, yp: torch.Tensor, num_classes=10) -> Dict[str, float]:
    C=(yt*num_classes+yp).bincount(minlength=num_classes**2).view(num_classes,num_classes).float()
    return {
        'Acc': C.diag().sum().item() / yt.shape[0],
        'mAcc': (C.diag()/C.sum(-1)).mean().item(),
        'mIoU': (C.diag()/(C.sum(0)+C.sum(1)-C.diag())).mean().item()
    }