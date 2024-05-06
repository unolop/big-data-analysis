import torch
import torch.nn as nn
import torch.nn.functional as F

class FocalLoss(nn.Module):
    def __init__(self, gamma=0, weight = None, alpha=None, size_average=True, device='cpu'):
        super(FocalLoss, self).__init__()
        """
        gamma(int) : focusing parameter.
        alpha(list) : alpha-balanced term.
        size_average(bool) : whether to apply reduction to the output.
        """
        self.gamma = gamma
        self.alpha = alpha
        self.size_average = size_average
        self.device = device
        self.weight = weight

    def forward(self, input, target):
        # input : N * C (btach_size, num_class)
        # target : N (batch_size)

        CE = F.cross_entropy(input, target, weight=self.weight, reduction='none')  # -log(pt)
        pt = torch.exp(-CE)  # pt
        loss = (1 - pt) ** self.gamma * CE  # -(1-pt)^rlog(pt)

        if self.alpha is not None:
            alpha = torch.tensor(self.alpha, dtype=torch.float).to(self.device)
            # in case that a minority class is not selected when mini-batch sampling
            if len(self.alpha) != len(torch.unique(target)):
                temp = torch.zeros(len(self.alpha)).to(self.device)
                temp[torch.unique(target)] = alpha.index_select(0, torch.unique(target))
                alpha_t = temp.gather(0, target)
                loss = alpha_t * loss
            else:
                alpha_t = alpha.gather(0, target)
                loss = alpha_t * loss

        if self.size_average:
            loss = torch.mean(loss)

        return loss    
    

def weighted_CrossEntropyLoss(output, target, classes_weights, device):
    cr = nn.CrossEntropyLoss(weight=torch.tensor(classes_weights).to(device))
    return cr(output, target)


def set_loss(args,device_weight = None):
    if args.augmentation is None:
        loss = nn.CrossEntropyLoss()
    else:
        loss = nn.CrossEntropyLoss(weight=device_weight)
    return loss