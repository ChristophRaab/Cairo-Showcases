from torch import nn
import torch 
class CenterLoss(nn.Module):
    """Center loss.
    
    Reference:
    Wen et al. A Discriminative Feature Learning Approach for Deep Face Recognition. ECCV 2016.
    
    Args:
        num_classes (int): number of classes.
        feat_dim (int): feature dimension.
    """
    def __init__(self, num_classes=10, feat_dim=2, cuda=":1"):
        super(CenterLoss, self).__init__()
        self.num_classes = num_classes
        self.feat_dim = feat_dim
        self.cuda = cuda

        if self.cuda is not None:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim).to(cuda))
        else:
            self.centers = nn.Parameter(torch.randn(self.num_classes, self.feat_dim))

    def forward(self, x, labels):
        """
        Args:
            x: feature matrix with shape (batch_size, feat_dim).
            labels: ground truth labels with shape (batch_size).
        """
        batch_size = x.size(0)
        distmat = torch.pow(x, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(1, -2, x, self.centers.t())

        classes = torch.arange(self.num_classes).long()
        if self.cuda: classes = classes.to(self.cuda)
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return loss


def slda(x,y=None):

    mean = torch.mean(x,0)

    classes = torch.unique(y)
    din = torch.tensor([0]).to(x.device)
    dbet = torch.tensor([0]).to(x.device)
    ns = 1/len(x)

    for c in classes:
        cx = x[y==c]
        c_mean = torch.mean(cx,0)

        c_din = torch.norm(cx-c_mean)**2
        din = din+c_din

        dbet = dbet + len(cx) * torch.norm(c_mean-mean)**2
    din = din * ns
    dbet = dbet * ns

    loss = (dbet / (din + dbet))
    return loss
