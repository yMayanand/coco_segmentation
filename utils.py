import torch

class Meter:
    """
    This class helps to record metrics and losses.
    """
    def __init__(self, name='meter'):
        self.name = name
        self.reset()

    def reset(self):
        self.val = 0
        self.sum = 0
        self.count = 0
        self.avg = 0

    def update(self, val):
        self.val = val
        self.sum += val
        self.count += 1
        self.avg = self.sum / self.count

    def __str__(self):
        format = f"current_{self.name}: {self.val:6.4f}, average_{self.name}: {self.avg:6.4f}"
        return format

def iou_metric(pred, label):
    # intersection
    intersection  = torch.sum(pred == label)
    union = 2 * pred.numel() - intersection
    iou = intersection / union
    return iou

def train_one_batch(
    model, 
    batch, 
    criterion,
    optimizer,
    device
):
    image, mask = batch
    image = image.to(device)
    mask = mask.to(device)
    model.train()
    out = model(image)['out']
    loss = criterion(out, mask)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    return loss.item()

def predict_one_batch(
    model,
    batch,
    device
):
    image, mask = batch
    image = image.to(device)
    mask = mask.to(device)
    model.eval()
    with torch.no_grad():
        out = model(image)['out']
        _, pred = torch.max(out, dim=1)
        
    return pred.cpu().detach()

def validate_one_batch(
    model,
    batch,
    metric,
    device
):
    image, mask = batch
    image = image.to(device)
    mask = mask.to(device)
    model.eval()
    with torch.no_grad():
        out = model(image)['out']
        _, pred = torch.max(out, dim=1)
        model_metric = metric(pred, mask)
    return model_metric.item()





