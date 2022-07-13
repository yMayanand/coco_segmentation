import torch
import gc
import GPUtil

def get_gpu_usage():
    gpu = GPUtil.getGPUs()[0]
    gpu_load = gpu.load * 100
    gpu_memory_util = gpu.memoryUtil * 100
    return gpu_load, gpu_memory_util


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
    intersection  = (pred == label).float()
    background_mask = (label != 0).float()
    intersection = torch.sum(intersection * background_mask)
    # union
    union = 2 * pred.numel() - intersection - (2 * torch.sum(label == 0)) # removing pixels with '0' label
    iou = intersection / union
    return iou

def train_one_batch(
    model, 
    batch, 
    criterion,
    optimizer,
    device,
    global_step=None,
    writer=None
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
    if writer is not None:
        load, mem_util = get_gpu_usage()
        writer.add_scalar('gpu_load', load, global_step)
        writer.add_scalar('gpu_mem_util', mem_util, global_step)
    del out, image, mask
    gc.collect()
    torch.cuda.empty_cache()
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

    del out, image, mask
    gc.collect()
    torch.cuda.empty_cache()
        
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

    del out, image, mask
    gc.collect()
    torch.cuda.empty_cache()
    
    return model_metric.item()





