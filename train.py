from utils import *
from transforms import NumpyToTensor, Resize, Compose, RandomResizedCrop
from dataset import Dataset
import argparse
import math
import time
import os

import torch
import torch.nn as nn
import segmentation_models_pytorch as smp
from torch.jit import script
from torchvision import models
from torch.utils.tensorboard import SummaryWriter


def main(args):
    writer = SummaryWriter(comment=args.log_folder)

    os.makedirs(args.model_dir, exist_ok=True)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # train transforms
    train_tfms = Compose([
        NumpyToTensor(),
        RandomResizedCrop(args.size, scale=(0.4, 1.), ratio=(0.95, 1.))
        #script(Resize((args.size, args.size)))
    ])

    # validation transforms
    val_tfms = Compose([
        NumpyToTensor(),
        #RandomResizedCrop(256, scale=(0.4, 1.), ratio=(0.95, 1.))
        #script(Resize((args.size, args.size)))
    ])

    # main dataset
    ds = Dataset(args.root_dir, args.image_set, transforms=train_tfms)

    # splitting the main dataset
    val_split = math.floor(args.val_split * len(ds))
    train_split = len(ds) - val_split

    train_ds, val_ds = torch.utils.data.random_split(ds, (train_split, val_split))
    val_ds.transform = val_tfms

    # segmentation model
    # model = models.segmentation.fcn_resnet50(num_classes=args.num_classes, weights_backbone=models.ResNet50_Weights.DEFAULT)

    model = smp.Unet(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=args.num_classes,                     # model output channels (number of classes in your dataset)
    )
    model.to(device)

    if args.finetune:
        freeze_backbone(model)
    
    if args.resume:
        if os.path.exists(args.resume):
            state_dict = torch.load(args.resume)
            model.load_state_dict(state_dict['model_state'])


    # dataloaders
    train_dl = torch.utils.data.DataLoader(
        train_ds, 
        batch_size=args.batch_size,  
        shuffle=True,
        num_workers=2
    )

    val_dl = torch.utils.data.DataLoader(val_ds, batch_size=1, num_workers=2)

    # loss function
    criterion = nn.CrossEntropyLoss(ignore_index=255)

    # metric to monitor
    metric = iou_metric

    # optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr)
    
    # loss Meter
    loss_meter = Meter('train_ce_loss')

    # loss Meter
    val_loss_meter = Meter('val_ce_loss')

    # metric meter
    metric_meter = Meter('val_iou_metric')

    # start time
    start_time = time.time()
    print('training started...')
    for epoch in range(args.epochs):
        train_start_time = time.time()

        # training phase
        for i, batch in enumerate(train_dl):
            global_step = (len(train_dl) * epoch) + i
            curr_loss = train_one_batch(
                model, batch, 
                criterion, optimizer, 
                device, global_step=global_step, 
                writer=writer
            )

            loss_meter.update(curr_loss)
            writer.add_scalar('train_loss', curr_loss, global_step)
        
        train_end_time = time.time()
        # adding number of training images processed
        processing_time = len(train_ds) / (train_end_time - train_start_time)
        writer.add_scalar('train_images_processed', processing_time, epoch)

        val_start_time = time.time()

        # validation phase
        for i, batch in enumerate(val_dl):
            curr_metric, val_loss = validate_one_batch(model, batch, metric, criterion, device)
            val_loss_meter.update(val_loss)
            metric_meter.update(curr_metric)
            #global_step = (len(val_dl) * epoch) + i
        writer.add_scalar('val_metric', metric_meter.avg, global_step)
        writer.add_scalar('val_loss', val_loss_meter.avg, global_step)
        
        
        val_end_time = time.time()
        # adding number of training images processed
        processing_time = len(val_ds) / (val_end_time - val_start_time)
        writer.add_scalar('val_images_processed', processing_time, epoch)
        print(f"epoch: {epoch:04d}, train_loss: {loss_meter}, val_metric: {metric_meter}")
    
        dir_name = f"{time.strftime('%Y-%m-%d-%H')}_model_checkpoints"
        session_dir = os.path.join(args.model_dir, dir_name)
        if not os.path.exists(session_dir):
            os.makedirs(session_dir)
        
        save_path = os.path.join(session_dir, f"epoch_{epoch}.pt")

        state_dict = {'model_state': model.state_dict()}
        torch.save(state_dict, save_path)

    end_time = time.time()
    print(f"total time taken to finish trainig: {end_time - start_time}")


    

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Training", add_help=add_help)

    parser.add_argument("--root_dir", default="./", type=str, help="root directory path")
    parser.add_argument("--image_set", default="val", type=str, help="image set name 'train' or 'val'")
    parser.add_argument("--val_split", default=0.2, type=float, help="validation split percentage")
    parser.add_argument("--size", default=256, type=int, help="training image size")
    parser.add_argument("--num_classes", default=91, type=int, help="number of classes in segmentation task")
    parser.add_argument("--batch_size", default=16, type=int, help="batch size for training")
    parser.add_argument("--epochs", default=10, type=int, help="number of epochs to train")
    parser.add_argument("--lr", default=1e-3, type=float, help="learning rate for optimizer")
    parser.add_argument("--log_folder", default='general', type=str, help="folder name for tensorboard logging")
    parser.add_argument("--model_dir", default='./checkpoints', type=str, help="directory path to save our model")
    parser.add_argument("--finetune", default=True, type=bool, help="flag to start finetuning")
    parser.add_argument("--resume", default=None, type=str, help="path to the checkpoint")


    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
