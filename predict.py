import os
import gc
import cv2
import numpy as np
import argparse

import torch
import torchvision.transforms as T
import segmentation_models_pytorch as smp

def add_image_and_mask(image, outputs):
    # making color map
    label_color_map = [np.array([0, 0, 0])]
    for i in range(90):
        not_found = True
        while not_found:
            color = np.random.randint(0, 256, size=3)
            for j in label_color_map:
                if np.all(color == j):
                    break
                elif np.all(j == label_color_map[-1]):
                    label_color_map.append(color)
                    not_found = False
                  
    label_map = np.array(label_color_map)
                  
    def draw_segmentation_map(labels):
      red_map = np.zeros_like(labels).astype(np.uint8)
      green_map = np.zeros_like(labels).astype(np.uint8)
      blue_map = np.zeros_like(labels).astype(np.uint8)

      unique_labels = np.unique(labels)
      
      for label_num in unique_labels:
          index = labels == label_num
          red_map[index] = label_map[label_num, 0]
          green_map[index] = label_map[label_num, 1]
          blue_map[index] = label_map[label_num, 2]
          
      segmented_image = np.stack([red_map, green_map, blue_map], axis=2)
      return segmented_image

    mask = draw_segmentation_map(outputs)

    def image_overlay(image, segmented_image):
      alpha = 0.6 # how much transparency to apply
      beta = 1 - alpha # alpha + beta should equal 1
      gamma = 0 # scalar added to each sum
  
      image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
      segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_RGB2BGR)
    
      #  image = image + 0.6 * segmented_image
      cv2.addWeighted(segmented_image, alpha, image, beta, gamma, image)
      return image

    masked_image = image_overlay(image, mask)
    return masked_image
        

def main(
    args
):

    os.makedirs(args.output_dir, exist_ok=True)
  
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = smp.Unet(
        encoder_name="resnet18",        # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",     # use `imagenet` pre-trained weights for encoder initialization
        in_channels=3,                  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=args.num_classes,       # model output channels (number of classes in your dataset)
    )
    model.load_state_dict(torch.load(args.ckpt)['model_state'])
    model.to(device)
  
    image = cv2.imread(args.img_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    image = cv2.resize(image, (640, 608))
  
    tensor_image = T.ToTensor()(image)
    tensor_image = tensor_image.to(device)
    tensor_image = tensor_image.unsqueeze(0)
  
    
    model.eval()
    with torch.no_grad():
        out = model(tensor_image)
        _, pred = torch.max(out, dim=1)
    pred = pred[0].detach().cpu().numpy()

    masked_image = add_image_and_mask(image, pred)

    del out, image
    gc.collect()
    torch.cuda.empty_cache()

    cv2.imwrite(os.path.join(args.output_dir, 'output.jpg'), masked_image)
        

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description="PyTorch Segmentation Prediction", add_help=add_help)

    parser.add_argument("--ckpt", type=str, help="checkpoint to the saved model")
    parser.add_argument("--img_path", type=str, help="path to the image file")
    parser.add_argument("--output_dir", default="results", type=str, help="directory where images will be saved")
    parser.add_argument("--num_classes", default=91, type=int, help="number of classes in segmentation task")

    return parser

if __name__ == "__main__":
    args = get_args_parser().parse_args()
    main(args)
