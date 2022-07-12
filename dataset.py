import os
import cv2
import numpy as np
from copy import deepcopy
from pycocotools.coco import COCO

class Dataset:
    """
    creates coco dataset 
    """
    def __init__(self, root, image_set, transforms=None):
        PATH = {
            "train": ("train2017", os.path.join("annotations", "instances_train2017.json")),
            "val": ("val2017", os.path.join("annotations", "instances_val2017.json")),
        }

        img_folder, ann_file = PATH[image_set]
        self.img_folder = os.path.join(root, img_folder)
        ann_file = os.path.join(root, ann_file)

        self.transform = transforms

        self.coco = COCO(ann_file)
        self.img_ids = self.coco.getImgIds()
        self.filter_empty()


    def __len__(self):
        return len(self.img_ids)

    def get_image(self, idx):
        img_data = self.coco.loadImgs(self.img_ids[idx])[0]
        path = os.path.join(self.img_folder, img_data['file_name'])
        image = cv2.imread(path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        return image

    def get_anno(self, idx):
        anno_ids = self.coco.getAnnIds(imgIds=self.img_ids[idx], iscrowd=None)
        anno = self.coco.loadAnns(anno_ids)
        # convert annotations to 2d array binary mask
        masks = []
        cat_ids = []
        for i in anno:
            mask = self.coco.annToMask(i)
            masks.append(mask)
            cat_ids.append(i['category_id'])

        # create category tensor
        cat_ids = np.array(cat_ids)

        masks = np.stack(masks) 
        mask = masks * cat_ids[:, None, None] # shape (num_instances, h, w)
        mask = np.max(mask, axis=0) # merge all instances
        mask[masks.sum(axis=0) > 1] = 0 # ignore overlapping part of instances
        return mask
    
    def filter_empty(self):
        print('removing samples without annotations...')
        for id in deepcopy(self.img_ids):
            anno_ids = self.coco.getAnnIds(imgIds=id, iscrowd=None)
            if len(anno_ids) < 1:
                self.img_ids.remove(id)
        

    def __getitem__(self, idx):
        image, mask = self.get_image(idx), self.get_anno(idx)
        if self.transform is not None:
            image, mask = self.transform(image, mask)
        return image, mask