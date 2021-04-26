import torch
import torch.nn.functional as F
import xmltodict
import torch.utils.data as data
from PIL import Image
import cv2
import math
import random
import numpy as np

from utils.utils_module import TwoCropsTransform

def get_density_map_gaussian(h, w,  points, ratio_h=1., ratio_w=1., fixed_value=3):
    density_map = np.zeros([h, w], dtype=np.float32)
    num_gt = np.squeeze(points).shape[0]
    if num_gt == 0:
        return density_map

    for idx, p in enumerate(points):
        p = np.round(p).astype(int)
        p[0], p[1] = min(h-1, math.floor(p[1] * ratio_h)), min(w-1, math.floor(p[0] * ratio_w))
        sigma = fixed_value
        sigma = max(1, sigma)
        gaussian_radius = sigma * 3
        gaussian_map = np.multiply(
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma),
            cv2.getGaussianKernel(gaussian_radius*2+1, sigma).T
        )
        x_left, x_right, y_up, y_down = 0, gaussian_map.shape[1], 0, gaussian_map.shape[0]
        # cut the gaussian kernel
        if p[1] < 0 or p[0] < 0:
            continue
        if p[1] < gaussian_radius:
            x_left = gaussian_radius - p[1]
        if p[0] < gaussian_radius:
            y_up = gaussian_radius - p[0]
        if p[1] + gaussian_radius >= w:
            x_right = gaussian_map.shape[1] - (gaussian_radius + p[1] - w) - 1
        if p[0] + gaussian_radius >= h:
            y_down = gaussian_map.shape[0] - (gaussian_radius + p[0] - h) - 1
        density_map[
            max(0, p[0]-gaussian_radius):min(h, p[0]+gaussian_radius+1),
            max(0, p[1]-gaussian_radius):min(w, p[1]+gaussian_radius+1)
        ] += gaussian_map[y_up:y_down, x_left:x_right]
    return density_map

def makeGaussian(size=None, center=None):
    """ Make a square gaussian kernel.

    size is the length of a side of the square
    fwhm is full-width-half-maximum, which
    can be thought of as an effective radius.
    """
    # x = np.arange(0, size, 1, float)
    # y = x[:, np.newaxis]
    # x0 = center[0]
    # y0 = center[1]
    # fwhm = 3
    # dist = np.exp(-4 * np.log(2) * ((x - x0) ** 2 + (y - y0) ** 2) / fwhm ** 2)
    x, y = np.meshgrid(np.linspace(-1, 1, size), np.linspace(-1, 1, size))
    d = np.sqrt(x * x + y * y)
    sigma, mu = 1.0, 0.0
    g = np.exp(-((d - mu) ** 2 / (2.0 * sigma ** 2)))
    g /= g.sum()
    return g


def gaussian_apply(doc, img):
    label = np.zeros((img.size[1] * 2, img.size[0] * 2), np.float32)
    x_start = img.size[0] // 2
    y_start = img.size[1] // 2
    if 'vehicle' in doc['annotation'].keys():
        if type(doc['annotation']['vehicle']) is not list:
            bbox = doc['annotation']['vehicle']['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])
            
            center_x = (xmax - xmin) // 2
            center_y = (ymax - ymin) // 2
#             density_mp = makeGaussian(size=5, center=(center_x, center_y))
#             label[center_y - 2:center_y+3, center_x - 2:center_x+3] += density_mp
            density_mp = makeGaussian(size=min((xmax - xmin), (ymax - ymin)), center=((xmax - xmin) // 2, (ymax - ymin) // 2))

            if xmax - xmin < ymax - ymin:
                label[y_start + ymin: y_start + ymin + xmax - xmin, x_start + xmin: x_start + xmax] += density_mp
            else:
                label[y_start + ymin:y_start + ymax, x_start + xmin: x_start + xmin + ymax - ymin] += density_mp
        else:
            for vehicle in doc['annotation']['vehicle']:
                bbox = vehicle['bndbox']
                xmin = int(bbox['xmin'])
                ymin = int(bbox['ymin'])
                xmax = int(bbox['xmax'])
                ymax = int(bbox['ymax'])
#                 center_x = (xmax - xmin) // 2
#                 center_y = (ymax - ymin) // 2
# #                 min(xmax - xmin, ymax - ymin)
#                 density_mp = makeGaussian(size=5, center=(center_x, center_y))
#                 label[center_y - 2:center_y+3, center_x - 2:center_x+3] += density_mp
                density_mp = makeGaussian(size=min((xmax - xmin), (ymax - ymin)), center=((xmax - xmin) // 2, (ymax - ymin) // 2))
                if xmax - xmin < ymax - ymin:
                    label[y_start + ymin: y_start + ymin + xmax - xmin, x_start + xmin: x_start + xmax] += density_mp
                else:
                    label[y_start + ymin:y_start + ymax, x_start + xmin: x_start + xmin + ymax - ymin] += density_mp

    return label[y_start: 3 * y_start, x_start: 3 * x_start]

def get_points(doc):
    points = []
    if 'vehicle' in doc['annotation'].keys():
        if type(doc['annotation']['vehicle']) is not list:
            bbox = doc['annotation']['vehicle']['bndbox']
            xmin = int(bbox['xmin'])
            ymin = int(bbox['ymin'])
            xmax = int(bbox['xmax'])
            ymax = int(bbox['ymax'])
            center_x = (xmax - xmin) // 2 + xmin
            center_y = (ymax - ymin) // 2 + ymin
            points.append((center_x, center_y))
            
        else:
            for vehicle in doc['annotation']['vehicle']:
                bbox = vehicle['bndbox']
                xmin = int(bbox['xmin'])
                ymin = int(bbox['ymin'])
                xmax = int(bbox['xmax'])
                ymax = int(bbox['ymax'])
                center_x = (xmax - xmin) // 2 + xmin
                center_y = (ymax - ymin) // 2 + ymin
                points.append((center_x, center_y))
    return points

def read_xml(path):
    with open(path, 'r') as xml_d:
        ss = xml_d.read()
        try:
            doc = xmltodict.parse(ss)
        except:
            try:
                ss = ss.replace("&", "")
                doc = xmltodict.parse(ss)
            except:
                print(path + " cannot be read")
    return doc

class CitycamParams(object):
    num_channels = 3
    image_size = 224 # 384, 216
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]
    num_cls = 1
    target_transform = None
    
class CITYCAM(data.Dataset):
    def __init__(self, root, num_cls, transform, data=None):
        self.root = root
        self.transform = transform
        self.classes = []
        for i in range(num_cls):
            self.classes.append(i)
        self.num_cls = num_cls
        self.images, self.labels, self.masks = data['ids'], data['labels'], data['masks']
        self.transform = TwoCropsTransform(self.transform)
        
    def __getitem__(self, index, debug=False):
        
        image_pth = self.images[index % len(self.images)]
        label_pth = self.labels[index % len(self.images)]
        mask_pth = self.masks[index % len(self.images)]
        img = Image.open(image_pth).convert('RGB')
        gt_points = get_points(read_xml(label_pth))
        gt_map = torch.from_numpy(get_density_map_gaussian(
            *np.array(img).shape[0:2],  
            gt_points, 
            ratio_h=1., ratio_w=1., 
            fixed_value=3
        ))
        gt_map = gt_map.reshape(1, 1, *gt_map.shape)
        mask = (torch.from_numpy(np.array(Image.open(mask_pth).convert('F'))) > 0).to(torch.float32)
        mask = mask.reshape(1, 1, *mask.shape)

        rand_index = random.randint(0, len(self.images) - 1)
        rand_image_pth = self.images[rand_index % len(self.images)]
        rand_label_pth = self.labels[rand_index % len(self.images)]
        rand_mask_pth = self.masks[rand_index % len(self.images)]
        rand_img = Image.open(rand_image_pth).convert('RGB')
        rand_gt_points = get_points(read_xml(rand_label_pth))
        rand_gt_map = torch.from_numpy(get_density_map_gaussian(
            *np.array(rand_img).shape[0:2],  
            rand_gt_points, 
            ratio_h=1., ratio_w=1., 
            fixed_value=3
        ))
        rand_gt_map = rand_gt_map.reshape(1, 1, *rand_gt_map.shape)
        rand_mask = (torch.from_numpy(np.array(Image.open(rand_mask_pth).convert('F'))) > 0).to(torch.float32)
        rand_mask = rand_mask.reshape(1, 1, *rand_mask.shape)
        img_q, img_k = self.transform(img)
        rand_img_q, rand_img_k = self.transform(rand_img)
        # resize
        gt_map = F.interpolate(gt_map, img_q.shape[1:3] , mode='nearest').squeeze(dim=0)
        
        rand_gt_map = F.interpolate(rand_gt_map, rand_img_q.shape[1:3] , mode='nearest').squeeze(dim=0)
        mask = F.interpolate(mask, img_q.shape[1:3], mode='nearest').squeeze(dim=0)
        rand_mask = F.interpolate(rand_mask, rand_img_q.shape[1:3], mode='nearest').squeeze(dim=0)
        # normalize
        gt_map *= (float(len(gt_points)) / (gt_map.sum() + 1e-8))
        rand_gt_map *= (float(len(rand_gt_points)) / (rand_gt_map.sum() + 1e-8))
        return {
            'sample_1_q':(img_q, gt_map, mask),
            'sample_1_k':(img_k, gt_map, mask),
            'sample_2_q':(rand_img_q, rand_gt_map, rand_mask),
            'sample_2_k':(rand_img_k, rand_gt_map, rand_mask),
        }

    def __len__(self):
        return len(self.images)
