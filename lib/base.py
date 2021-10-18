import os
import sys
import cv2
import shutil
import json
import random
import math
import numpy as np

from glob import glob

def plot_one_bbox(xyxy, img, label_inside_pos: bool=False, color=None, label=None, line_thickness=None, annotator=None):
    # plots one bbox on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(xyxy[0]), int(xyxy[1])), (int(xyxy[2]), int(xyxy[3])) # corner points top-left, bottom-right
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label: # place label
        tf = max(tl - 1, 1)  # font thickness
        fs = tl / 2.5 # font scale
        t_size_wh = cv2.getTextSize(label, 0, fontScale=fs, thickness=tf)[0]

        pos = (c1[0], c1[1] + t_size_wh[1] + 2) if label_inside_pos else (c1[0], c1[1] - 2) # label position
        c_label = (c1[0] + t_size_wh[0], c1[1] + t_size_wh[1] + 3) if label_inside_pos else (c1[0] + t_size_wh[0], c1[1] - t_size_wh[1] - 3)
        cv2.rectangle(img, c1, c_label, color, thickness=-1, lineType=cv2.LINE_AA)  # label rectangle thickness=-1 = filled
        cv2.putText(img, label, pos, fontFace=0, fontScale=fs, color=[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA) # place label text in label rectangle
    if annotator: # place annotator
        tf = max(tl - 1, 1)  # font thickness
        fs = tl / 2.5 # font scale
        t_size_wh = cv2.getTextSize(annotator, 0, fontScale=fs, thickness=tf)[0]
        
        c3 = (int(xyxy[0]), int(xyxy[3])) # corner point bottom-left
        pos = (c3[0], c3[1] - 2) if label_inside_pos else (c3[0], c3[1] + t_size_wh[1] + 2) # annotator position
        c_annotator = (c3[0] + t_size_wh[0], c3[1] - t_size_wh[1] - 3) if label_inside_pos else (c3[0] + t_size_wh[0], c3[1] + t_size_wh[1] + 3)
        cv2.rectangle(img, c3, c_annotator, color, thickness=-1, lineType=cv2.LINE_AA)  # label rectangle thickness=-1 = filled
        cv2.putText(img, annotator, pos, fontFace=0, fontScale=fs, color=[225, 255, 255], thickness=tf, lineType=cv2.LINE_AA) # place label text in label rectangle

def xybwh2coco(bbox): # from [x2, y2, w, h] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right, w=width, h=hight
    x1 = int(bbox[0] - bbox[2]) # x-top-left
    y1 = int(bbox[1] - bbox[3]) # y-top-left
    w = int(bbox[2]) # width
    h = int(bbox[3]) # height
    return [x1, y1, w, h]

def xywh2xyxy(bbox): # from [x, y, w, h] to [x1, y1, x2, y2] where xy = center, xy1=top-left, xy2=bottom-right, w=width, h=hight
    x1 = int(bbox[0] - bbox[2] / 2) # x-top-left
    y1 = int(bbox[1] - bbox[3] / 2) # y-top-left
    x2 = int(bbox[0] + bbox[2] / 2) # x-bottom-right
    y2 = int(bbox[1] + bbox[3] / 2) # y-bottom-right
    return [x1, y1, x2, y2]

def coco2xyxy(bbox): # from [x1, y1, w, h] to [x1, y1, x2, y2] where xy1=top-left, xy2=bottom-right, w=width, h=hight
    x1 = int(bbox[0]) # x-top-left
    y1 = int(bbox[1]) # y-top-left
    x2 = int(bbox[0] + bbox[2]) # x-bottom-right
    y2 = int(bbox[1] + bbox[3]) # y-bottom-right
    return [x1, y1, x2, y2]

def xyxy2coco(bbox): # from [x1, y1, x2, y2] to [x1, y1, w, h] where xy1=top-left, xy2=bottom-right, w=width, h=hight
    x1 = int(bbox[0]) # x-top-left
    y1 = int(bbox[1]) # y-top-left
    w = int(bbox[2] - bbox[0]) # w
    h = int(bbox[3] - bbox[1]) # h
    return [x1, y1, w, h]

def get_projected_point(o: tuple, xy: tuple, degree: float):
    # projects point (`xy`) to new 2d plane after rotation defined by `degree` around point (`o`)rigin
    radian = math.radians(degree)
    s = math.sin(radian)
    c = math.cos(radian)
    # translate point back to origin
    xyt = (xy[0] - o[0], xy[1] - o[1])
    # rotate point
    xyr = (xyt[0] * c - xyt[1] * s, xyt[0] * s + xyt[1] * c)
    # translate point back
    return (xyr[0] + o[0], xyr[1] + o[1])

def get_xyxy_bbox_image_center_rotation(bbox, image_w, image_h, degree: float):
    # projects the old x-top left and y-top left to the new 2d image plane after rotation around the image center
    # TODO: atm only a rotation for 180 degrees work properly; add calculation for new bbox width and height depending on the degree
    xyr = get_projected_point(o=(image_w / 2, image_h / 2), xy=(bbox[0], bbox[1]), degree=degree)
    bbox[0] = round(xyr[0]) # round to nearest integer
    bbox[1] = round(xyr[1])
    return xybwh2coco(bbox)

def sort_dist_origin(bboxes: list):
    # returns sorted list of bboxes [x1, y1, _, _] sorted by the euclidean distance from xy1-top-left to origin (0,0)
    return sorted(bboxes, key=lambda xy: xy[0] ** 2 + xy[1] ** 2)

def rle2mask(rle, image_w, image_h):
    # run length encoding to display mask
    rows, cols = image_h, image_w
    img = np.zeros(rows * cols, dtype=np.uint8)
    current_index = 0
    for i, val in enumerate(rle):
        if i % 2 == 0:
            img[current_index : current_index + val] = 0
        else:
            img[current_index : current_index + val] = 255
        current_index += val
    img = img.reshape(rows, cols)
    img = np.array([img, img, img])
    img = np.moveaxis(img, [0], [2])
    return img

def init_output_path(output_path):
    # initialize output_path and clean existing output folder
    if os.path.exists(output_path):
        shutil.rmtree(output_path)
    os.makedirs(output_path)

def get_image_paths(coco_dir):
    # return a list of all image paths in a given directory
    return sorted(glob(os.path.join(coco_dir, 'images/*.png')))

def get_subdir_paths(dir):
    # return a list of all subdir paths of a given directory
    return sorted(glob(os.path.join(dir, '*/')))

def get_annoation_data_fpath(coco_dir):
    # return file path to coco_annotation_data (annotations/data.json)
    return os.path.join(coco_dir, 'annotations/data.json')

def get_yolo_data_fpath(coco_dir):
    # return file path to coco_annotation_data (annotations/data.json)
    return os.path.join(coco_dir, 'detected_images/data.json')

def check_img_id_type_value(img_id, num_images: int, arg_name: str):
    try: 
        # type cast
        img_id = int(img_id)
        # value check
        if img_id < 1 or img_id > num_images: # img_id must be greater than zero and and smaller or equal to population size
            raise ValueError('Image id is greater than the actual population of images or smaller than 1', img_id)
    except Exception as e:
            print('Argument {} caused an exception: {}'.format(arg_name, str(e)), file=sys.stderr)
            sys.exit(1)
    return img_id

def get_img_ids_from_arguments(opt_image_ids, num_images: int, arg_name: str):
    if opt_image_ids is None:                                  # all images processed e.g., no --image_ids argument passed == None
        print('No --image_ids passed, processing all images.')
        return np.arange(1, num_images+1) # Generate compatible ids from 1 to num_images (1...num_images)
    elif opt_image_ids[0] == 'random':                           # random images processed e.g., --image_ids random 5 
        num_random_img = check_img_id_type_value(opt_image_ids[1], num_images, arg_name)
        return np.random.permutation(
            np.arange(1, num_images+1))[:num_random_img] # Generate uniform random sample from 1 to num_images (1...num_images) of size num_random_img
    else:                                                       # only specific images processed e.g., --image_ids 2 4 8 16 32
        return [check_img_id_type_value(id, num_images, arg_name) for id in opt_image_ids]

def load_json(fpath):
    # load JSON document from binary filepath
    with open(fpath) as f:
        data = json.load(f)
    return data

def load_labels(fpath):
    with open(fpath) as f:
        labels = {i: line.rstrip() for i, line in enumerate(f)}
    return labels

def load_custom_labels(fpath):
    with open(fpath) as f:
        labels = {int(k): v for line in f for (k, v) in [line.strip().split(None, 1)]}
    return labels

def get_id_img_path(img_id: int, all_img_paths: list):
    # get image path from image ID (img_id must be checked previously for out of bounds)
    # TODO: works only if 'image_id' corresponds to a sorted list of image paths (see: get_image_paths(dir))
    # add compatibility for unsorted filenames
    return all_img_paths[img_id - 1] # Index correction: image IDs start with 1 that corresponds to index 0

def get_id_img_paths(img_ids: list, all_img_paths: list):
    # get image paths from image IDs (img_ids must be checked previously for out of bounds)
    return [get_id_img_path(id, all_img_paths) for id in img_ids]

def get_id_img_annotations(img_id: int, coco_annotation_data):
    # get all annotations (several or 1 object(s) in an image) for image ID
    return [annotation for annotation in coco_annotation_data['annotations'] 
            if annotation['image_id'] == img_id]