import os
import sys
import cv2
import shutil
import json
import random
import math
import numpy as np
import yaml
import rosbag
from cv_bridge import CvBridge
from copy import deepcopy

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

def get_distorted_image_paths(coco_dir):
    # return a list of all distorted image paths in a given directory
    return sorted(glob(os.path.join(coco_dir, 'distorted_images/*.png')))

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
    # file encoding: label_string
    with open(fpath) as f:
        labels = {i: line.rstrip() for i, line in enumerate(f)}
    return labels

def load_custom_labels(fpath):
    # file encoding: category_id label_string
    with open(fpath) as f:
        labels = {int(k): v for line in f for (k, v) in [line.strip().split(None, 1)]}
    return labels

def load_mapping(fpath):
    # file encoding: category_id new_id label_string
    with open(fpath) as f:
        mapping = {int(k): (new_id, name) for line in f for (k, new_id, name) in [line.strip().split(None, 2)]}
    return mapping

def load_camera_intrinsics(yaml_fpath):
    with open(yaml_fpath) as f:
        camera_intrinsics = yaml.load(f, Loader=yaml.FullLoader)
        camera_matrix = np.array(camera_intrinsics['camera_matrix']['data']).reshape(3, 3)
        dist_coefficients = np.array([camera_intrinsics["distortion_coefficients"]["data"]])
        img_size = (camera_intrinsics["image_width"], camera_intrinsics["image_height"])

    return {'camera_matrix': camera_matrix, 'dist_coefficients': dist_coefficients, 'img_size': img_size}

def load_B_C(path):
    # loads rigid body camera frame (B->C) transformation matrix
    with open(path) as f:
        B_C = yaml.load(f, Loader=yaml.FullLoader)
    B_C = np.array(B_C['transformation']).reshape(4, 4)
    return B_C

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

def extract_rosbag_images(image_bag_fpath: str, image_bag_topic: str, output_path: str, undistort: bool, intrinsics: dict):
    # extract images from rosbag/topic, save to a given path and optionally undistort the image given camera intrinsics
    
    if undistort:
        mapx, mapy = \
            cv2.initUndistortRectifyMap(intrinsics['camera_matrix'], intrinsics['dist_coefficients'], None, 
                                        intrinsics['camera_matrix'], intrinsics['img_size'], 5)
    
    bag = rosbag.Bag(image_bag_fpath, 'r')
    bridge = CvBridge()
    lossless_compression_factor = 3 #9

    print('Start extracting images from ROSbag ... (%s)' % image_bag_fpath)
    messages = bag.read_messages(topics=[image_bag_topic])
    for i, (topic, msg, ts) in enumerate(messages):
        img = bridge.imgmsg_to_cv2(msg, desired_encoding='passthrough')
        
        if undistort:
            img = cv2.remap(img, mapx, mapy, cv2.INTER_LINEAR)
        
        save_path = os.path.join(output_path, '%06i.png' % (i+1))
        cv2.imwrite(save_path, cv2.cvtColor(img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, lossless_compression_factor])
    
    print('Done extracting images from ROSbag to `output_path` (%s)' % output_path)

    bag.close()

def quat2rot(q): # from [c, xs, ys, zs] to rotation matrix [[r00, r01, r02],[r10, r11, r12],[r20, r21, r22]]
    # Covert a quaternion into a full three-dimensional rotation matrix.
    
    # Extract the values from Q
    q0 = q[0]
    q1 = q[1]
    q2 = q[2]
    q3 = q[3]

    # First row of the rotation matrix
    r00 = 2 * (q0 * q0 + q1 * q1) - 1
    r01 = 2 * (q1 * q2 - q0 * q3)
    r02 = 2 * (q1 * q3 + q0 * q2)

    # Second row of the rotation matrix
    r10 = 2 * (q1 * q2 + q0 * q3)
    r11 = 2 * (q0 * q0 + q2 * q2) - 1
    r12 = 2 * (q2 * q3 - q0 * q1)

    # Third row of the rotation matrix
    r20 = 2 * (q1 * q3 - q0 * q2)
    r21 = 2 * (q2 * q3 + q0 * q1)
    r22 = 2 * (q0 * q0 + q3 * q3) - 1

    # 3x3 rotation matrix
    rot_matrix = [[r00, r01, r02],
                  [r10, r11, r12],
                  [r20, r21, r22]]

    return rot_matrix

def calc_add_rmatrix(coco_annotation_data: dict):
    # calculates the rotation matrix from `coco_annotation_data` quaterions and returns new annotation data containing both, quaternions and rotation matrix
    rmatrix_coco_annotation_data = deepcopy(coco_annotation_data)

    for i, annotation in enumerate(coco_annotation_data['annotations']):
        object_pose_rmatrix = quat2rot(annotation['object_pose']['quaternion'])
        camera_pose_rmatrix = quat2rot(annotation['camera_pose']['quaternion'])
        relative_pose_rmatrix = quat2rot(annotation['relative_pose']['quaternion'])
        
        rmatrix_annotation = rmatrix_coco_annotation_data['annotations'][i]
        try: # equal annotations check
            if rmatrix_annotation != annotation:
                raise ValueError('Annotation data values not identical (index bug). Not adding a wrong rotation matrix. Abort.')
        except Exception as e:
                print('Exception: {}'.format(str(e)), file=sys.stderr)
                sys.exit(1)

        rmatrix_annotation['object_pose']['rotation'] = object_pose_rmatrix
        rmatrix_annotation['camera_pose']['rotation'] = camera_pose_rmatrix
        rmatrix_annotation['relative_pose']['rotation'] = relative_pose_rmatrix

    return rmatrix_coco_annotation_data

def align_coco_annotation_data(coco_annotation_data: dict, mapping: dict):
    # takes current coco_annotation data and aligns category_ids accoring to `mapping` where key = old category_id; value = new category_id;
    # TODO: search for modifications.txt in root folder and find 'aligned' to abort this function.

    coco_annotation_data_aligned = deepcopy((coco_annotation_data))

    for i, annotation in enumerate(coco_annotation_data['annotations']):
        category_id = annotation['category_id']

        annotation_aligned = coco_annotation_data_aligned['annotations'][i]
        try: # equal annotations check
            if annotation_aligned != annotation:
                raise ValueError('Annotation data values not identical (index bug). Not further aligning category_ids. Abort.')
        except Exception as e:
                print('Exception: {}'.format(str(e)), file=sys.stderr)
                sys.exit(1)

        annotation_aligned['category_id'] = int(mapping[category_id][0])
    
    return coco_annotation_data_aligned

def add_coco_style_categories(coco_annotation_data: dict):
    # adds COCO data format categories section to coco_annotation_data if not present
    if 'categories' not in coco_annotation_data:

        labels = load_labels('labels/coco.names')
        category_ids = []
        categories = [] # build categories dict

        for annotation in coco_annotation_data['annotations']:
            category_id = annotation['category_id']
            label = labels[category_id]
            if category_id not in category_ids:
                categories.append({'name': label, 'id': category_id})
                category_ids.append(category_id)

        coco_annotation_data['categories'] = categories

    return coco_annotation_data


def get_image_annotation_object_center(bbox: list): # from [x2, y2, w, h] to (cx, cy)
    # returns the center point of the object from the associated bounding box (image coordinates top left 0/0)
    return (bbox[0] + bbox[2] / 2, bbox[1] + bbox[3] / 2)