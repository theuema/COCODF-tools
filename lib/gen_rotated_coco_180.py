from pathlib import Path
import copy
import os
import cv2
import json
import sys

from lib.base import get_xyxy_bbox_image_center_rotation, load_json

def gen_180_rotated_coco_annotation_data(coco_annotation_data_fpath, output_path, make_img_id_unique: bool=False):
    # rotates all annotations and segmentation from a COCO data format annotations/data.json for 180 degrees
    # stores rotated annotations to `output_path/annotations/data.json`
    # :Param make_img_id_unique: if `images` in annotation data repeats: transform to COCO data format with unique image_ids
    if make_img_id_unique: 
        image_ids = []
        images = []
    
    # get annotation data
    coco_annotation_data = load_json(coco_annotation_data_fpath)
    coco_rotated_annotation_data = copy.deepcopy(coco_annotation_data)

    # rotate all bboxes and segmentation rle    
    for annotation in coco_rotated_annotation_data['annotations']:
        # get image for annotation
        image = next((image for image in coco_rotated_annotation_data['images'] if image["id"] == annotation['image_id']), None)
        if image is None:
            print('Error: image with id {} not found in {}'.format(annotation['image_id'], coco_annotation_data_fpath), file=sys.stderr)
            print('Abort rotation.')
            sys.exit(1)
        
        #rotate annotation
        annotation['bbox'] = get_xyxy_bbox_image_center_rotation(annotation['bbox'], 
                                        image['width'], image['height'], degree=180)
        
        # rotate segmentation
        annotation['segmentation']['counts'] = annotation['segmentation']['counts'][::-1]

        if make_img_id_unique:
            if annotation['image_id'] not in image_ids:
                images.append(image)
                image_ids.append(annotation['image_id'])
    
    if make_img_id_unique:
        coco_rotated_annotation_data['images'] = images

    # save rotated annotation data
    f = os.path.join(output_path, 'data.json')
    with open(f, 'w') as outfile:
        json.dump(coco_rotated_annotation_data, outfile)
    
    print('Rotated annotation data saved. (%s)' % f)

def gen_180_rotated_images(all_img_paths: list, output_path):
    # rotate and store all images to `output_path/images`
    for img_path in all_img_paths:
        # rotate
        img = cv2.imread(img_path)
        id_rotated_img = copy.deepcopy(img)
        cv2.rotate(img, cv2.ROTATE_180, id_rotated_img)

        # save
        save_path = str(Path(output_path) / Path(img_path).name) 
        cv2.imwrite(save_path, id_rotated_img)
    print('Rotated images saved. (%s)' % output_path)