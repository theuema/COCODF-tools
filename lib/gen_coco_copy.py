from pathlib import Path
import copy
import os
import cv2
import json
import sys

from lib.base import load_json

def gen_coco_annotation_data_copy(coco_annotation_data_fpath, output_path, make_img_id_unique: bool=False):
    # rotates all annotations and segmentation from a COCO data format annotations/data.json for 180 degrees
    # stores rotated annotations to `output_path/annotations/data.json`
    # :Param make_img_id_unique: if `images` in annotation data repeats: transform to COCO data format with unique image_ids
    # TODO: function not tested!

    # get annotation data
    coco_annotation_data = load_json(coco_annotation_data_fpath)
    coco_annotation_data_copy= copy.deepcopy(coco_annotation_data)
        
    if make_img_id_unique: 
        image_ids = []
        images = []
    
        # store unique images dictionary 
        for annotation in coco_annotation_data_copy['annotations']:
            # get image for annotation
            image = next((image for image in coco_annotation_data_copy['images'] if image["id"] == annotation['image_id']), None)
            if image is None:
                print('Error: image with id {} not found in {}'.format(annotation['image_id'], coco_annotation_data_fpath), file=sys.stderr)
                print('Abort rotation.')
                sys.exit(1)
            
            if annotation['image_id'] not in image_ids:
                images.append(image)
                image_ids.append(annotation['image_id'])
    
        coco_annotation_data_copy['images'] = images

    # save copy of annotation data
    f = os.path.join(output_path, 'data.json')
    with open(f, 'w') as outfile:
        json.dump(coco_annotation_data_copy, outfile)
    
    print('Dataset annotation data copied. (%s)' % f)

def gen_images_copy(all_img_paths: list, output_path):
    # rotate and store all images to `output_path/images`
    # TODO: function not tested!

    for img_path in all_img_paths:
        # copy 
        id_img = cv2.imread(img_path)
        id_copied_img = copy.deepcopy(id_img)

        # save
        save_path = str(Path(output_path) / Path(img_path).name) 
        cv2.imwrite(save_path, id_copied_img)
    print('Dataset images copied. (%s)' % output_path)