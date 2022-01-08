import argparse
import os
from pathlib import Path

from lib.base import init_output_path, get_image_paths, get_distorted_image_paths, get_annoation_data_fpath
from lib.gen_rotated_coco_180 import gen_180_rotated_coco_annotation_data, gen_180_rotated_images

'''
    :Reads COCO data format dataset: from `--coco_path` with the following filestructure: 
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json
    :Writes a annotation/data.json with 180° rotated bounding boxes; Rotates all images in `--coco_path` and its subfolders;
    :Stores the new rotated dataset to `output_path`
'''
def rotate():
    coco_path, output_path, make_img_id_unique = \
    opt.coco_path, opt.output_path, opt.unique_img_id
    
    # init filepaths
    all_img_paths = get_image_paths(coco_path)
    coco_annotation_data_fpath = get_annoation_data_fpath(coco_path)
    if not len(all_img_paths) or not os.path.isfile(coco_annotation_data_fpath):
        print('Error: No images or annotations to rotate found in coco path (%s)' % coco_path)
        return

    r_annotation_data_path = os.path.join(output_path, 'annotations')
    init_output_path(r_annotation_data_path)
    r_images_output_path = os.path.join(output_path, 'images')
    init_output_path(r_images_output_path)
    
    # rotate data
    gen_180_rotated_coco_annotation_data(coco_annotation_data_fpath, output_path=r_annotation_data_path, make_img_id_unique=make_img_id_unique)
    gen_180_rotated_images(all_img_paths, output_path=r_images_output_path)

    # also rotate distorted image data if exists
    if os.path.exists(str(Path(coco_path) / 'distorted_images')):
        all_img_paths = get_distorted_image_paths(coco_path)
        r_images_output_path = str(Path(output_path) / 'distorted_images')
        init_output_path(r_images_output_path)
        gen_180_rotated_images(all_img_paths, output_path=r_images_output_path)

    print('Done rotating COCO data format data (%s)' % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rotates all images for 180 degrees and projects annotation data points of annotation/image pairs in COCO data format to fit the rotated image dataset')
    parser.add_argument('--coco-path', type=str, required=True,
                        help='path to the COCO dataset directory containing annotations/data.json file, images/ and distorted_images/ (optional) folder')
    parser.add_argument('--output-path', type=str, required=True,
                        help='rotated dataset output folder (creates annotations/ and images/ folders)')
    parser.add_argument('--unique-img-id', action='store_true', help='make sure that rotated data `images` have unique `image_id` (e.g., each image is stored to `images` in `data.json` only once!)')
    opt = parser.parse_args()
    print(opt)

    rotate()