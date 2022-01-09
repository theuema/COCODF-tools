import argparse
import os
from pathlib import Path

from lib.base import init_output_path, get_image_paths, get_distorted_image_paths, get_annoation_data_fpath
from lib.gen_coco_copy import gen_coco_annotation_data_copy, gen_images_copy 

'''
    :Reads COCO data format dataset: from `--coco_path` with the following filestructure: 
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json
        
    :Saves annotation/data.json and images/ to a new location specified by `output_path`
'''
def copy():
    coco_path, output_path, make_img_id_unique = \
    opt.coco_path, opt.output_path, opt.unique_img_id
    # TODO: function not tested! 
    
    # init filepaths
    all_img_paths = get_image_paths(coco_path)
    coco_annotation_data_fpath = get_annoation_data_fpath(coco_path)
    if not len(all_img_paths) or not os.path.isfile(coco_annotation_data_fpath):
        print('Error: No images or annotations to copy found in coco path (%s)' % coco_path)
        return

    c_annotation_data_path = os.path.join(output_path, 'annotations')
    init_output_path(c_annotation_data_path)
    c_images_output_path = os.path.join(output_path, 'images')
    init_output_path(c_images_output_path)
    
    # copy data
    gen_coco_annotation_data_copy(coco_annotation_data_fpath, output_path=c_annotation_data_path, make_img_id_unique=make_img_id_unique)
    gen_images_copy(all_img_paths, output_path=c_images_output_path)

    # also copy distorted image data if exists
    if os.path.exists(str(Path(coco_path) / 'distorted_images')):
        all_img_paths = get_distorted_image_paths(coco_path)
        c_images_output_path = str(Path(output_path) / 'distorted_images')
        init_output_path(c_images_output_path)
        gen_images_copy(all_img_paths, output_path=c_images_output_path)

    print('Done copying COCO data format data (%s)' % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copies all images in COCO data format dataset and stores a copied dataset to a folder specified by `--output-path`')
    parser.add_argument('--coco-path', type=str, required=True,
                        help='path to the COCO dataset directory containing annotations/data.json file, images/ and distorted_images/ (optional) folder')
    parser.add_argument('--output-path', type=str, required=True,
                        help='copied dataset output folder')
    parser.add_argument('--unique-img-id', action='store_true', help='make sure that copied data `images` dict has unique `image_id` (e.g., no same images in `images` dict of data.json')
    opt = parser.parse_args()
    print(opt)

    copy()