import argparse
import os
import sys
from pathlib import Path

from lib.base import init_output_path, get_image_paths, get_distorted_image_paths, get_annoation_data_fpath, get_subdir_paths
from lib.gen_rotated_coco_180 import gen_180_rotated_coco_annotation_data, gen_180_rotated_images

'''
    :Reads dataset with multiple recordings: from `--recordings_path` with the following filestructure: 
    ./recordings_path
        ./recordings_path/1/output/images/
            ./recordings_path/1/output/images/*.png
        ./recordings_path/1/output/annotations/
            ./recordings_path/1/output/annotations/data.json
            
    :Writes a annotation/data.json with 180° rotated bounding boxes; Rotates all images in `./recordings_path/1/output/` and its subfolders;
    :Stores the new rotated datasets of each recording to `output_path`
'''
def rotate():
    recordings_path, output_path, make_img_id_unique = \
    opt.recordings_path, opt.output_path, opt.unique_img_id
    
    
    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    # rotate recording wise
    for recording_path in recording_paths: 
        # init filepaths for current recording
        coco_path = os.path.join(recording_path, 'output')
        all_img_paths = get_image_paths(coco_path)
        coco_annotation_data_fpath = get_annoation_data_fpath(coco_path)
        if not len(all_img_paths) or not os.path.isfile(coco_annotation_data_fpath):
            print('Error: No images or annotations to rotate found in coco path (%s)' % coco_path)
            return
        
        r_annotation_data_path = str(Path(output_path) / Path(recording_path).name / 'output' / 'annotations')
        init_output_path(r_annotation_data_path)
        r_images_output_path = str(Path(output_path) / Path(recording_path).name / 'output' / 'images')
        init_output_path(r_images_output_path)
        
        # rotate 2D data
        gen_180_rotated_coco_annotation_data(coco_annotation_data_fpath, output_path=r_annotation_data_path, make_img_id_unique=make_img_id_unique)
        gen_180_rotated_images(all_img_paths, output_path=r_images_output_path)

        # also rotate distorted image data if exists
        if os.path.exists(str(Path(coco_path) / 'distorted_images')):
            all_img_paths = get_distorted_image_paths(coco_path)
            r_images_output_path = str(Path(output_path) / Path(recording_path).name / 'output' / 'distorted_images')
            init_output_path(r_images_output_path)
            gen_180_rotated_images(all_img_paths, output_path=r_images_output_path)
    
    print('Done rotating COCO data format data recordings (%s)' % (output_path + '/N' + '/output'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='rotates all images of multiple recordings in COCO data format for 180 degrees and projects annotation data points of annotation/image pairs in COCO data format to fit the rotated image dataset')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/ ... OptiTrack_recordings/N/output/ 
                        containing COCO data format annotation/image pairs (annotations/data.json and images/ folder)
                        ''')
    parser.add_argument('--output-path', type=str, required=True,
                        help='rotated dataset output folder')
    parser.add_argument('--unique-img-id', action='store_true', help='make sure that rotated data `images` have unique `image_id` (e.g., each image is stored to `images` in `data.json` only once!)')
    opt = parser.parse_args()
    print(opt)

    rotate()