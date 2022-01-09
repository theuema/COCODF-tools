import argparse
import os
import sys
from pathlib import Path

from lib.base import init_output_path, get_image_paths, get_distorted_image_paths, get_annoation_data_fpath, get_subdir_paths
from lib.gen_coco_copy import gen_coco_annotation_data_copy, gen_images_copy

'''
    :Reads dataset with multiple recordings: from `--recordings_path` with the following filestructure: 
    ./recordings_path
        ./recordings_path/1/output/images/
            ./recordings_path/1/output/images/*.png
        ./recordings_path/1/output/annotations/
            ./recordings_path/1/output/annotations/data.json
            
    :Saves annotation/data.json and images/ to a new location specified by `output_path`
'''
def copy():
    recordings_path, output_path, make_img_id_unique = \
    opt.recordings_path, opt.output_path, opt.unique_img_id
    # TODO: function not tested! 
    
    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    # copy recording wise
    for recording_path in recording_paths: 
        # init filepaths for current recording
        coco_path = os.path.join(recording_path, 'output')
        all_img_paths = get_image_paths(coco_path)
        coco_annotation_data_fpath = get_annoation_data_fpath(coco_path)
        if not len(all_img_paths) or not os.path.isfile(coco_annotation_data_fpath):
            print('Error: No images or annotations to copy found in coco path (%s)' % coco_path)
            return
        
        c_annotation_data_path = str(Path(output_path) / Path(recording_path).name / 'output' / 'annotations')
        init_output_path(c_annotation_data_path)
        c_images_output_path = str(Path(output_path) / Path(recording_path).name / 'output' / 'images')
        init_output_path(c_images_output_path)
        
        # copy data
        gen_coco_annotation_data_copy(coco_annotation_data_fpath, output_path=c_annotation_data_path, make_img_id_unique=make_img_id_unique)
        gen_images_copy(all_img_paths, output_path=c_images_output_path)

        # also copy distorted image data if exists
        if os.path.exists(str(Path(coco_path) / 'distorted_images')):
            all_img_paths = get_distorted_image_paths(coco_path)
            c_images_output_path = str(Path(output_path) / Path(recording_path).name / 'output' / 'distorted_images')
            init_output_path(c_images_output_path)
            gen_images_copy(all_img_paths, output_path=c_images_output_path)
    
    print('Done copying COCO data format data recordings (%s)' % (output_path + '/N' + '/output'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='copies all images of multiple recordings in COCO data format dataset and stores the copied dataset to a folder specified by `--output-path`')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/ ... OptiTrack_recordings/N/output/ 
                        containing COCO data format annotation/image pairs (annotations/data.json and /images folder)
                        ''')
    parser.add_argument('--output-path', type=str, required=True,
                        help='copied dataset output folder')
    parser.add_argument('--unique-img-id', action='store_true', help='make sure that copied data `images` dict has unique `image_id` (e.g., no same images in `images` dict of data.json')
    opt = parser.parse_args()
    print(opt)

    copy()