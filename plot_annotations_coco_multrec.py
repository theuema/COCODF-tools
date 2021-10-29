import sys
import argparse
import os
from numpy import random

from lib.image_plot_annotations import image_plot_annotations
from lib.base import (init_output_path, get_image_paths, get_img_ids_from_arguments, load_json, get_annoation_data_fpath, 
                        get_id_img_path, get_id_img_annotations, load_labels, load_custom_labels, get_subdir_paths)

'''
    :Reads dataset with multiple COCO data format recordings: from `--recordings_path` with the following filestructure: 
    ./recordings_path
        ./recordings_path/1/coco_output/images/
            ./recordings_path/1/coco_output/images/*.png
        ./recordings_path/1/coco_output/annotations/
            ./recordings_path/1/coco_output/annotations/data.json
            
    :Plots bounding boxes from annotation/image pairs: 
        specified by `--image_ids` and saves the results to 
        a new folder `annotated_images/` in the corresponding ./recordings_path/N/output/ folder
    :Segmentation mask is visualized: by passing `--segmentation`
'''
def plot():
    recordings_path, image_ids, segmentation, annotator, labels_fpath, category_id_is_line = \
    opt.recordings_path, opt.image_ids, opt.segmentation, opt.annotator, opt.labels_path, opt.category_id_is_line

    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    # get annotation labels
    labels = load_labels(labels_fpath) if category_id_is_line else load_custom_labels(labels_fpath)

    # plot recording wise
    for rec_id, recording_path in enumerate(recording_paths):   
        # init filepaths for current recording
        coco_path = os.path.join(recording_path, 'output')
        all_img_paths = get_image_paths(coco_path)
        coco_annotation_data_fpath = get_annoation_data_fpath(coco_path)
        if not len(all_img_paths) or not os.path.isfile(coco_annotation_data_fpath):
            print('Error: No images or annotations to process found in coco path (%s)' % coco_path)
            return

        a_images_output_path = os.path.join(coco_path, 'annotated_images')
        init_output_path(a_images_output_path)

        # get annotation data and image_ids
        coco_annotation_data = load_json(coco_annotation_data_fpath)
        image_ids = get_img_ids_from_arguments(image_ids, len(all_img_paths), '--image_ids')

        # plot annotations to all images of recording `rec_id`
        for img_id in image_ids:
            # get annotations and image path
            id_img_path = get_id_img_path(img_id, all_img_paths)
            id_img_annotations = get_id_img_annotations(img_id, coco_annotation_data) # Get all annotations (1...N objects) in image of ID

            # gen colors for #-annotations
            colors = [[random.randint(0, 255) 
                for _ in range(3)] for _ in range(len(id_img_annotations))]

            # plot all annotations for image
            _ = image_plot_annotations(id_img_path, id_img_annotations, colors, labels, annotator, segmentation, out=a_images_output_path, img_id=img_id)
            
            print('Annotated images saved (recording: %s)' % rec_id)

    print('Done annotating COCO data format data recordings (%s)' % a_images_output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='uses annotation/image pairs in COCO format to plot bounding boxes and generates annotated images')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/ ... OptiTrack_recordings/N/output/ 
                        containing COCO data format annotation/image pairs (annotations/data.json and images/ folder)
                        ''')
    parser.add_argument('--image-ids', nargs='*', required=True, # nargs: creates a list; 0 or more values expected
                        help='''
                        ID list of images that are annotated and saved - same for every recording (e.g., `--image_ids 2 4 8 16 32`);
                        process a number of randomly selected images - different for every recording (e.g., `--image_ids random 5`);
                        if not passed as an argument all images are processed (assumes consecutive numbering starting with ID 1)
                        ''')
    parser.add_argument('--annotator', type=str,
                        help='''
                        define an additional label to shown with plotted bounding box.
                        (e.g., `--annotator manual` to label manual annotation, `--annotator OptiTrack` for automated annotation from OptiTrack)
                        ''')
    parser.add_argument('--segmentation', action='store_true',
                        help='enable binary segmentation mask in the output')
    parser.add_argument('--labels-path', type=str, required=True, help='''
                        path to file containing `category_id labels` corresponding to annotations/data.json (e.g., labels/aau.customnames)
                        ''')
    parser.add_argument('--category-id-is-line', action='store_true', 
                        help='enable if `--labels-path` contains `labels` and the line number equals the `category_id`')
    opt = parser.parse_args()
    print(opt)

    plot()