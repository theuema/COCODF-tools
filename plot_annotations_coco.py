import sys
import argparse
import os
from numpy import random

from lib.image_plot_annotations import plot_image_annotations
from lib.base import (init_output_path, get_image_paths, get_img_ids_from_arguments, load_json, get_annoation_data_fpath, 
                        get_id_img_path, get_id_img_annotations, load_labels, load_custom_labels)

'''
    :Reads COCO data format dataset: from `--coco_path` with the following filestructure: 
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json
    :Plots bounding boxes from annotation/image pairs: specified by `--image_ids` and saves the results to `--output_path`
    :Plots bounding boxes from annotation/image pairs: specified by `--show_ids` and directly shows the results
        `--image_ids` and `--show_ids` correspond to the images present in images/ sorted by filename
    :Segmentation mask is visualized: by passing `--segmentation`
'''
def plot():
    coco_path, output_path, show_ids, image_ids, segmentation, annotator, labels_fpath, category_id_is_line = \
    opt.coco_path, opt.output_path, opt.show_ids, opt.image_ids, opt.segmentation, opt.annotator, opt.labels_path, opt.category_id_is_line

    try: # save and show arguments check
        if image_ids is None and show_ids is None:
            raise AttributeError('Nothing to process. Missing `--image_ids` and/or `--show-ids` argument.')
        if output_path is None and image_ids is not None:
            raise AttributeError('Missing `--output-path` argument to save `--image-ids`.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    # saving and showing conditions for results
    save = False if image_ids is None else True
    show = False if show_ids is None else True

    # get annotation labels
    labels = load_labels(labels_fpath) if category_id_is_line else load_custom_labels(labels_fpath)

    # init filepaths
    all_img_paths = get_image_paths(coco_path)
    coco_annotation_data_fpath = get_annoation_data_fpath(coco_path)
    if not len(all_img_paths) or not os.path.exists(coco_annotation_data_fpath):
        print('Error: No images or annotations to process found in coco path (%s)' % coco_path)
        return

    # init save
    if save: 
        init_output_path(output_path)
        image_ids = get_img_ids_from_arguments(image_ids, len(all_img_paths), '--image_ids')

    # init show
    if show: show_ids = get_img_ids_from_arguments(show_ids, len(all_img_paths), '--show_ids')
    
    # get annotation data
    coco_annotation_data = load_json(coco_annotation_data_fpath)

    # plot bounding boxes and/or save annotated image(s)
    if show: # plot bounding boxes and show images
        for img_id in show_ids:
            # get annotations and image path
            id_img_path = get_id_img_path(img_id, all_img_paths)
            id_img_annotations = get_id_img_annotations(img_id, coco_annotation_data) # Get all annotations (1...N objects) in image of ID

            # gen colors for #-annotations
            colors = [[random.randint(0, 255) 
                for _ in range(3)] for _ in range(len(id_img_annotations))]

            # plot annotations for image
            _ = plot_image_annotations(id_img_path, id_img_annotations, colors, labels, annotator, segmentation, show=True, img_id=img_id)

        print('Done showing annotated images.')

    if save: # plot bounding boxes and save images
        for img_id in image_ids:
            # get annotations and image path
            id_img_path = get_id_img_path(img_id, all_img_paths)
            id_img_annotations = get_id_img_annotations(img_id, coco_annotation_data) # Get all annotations (1...N objects) in image of ID

            # gen colors for #-annotations
            colors = [[random.randint(0, 255) 
                for _ in range(3)] for _ in range(len(id_img_annotations))]

            # plot annotations for image
            _ = plot_image_annotations(id_img_path, id_img_annotations, colors, labels, annotator, segmentation, out=output_path, img_id=img_id)
            
        print('Done saving annotated images (%s)' % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='uses annotation/image pairs in COCO format to plot bounding boxes and generates annotated images')
    parser.add_argument('--coco-path', type=str, required=True,
                        help='path to the COCO dataset directory containing annotations/data.json file and images/ folder')
    parser.add_argument('--output-path', type=str, help='annotated image(s) output folder')
    parser.add_argument('--show-ids', nargs='*', help='ID list of images that are displayed as an example')
    parser.add_argument('--image-ids', nargs='*', # nargs: creates a list; 0 or more values expected
                        help='''
                        ID list of images that are annotated and saved (e.g., `--image_ids 2 4 8 16 32`);
                        process a number of randomly selected images (e.g., `--image_ids random 5`);
                        if not passed as an argument all images are processed (assumes consecutive numbering starting with ID 1)
                        ''')
    parser.add_argument('--annotator', type=str,
                        help='''
                        define an additional label to shown with plotted bounding box.
                        (e.g., `--annotator manual` to label manual annotation, `--annotator OptiTrack` for automated annotation from OptiTrack)
                        ''')
    parser.add_argument('--segmentation', action='store_true', help='enable binary segmentation mask in the output')
    parser.add_argument('--labels-path', type=str, required=True, help='''
                        path to file containing `category_id labels` corresponding to annotations/data.json (e.g., labels/aau.customnames)
                        ''')
    parser.add_argument('--category-id-is-line', action='store_true', 
                        help='enable if `--labels-path` contains `labels` and the line number equals the `category_id`')
    opt = parser.parse_args()
    print(opt)

    plot()