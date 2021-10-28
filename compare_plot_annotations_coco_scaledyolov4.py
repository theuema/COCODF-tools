import sys
import argparse
import os
import time
import torch
import json
import cv2
import copy
from numpy import random
from pathlib import Path
from glob import glob

from ScaledYOLOv4.utils.general import strip_optimizer
from ScaledYOLOv4.utils.torch_utils import select_device
from ScaledYOLOv4.models.experimental import attempt_load
from ScaledYOLOv4.utils.general import check_img_size

from lib.image_plot_annotations import plot_image_annotations
from lib.scaledyolov4_detect import scaledyolov4_detect
from lib.base import (coco2xyxy, init_output_path, get_image_paths, get_img_ids_from_arguments, load_json, get_annoation_data_fpath, 
                        get_yolo_data_fpath, get_id_img_path, get_id_img_annotations, load_labels, plot_one_bbox, 
                        rle2mask, get_id_img_paths, load_custom_labels)
from lib.handle_detected_annotation_data import add_detected_annotations, add_detected_categories, add_image_info

'''
    :Reads COCO data format dataset: from `--coco_path` with the following filestructure: 
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json

    :Compares bounding boxes of (manual, optitrack or ...) annotations and ScaledYOLOv4 annotations: and saves the resulting images to `--output_path`
    :More detailed textual comparison results can be stored: using `--save-results`
    :Segmentation mask is visualized: by passing `--segmentation`
    :YOLO implementation taken from: https://github.com/WongKinYiu/ScaledYOLOv4
'''
def plot_compare():
    coco_path, output_path, segmentation, annotator, labels_fpath, weights, save_txt, img_size, image_ids, category_id_is_line, save_results = \
        opt.coco_path, opt.output_path, opt.segmentation, opt.annotator, opt.annotator_labels_path, opt.weights, opt.save_txt, opt.img_size, \
            opt.image_ids, opt.category_id_is_line, opt.save_results

    # init detect condition
    detect_images = not os.path.exists(get_yolo_data_fpath(coco_path)) # detect and save if no annotation data from a previous detection exists

    try: # detect status check
        if image_ids is None and detect_images:
            raise AttributeError('No detected images found. Missing `--image_ids` argument to perform object detection.')
        if weights is None and detect_images:
            raise AttributeError('No detected_images/data.json file found. Missing `--weights` argument to perform object detection.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    # init filepaths / dataset related
    all_img_paths = get_image_paths(coco_path)
    coco_annotation_data_fpath = get_annoation_data_fpath(coco_path)
    if not len(all_img_paths) or not os.path.exists(coco_annotation_data_fpath):
        print('Error: No images or annotations to process found in coco path (%s)' % coco_path)
        return
    coco_annotation_data = load_json(coco_annotation_data_fpath)
    annotated_images_output_path = os.path.join(coco_path, 'annotated_images')
    
    # perform object detection if needed
    if detect_images:
        image_ids = get_img_ids_from_arguments(image_ids, len(all_img_paths), '--image_ids')
        detected_images_output_path = os.path.join(coco_path, 'detected_images')
        init_output_path(detected_images_output_path)

        # --- ScaledYolov4
        # initialize
        device = select_device(opt.device)
        half = device.type != 'cpu'  # half precision only supported on CUDA
        
        # load model
        model = attempt_load(weights, map_location=device)  # load FP32 model
        img_size = check_img_size(img_size, s=model.stride.max())  # img_size check
        if half:
            model.half()  # to FP16

        # get names and colors of model
        names = model.module.names if hasattr(model, 'module') else model.names
        colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

        t0 = time.time() # init time
        img = torch.zeros((1, 3, img_size, img_size), device=device) # Init img
        _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

        # init to save yolo annotations
        coco_annotation_data_fpath = get_annoation_data_fpath(coco_path) # get annotation data to gather images-info for detection *.json file
        if not len(all_img_paths) or not os.path.exists(coco_annotation_data_fpath):
            print('Error: No images or annotations to process found in coco path (%s)' % coco_path)
            return
        coco_annotation_data = load_json(coco_annotation_data_fpath)
        annotations = []
        categories = []
        images = [] # image-info corresponding to annotation COCO data format
        annotation_id = 0

        # inference for 1 image at a time
        for img_id in image_ids:
            id_img_path = get_id_img_path(img_id, all_img_paths)
            id_img_annotations, id_img_categories = scaledyolov4_detect(id_img_path, img, model, names, colors, device, half, save_txt, opt, 
                                                        output_path=detected_images_output_path, ret_data=True)
            
            # add detected annotations to annotations
            add_detected_annotations(id_img_annotations, annotations, annotation_id=annotation_id, img_id=img_id)
            annotation_id += len(id_img_annotations)

            # add detected category to categories
            add_detected_categories(id_img_categories, categories)

            # add image info to images
            add_image_info(coco_annotation_data, coco_annotation_data_fpath, images, img_id)

        # store COCO data format yolo annotations
        json_path = str(Path(detected_images_output_path) / 'data.json')
        data = {'images': images, 'annotations': annotations, 'categories': categories}
        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)
        print('Yolo annotation file saved (%s)' % json_path)

        print('Inference done for all images (%s)' % detected_images_output_path)
        print('Object detection done. (%.3fs)' % (time.time() - t0))

    # init filepaths / yolo related
    yolo_annotation_data_fpath = get_yolo_data_fpath(coco_path)
    yolo_annotation_data = load_json(yolo_annotation_data_fpath)
    yolo_labels = load_labels('labels/coco.names')
    yolo_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(yolo_labels))]

    # get yolo detected image_ids
    image_ids = [image['id'] for image in yolo_annotation_data['images']]

    # detected image_ids images are a subset of annotated images, otherwise plot and annotate images for image_ids check
    annotated_img_path_names = [Path(path).name for path in sorted(glob(os.path.join(annotated_images_output_path, '*.png')))]
    image_ids_path_names = [Path(path).name for path in get_id_img_paths(image_ids, all_img_paths)]
    plot_annotation = not set(image_ids_path_names).issubset(annotated_img_path_names)

    # plot annotator annotations to images and save image if needed
    if plot_annotation:
        try: # labels path check
            if labels_fpath is None:
                raise AttributeError('Data needs to be annotated before comparison. Therefore annotator labels are required. Please provide `--labels_path` and `category-id-is-line` (if necessary) and restart the script.')
        except Exception as e:
                print('Exception: {}'.format(str(e)), file=sys.stderr)
                sys.exit(1)

        # get labels & colors
        annotator_labels = load_labels(labels_fpath) if category_id_is_line else load_custom_labels(labels_fpath)
        annotator_colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(annotator_labels))]

        init_output_path(annotated_images_output_path)
        for img_id in image_ids:
            # get annotations and image path
            id_img_path = get_id_img_path(img_id, all_img_paths)
            id_img_annotations = get_id_img_annotations(img_id, coco_annotation_data) # Get all annotations (1...N objects) in image of ID
            
            # plot annotations for image
            _ = plot_image_annotations(id_img_path, id_img_annotations, annotator_colors, annotator_labels, annotator, segmentation, 
                                        out=annotated_images_output_path, img_id=img_id)

    # init output path before writing results
    init_output_path(output_path)
    
    # conduct comparison of annotations and write plot of yolo annotations to annotator annotated images
    for img_id in image_ids:
        # get image path of annotated image for comparison (instead of using un-annotated image)
        id_img_path = get_id_img_path(img_id, all_img_paths)
        annotated_id_img_path = str(Path(annotated_images_output_path) / Path(id_img_path).name)

        # init img save path
        save_path = str(Path(output_path) / Path(annotated_id_img_path).name)

        # get annotator and yolo annotations for image of img_id
        id_img_annotator_annotations = get_id_img_annotations(img_id, coco_annotation_data) # Get all annotator annotations (1...N objects) in image of ID
        id_img_yolo_annotations = get_id_img_annotations(img_id, yolo_annotation_data) # Get all detected annotations (1...N objects) in image of ID
        
        #TODO: conduct comparison!
        # compare annotations, calculate and plot IoU and calculate euclidean distance between bbox centers
        # match the detected annotation to annotator annotation in original data.json
        # caluclate 3D position

        # print information
        print('Comparing image ID {} with {} annotations and {} detections ({})'.format(img_id, len(id_img_annotator_annotations), 
                                                                                            len(id_img_yolo_annotations), id_img_path))

        # write results
        img = cv2.imread(annotated_id_img_path)
        id_comparison_img = copy.deepcopy(img)
        
        for id_img_yolo_annotation in id_img_yolo_annotations:
            bbox = id_img_yolo_annotation['bbox']
            xyxy = coco2xyxy(bbox)

            # get category info
            category = next((category for category in yolo_annotation_data['categories'] if category['id'] == id_img_yolo_annotation['category_id']), None)
            if category is None:
                print('Error: category with id {} not found in {}'.format(bbox['category_id'], yolo_annotation_data_fpath), file=sys.stderr)
                print('Abort detection.')
                sys.exit(1)

            plot_one_bbox(xyxy, id_comparison_img, label_inside_pos=True, color=yolo_colors[int(category['id'])], 
                            label=category['name'], line_thickness=1, annotator='yolo')

        # insert segmentation mask
        if segmentation:
            print('Inserting segmentation mask ...')
            id_comparison_segm_img = id_comparison_img
            for annotation in annotations:
                binary_mask_img = rle2mask(annotation['segmentation']['counts'], 
                                                annotation['segmentation']['size'][1], annotation['segmentation']['size'][0])
                id_comparison_segm_img = cv2.addWeighted(id_comparison_segm_img, 1.0, binary_mask_img, 1, 0)
                # write compared image with segmentation mask
                cv2.imwrite(save_path, id_comparison_segm_img)
            continue
        
        # write compared image
        cv2.imwrite(save_path, id_comparison_img)
    
    if save_results:
        json_path = str(Path(output_path) / 'comparison.json')
        data = {'info': '#TODO: Store comparison data!'}
        with open(json_path, 'w') as outfile:
            json.dump(data, outfile)
        print('Comparison results file saved (%s)' % json_path)

    print('Done saving compared images (%s)' % save_path)
    return


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='plots both bounding boxes of (manual or optitrack or ...) annotation and ScaledYOLOv4 annotation and saves annotated images for comparison')
    parser.add_argument('--coco-path', type=str, required=True,
                        help='path to the COCO dataset directory containing annotator annoated images: annotations/data.json file and images/ folder')
    parser.add_argument('--output-path', type=str, required=True, help='comparison images of annotator annotations and yolo annotations output folder')
    parser.add_argument('--annotator', type=str,
                        help='''
                        define an additional label to shown near plotted bounding box.
                        (e.g., `--annotator manual` to label manual annotation, `--annotator OptiTrack` for automated annotation from OptiTrack)
                        ''')
    parser.add_argument('--segmentation', action='store_true',
                        help='enable binary segmentation mask in the output')
    parser.add_argument('--annotator-labels-path', type=str, help='''
                        provide path to file containing `category_id labels` corresponding to annotations/data.json (e.g., labels/aau.customnames)
                        ''')
    parser.add_argument('--category-id-is-line', action='store_true', 
                        help='enable if `--labels-path` contains `labels` and the line number equals the `category_id`')
    parser.add_argument('--image-ids', nargs='*', # nargs: creates a list; 0 or more values expected
                        help='''
                        ID list of images used for object detection and plotting of annotations (e.g., --image_ids 2 4 8 16 32).
                        Detect objects in a number of randomly selected images (e.g., --image_ids random 5).
                        If not passed as an argument all images are detected (assumes consecutive numbering starting with ID 1)
                        ''')
    parser.add_argument('--show-yolo', action='store_true', help='show `yolo` as additional label near plotted bounding box')
    parser.add_argument('--weights', nargs='+', type=str, help='model.pt path(s)')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save yolo-lable-format (xywh) results to *.txt')
    parser.add_argument('--save-results', action='store_true', help='save comparison results to `output-path/comparison.json`')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--inside-label', action='store_true', help='place yolo label inside the bounding box')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                plot_compare()
                strip_optimizer(opt.weights)
        else:
            plot_compare()