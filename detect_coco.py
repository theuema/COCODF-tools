import argparse
import sys
import torch
import time
import os
import json
from pathlib import Path
from numpy import random

from ScaledYOLOv4.utils.general import strip_optimizer
from ScaledYOLOv4.utils.torch_utils import select_device
from ScaledYOLOv4.models.experimental import attempt_load
from ScaledYOLOv4.utils.general import check_img_size

from lib.scaledyolov4_detect import scaledyolov4_detect
from lib.handle_detected_annotation_data import add_detected_annotations, add_detected_categories, add_image_info
from lib.base import (init_output_path, get_image_paths, get_img_ids_from_arguments, get_id_img_path, 
                        get_annoation_data_fpath, load_json)


'''
    :Reads COCO data format dataset: from `--coco_path` with the following filestructure: 
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json
    :Runs object detection on images and plots resulting bounding boxes to images: specified by `--image_ids` and saves the results to `--output_path`
    :Runs object detection on images and plots resulting bounding boxes to images: specified by `--show_ids` and directly shows the results
        `--image_ids` and `--show_ids` correspond to the images present in images/ sorted by filename
    :YOLO implementation taken from: https://github.com/WongKinYiu/ScaledYOLOv4
'''
def detect():
    output_path, coco_path, weights, save_txt, img_size, show_ids, image_ids, save_coco = \
        opt.output_path, opt.coco_path, opt.weights, opt.save_txt, opt.img_size, opt.show_ids, opt.image_ids, opt.save_coco

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

    # init filepaths
    all_img_paths = get_image_paths(coco_path)
    if not len(all_img_paths):
        print('Error: No images for object detection found in coco path (%s)' % coco_path)
        return

    # init save
    if save: 
        init_output_path(output_path)
        image_ids = get_img_ids_from_arguments(image_ids, len(all_img_paths), '--image_ids')

    # init show
    if show: show_ids = get_img_ids_from_arguments(show_ids, len(all_img_paths), '--show_ids')
    
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

    # run object detection, plot bounding boxes and show image(s)
    if show:
        # inference for 1 image at a time
        for img_id in show_ids:
            id_img_path = get_id_img_path(img_id, all_img_paths)
            _ = scaledyolov4_detect(id_img_path, img, model, names, colors, device, half, save_txt, opt, show=True)
        print('Inference done for all images.')
    
    # run object detection, plot bounding boxes and save image(s) / coco data
    if save:
        # init save_coco
        if save_coco:
            coco_annotation_data_fpath = get_annoation_data_fpath(coco_path) # get annotation data to gather images-info for detection *.json file
            if not len(all_img_paths) or not os.path.isfile(coco_annotation_data_fpath):
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
            img_id_annotations, id_img_categories = scaledyolov4_detect(id_img_path, img, model, names, colors, device, half, save_txt, opt, 
                                                        output_path=output_path, ret_data=save_coco)
            
            # optional: save results in COCO data format
            if save_coco:
                # add detected annotations to annotations
                add_detected_annotations(img_id_annotations, annotations, annotation_id=annotation_id, img_id=img_id)
                annotation_id += len(img_id_annotations)

                # add detected category to categories
                add_detected_categories(id_img_categories, categories)

                # add image info to images
                add_image_info(coco_annotation_data, coco_annotation_data_fpath, images, img_id)

        # store coco data
        if save_coco:
            json_path = str(Path(output_path) / 'data.json')
            data = {'images': images, 'annotations': annotations, 'categories': categories}
            with open(json_path, 'w') as outfile:
                json.dump(data, outfile)
            print('Yolo annotation file saved (%s)' % json_path)

        print('Inference done for all images (%s)' % output_path)
        

    print('Object detection done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs object detection on images of COCO dataset to plot resulting bounding boxes and generates annotated images')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--coco-path', type=str, required=True,
                        help='path to the COCO dataset directory containing annotations/data.json file and images/ folder')
    parser.add_argument('--output-path', type=str, help='image(s) output folder')
    parser.add_argument('--show-ids', nargs='*', help='ID list of images that are displayed as an example')
    parser.add_argument('--image-ids', nargs='*', # nargs: creates a list; 0 or more values expected
                    help='''
                    ID list of images used for object detection (e.g., --image_ids 2 4 8 16 32).
                    Detect objects in a number of randomly selected images (e.g., --image_ids random 5).
                    If not passed as an argument all images are detected (assumes consecutive numbering starting with ID 1)
                    ''')
    parser.add_argument('--show-yolo', action='store_true', help='show `yolo` as additional label near generated bounding box')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save yolo-lable-format (xywh) results to *.txt')
    parser.add_argument('--save-coco', action='store_true', help='save COCO data format bboxes to `output-path/data.json`')
    parser.add_argument('--classes', nargs='+', type=int, help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--inside-label', action='store_true', help='place label inside the bounding box')
    parser.add_argument('--update', action='store_true', help='update all models')
    opt = parser.parse_args()
    print(opt)

    with torch.no_grad():
        if opt.update:  # update all models (to fix SourceChangeWarning)
            for opt.weights in ['']:
                detect()
                strip_optimizer(opt.weights)
        else:
            detect()