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
                        get_annoation_data_fpath, load_json, get_subdir_paths)


'''
    :Reads dataset with multiple COCO data format recordings: from `--recordings_path` with the following filestructure: 
    ./recordings_path
        ./recordings_path/1/coco_output/images/
            ./recordings_path/1/coco_output/images/*.png
        ./recordings_path/1/coco_output/annotations/
            ./recordings_path/1/coco_output/annotations/data.json
    :Runs object detection on images and plots resulting bounding boxes to images: specified by `--image_ids` 
        and saves the results to a new folder `detected_images/` in the corresponding ./recordings_path/N/output/ folder
    :YOLO implementation taken from: https://github.com/WongKinYiu/ScaledYOLOv4
'''
def detect():
    recordings_path, weights, save_txt, img_size, image_ids, save_coco = \
        opt.recordings_path, opt.weights, opt.save_txt, opt.img_size, opt.image_ids, opt.save_coco

    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recording_paths)

    # --- ScaledYolov4
    # initialize
    device = select_device(opt.device)
    half = device.type != 'cpu'  # half precision only supported on CUDA
    
    # load model
    model = attempt_load(weights, map_location=device)  # load FP32 model
    img_size = check_img_size(img_size, s=model.stride.max())  # check img_size
    if half:
        model.half()  # to FP16

    # get names and colors of model
    names = model.module.names if hasattr(model, 'module') else model.names
    colors = [[random.randint(0, 255) for _ in range(3)] for _ in range(len(names))]

    t0 = time.time() # init time
    img = torch.zeros((1, 3, img_size, img_size), device=device) # Init img
    _ = model(img.half() if half else img) if device.type != 'cpu' else None  # run once

    # detect recording wise
    for rec_id, recording_path in enumerate(recording_paths):   
        # init filepaths
        coco_path = os.path.join(recording_path, 'output')
        all_img_paths = get_image_paths(coco_path)
        if not len(all_img_paths):
            print('Error: No images for object detection found in coco path (%s)' % coco_path)
            return

        d_images_output_path = os.path.join(coco_path, 'detected_images')
        init_output_path(d_images_output_path)
        image_ids = get_img_ids_from_arguments(image_ids, len(all_img_paths), '--image_ids')

        # init save_coco
        if save_coco:
            coco_annotation_data_fpath = get_annoation_data_fpath(coco_path) # get annotation data to gather images-info for detection *.json file
            if not len(all_img_paths) or not os.path.exists(coco_annotation_data_fpath):
                print('Error: No images or annotations to process found in coco path (%s)' % coco_path)
                return
            coco_annotation_data = load_json(coco_annotation_data_fpath)
            annotations = []
            categories = []
            images = [] # image-info corresponding to annotation COCO data format
            annotation_id = 0

        # inference for 1 image at a time of recording `rec_id`
        for img_id in image_ids: 
            id_img_path = get_id_img_path(img_id, all_img_paths)
            id_img_annotations, id_img_categories = scaledyolov4_detect(id_img_path, img, model, names, colors, device, half, save_txt, opt, 
                                                        output_path=d_images_output_path, ret_data=save_coco)
        
            # optional: save results in COCO data format
            if save_coco:
                # add detected annotations to annotations
                add_detected_annotations(id_img_annotations, annotations, annotation_id=annotation_id, img_id=img_id)
                annotation_id += len(id_img_annotations)

                # add detected category to categories
                add_detected_categories(id_img_categories, categories)

                # add image info to images
                add_image_info(coco_annotation_data, coco_annotation_data_fpath, images, img_id)
        
        # store coco data
        if save_coco:
            json_path = str(Path(d_images_output_path) / 'data.json')
            data = {'images': images, 'annotations': annotations, 'categories': categories}
            with open(json_path, 'w') as outfile:
                json.dump(data, outfile)
            print('Yolo annotation file saved (%s)' % json_path)
        
        print('Inference done for all images (recording: %s)' % rec_id)
            
    print('Object detection done. (%.3fs)' % (time.time() - t0))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='runs object detection on images of COCO dataset to plot resulting bounding boxes and generates annotated images')
    parser.add_argument('--weights', nargs='+', type=str, default='yolov4-p5.pt', help='model.pt path(s)')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/ ... OptiTrack_recordings/N/output/ 
                        containing COCO data format annotation/image pairs (annotations/data.json and images/ folder)
                        ''')
    parser.add_argument('--image-ids', nargs='*', required=True, # nargs: creates a list; 0 or more values expected
                    help='''
                    ID list of images used for object detection - same for every recording (e.g., --image_ids 2 4 8 16 32).
                    Detect objects in a number of randomly selected images - different for every recording (e.g., --image_ids random 5).
                    If not passed as an argument all images are detected
                    ''')
    parser.add_argument('--show-yolo', action='store_true', help='show `yolo` as additional label near generated bounding box')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.5, help='IOU threshold for NMS')
    parser.add_argument('--device', default='', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-coco', action='store_true', help='save COCO data format bboxes to `detected_images/data.json`')
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