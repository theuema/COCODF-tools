import cv2
import copy

from pathlib import Path
from lib.base import coco2xyxy, rle2mask, plot_one_bbox

'''
    :Plots bounding boxes of image: specified by `img_path`, generates annotated image and saves the result to `out`
    :Plots bounding boxes of image: specified by `img_path`, generates annotated image and directly shows the result if `show` is set `True`
    :Segmentation mask is visualized: if `segmentation` is set `True`
    :Returns: xyxy bboxes for processed image as list
'''
def plot_image_annotations(img_path, annotations, colors, labels: dict, annotator: str=None, segmentation: bool=False, out=None, show: bool=False, img_id=None):
    if out is None and show is False:
        return
    
    # saving condition for results
    save_img = False if out is None else True
    if save_img: save_path = str(Path(out) / Path(img_path).name) 

    # print information
    if img_id:
        print('Annotating image ID {} with number of objects: {}'.format(img_id, len(annotations)))
    else: 
        print('Annotating image {} with number of objects: {}'.format(Path(img_path).name, len(annotations)))
    
    # write results
    img = cv2.imread(img_path)
    id_annotated_img = copy.deepcopy(img)
    bboxes = []
    for i, annotation in enumerate(annotations):
        bbox = annotation['bbox']
        bboxes.append(bbox)
        xyxy = coco2xyxy(bbox)
        plot_one_bbox(xyxy, id_annotated_img, color=colors[i], label=labels[annotation['category_id']], line_thickness=1, annotator=annotator)

    # insert segmentation mask
    if segmentation:
        print('Inserting segmentation mask ...')
        id_annotated_segm_img = id_annotated_img
        for annotation in annotations:
            binary_mask_img = rle2mask(annotation['segmentation']['counts'], 
                                            annotation['segmentation']['size'][1], annotation['segmentation']['size'][0])
            id_annotated_segm_img = cv2.addWeighted(id_annotated_segm_img, 1.0, binary_mask_img, 1, 0)
        
        if save_img: # save image
            cv2.imwrite(save_path, id_annotated_segm_img)
        
        if show: # show image stages
            cv2.imshow('Raw image', img)
            cv2.waitKey(0)
            cv2.imshow('Annotated image', id_annotated_img)
            cv2.waitKey(0)
            cv2.imshow('Binary Mask', binary_mask_img)
            cv2.waitKey(0)
            cv2.imshow('Segmentation Image', id_annotated_segm_img)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        
        return

    if save_img:
        cv2.imwrite(save_path, id_annotated_img)
    
    if show: # show image stages
        cv2.imshow('Raw image', img)
        cv2.waitKey(0)
        cv2.imshow('Annotated image', id_annotated_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    
    return bboxes