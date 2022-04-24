from pathlib import Path
import cv2
import torch

from ScaledYOLOv4.utils.general import non_max_suppression, scale_coords, xyxy2xywh
from ScaledYOLOv4.utils.torch_utils import time_synchronized
from ScaledYOLOv4.utils.datasets import LoadImages

from lib.base import plot_one_bbox, xyxy2coco, get_bbox_object_center

'''
    :Runs object detection on image and plots resulting bounding boxes to image: specified by `id_img_path` and saves the result to `output_path`
    :Runs object detection on image and plots resulting bounding boxes to image: specified by `id_img_path` and directly shows the result if `show` is set `True`
    :YOLO implementation taken from: https://github.com/WongKinYiu/ScaledYOLOv4
    :Returns: xyxy bboxes for detected image as list
'''
def scaledyolov4_detect(id_img_path, img, model, names, colors, device, half, save_txt, opt, output_path=None, 
                            ret_data: bool=False, show: bool=False):
    save_txt, img_size, inside_label, plot_detection_centers = \
        opt.save_txt, opt.img_size, opt.inside_label, opt.plot_detection_centers
    
    if output_path is None and show is False:
        return

    # saving condition for results
    save_img = False if output_path is None else True

    # set Dataloader
    dataset = LoadImages(id_img_path, img_size=img_size)
    
    for path, img, im0s, vid_cap in dataset:
        img = torch.from_numpy(img).to(device)
        img = img.half() if half else img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)

        # inference
        t1 = time_synchronized()
        pred = model(img, augment=opt.augment)[0]

        # apply NMS
        pred = non_max_suppression(pred, opt.conf_thres, opt.iou_thres, classes=opt.classes, agnostic=opt.agnostic_nms)
        t2 = time_synchronized()

        # process detections
        for det in pred:  # detections per image
            p, s, im0 = path, '', im0s

            save_path = str(Path(output_path) / Path(p).name)
            txt_path = str(Path(output_path) / Path(p).stem) + ('_%g' % dataset.frame if dataset.mode == 'video' else '')
            s += '%gx%g ' % img.shape[2:]  # print string
            gn = torch.tensor(im0.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            if det is not None and len(det):
                # rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], im0.shape).round()

                # print results
                for c in det[:, -1].unique():
                    n = (det[:, -1] == c).sum()  # detections per class
                    s += '%g %ss, ' % (n, names[int(c)])  # add to string

                
                if ret_data:
                    annotations = [] # yolo generated bboxes
                    categories = [] # label categories
                    category_ids = []
                
                # write results
                for *xyxy, conf, cls in det: # TODO: use confidence score?

                    if save_txt:  # write to file
                        xywh = (xyxy2xywh(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * 5 + '\n') % (cls, *xywh))  # label format
                    
                    if save_img or show: # add bbox to image
                        label = '%s' % (names[int(cls)])
                        if opt.show_yolo:
                            plot_one_bbox(xyxy, im0, label_inside_pos=inside_label, color=colors[int(cls)], label=label, line_thickness=1, annotator='yolo')
                        else:
                            plot_one_bbox(xyxy, im0, label_inside_pos=inside_label, color=colors[int(cls)], label=label, line_thickness=1)

                        if plot_detection_centers:
                            bbox = xyxy2coco(xyxy)
                            center_point = get_bbox_object_center(bbox)
                            cv2.circle(im0, (round(center_point[0]), round(center_point[1])), radius=8, color=colors[int(cls)], thickness=-1)

                        if ret_data:
                            # save category
                            category_id = int(cls)
                            if category_id not in category_ids:
                                categories.append({'name': label, 'id': category_id})
                                category_ids.append(category_id)

                            # save bboxes
                            bbox = xyxy2coco(xyxy)
                            annotation = {'bbox': bbox, 'category_id': int(cls)}
                            annotations.append(annotation)

            # print time (inference + NMS)
            print('%sDone. (%.3fs)' % (s, t2 - t1))

            if save_img: # save results (image with detections)
                cv2.imwrite(save_path, im0)
            
            if show: # show results
                cv2.imshow(p, im0)
                if cv2.waitKey(1) == ord('q'):  # q to quit
                    raise StopIteration

    if save_txt:
        print('Result *txt saved to %s' % Path(output_path))

    if ret_data: return annotations, categories