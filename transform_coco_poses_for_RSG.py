import argparse
from re import I
import sys
import os
import numpy as np
from pathlib import Path

from lib.write_rsg_files import write_cl, write_enh, write_txt 
from lib.base import init_output_path, load_json, get_img_ids_from_arguments, quat2rot, get_image_annotation_object_center, calc_camera_frame, perform_custom_camera_frame_rotations

'''
    :Takes COCO data format annotation json file specified by `--annotation-data-path`
    :Works with the following json structure:
        images/
        images/1...N/file_name
        annotations/
            annotations/1...N/object_pose/position[3] (absolute position of an object in a predefined coordinate system)
            annotations/1...N/object_pose/quaterion[4] (object pose quaternion)
            annotations/1...N/object_pose/rotation[9] (object pose rotation matrix)
            annotations/1...N/camera_pose/position[3] (absolute position of the camera for a particular image/frame in a predefined coordinate system)
            annotations/1...N/camera_pose/quaterion[4] (camera pose quaternion)
            annotations/1...N/camera_pose/rotation[9] (camera pose rotation matrix)
            annotations/1...N/bbox[4] (bbox of the annotated object of the 2D image plane in COCO format)
    :Saves: the following files to a new directory `rsg/` in recording root 
         - rsg/CPOSs.txt (Camera Positions):        (1 file per recording, rows #image_ids)             the complete pose of the camera for an image/frame              (ImageId;X;Y;Z;Rm00;Rm01;Rm02;Rm10;Rm11;Rm12;Rm20;Rm21;Rm22;ImageName)
         - rsg/GCPs.enh (Ground Control Points):    (1 file per recording, rows: #N tracked objects)    the positions of all tracked objects in W 'digital twin'        (Id;E;N;H;SigE;SigN;SigH;Status;Comment)
         - rsg/cl_ann/image_name.cl:                (1 file per image, rows: #N annotated annotations)  the center points of the annotated bounding boxes in the image  (Id;C;L;Status;SigImg) 
         - rsg/cl_det/image_name.cl:                (1 file per image, rows: #N detected annotations)   the center points of the detected bounding boxes in the image   (Id;C;L;Status;SigImg) 
'''

def transform():
    ann_annotation_data_fpath, det_annotation_data_fpath, image_ids, B_C_fpath = opt.ann_annotation_data_path, opt.det_annotation_data_path, opt.image_ids, opt.BCcam_path
    
    # init (file)paths
    try: # annotation_data file path check
        if not os.path.isfile(ann_annotation_data_fpath):
            raise AttributeError('File given to `--opt-annotation-data-path` does not exist.')
        if not os.path.isfile(det_annotation_data_fpath): # detection_data file path check
            raise AttributeError('File given to `--det-annotation-data-path` does not exist.')
        if not os.path.isfile(B_C_fpath): # BCcam_path file path check
            raise AttributeError('File given to `--BCcam_path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    ann_coco_annotation_data = load_json(ann_annotation_data_fpath)
    det_coco_annotation_data = load_json(det_annotation_data_fpath)
    save_path = str(Path(ann_annotation_data_fpath).parents[1] / 'rsg')
    init_output_path(save_path)

    # get image_ids for which annotation data is transformed to RSG data
    image_ids = get_img_ids_from_arguments(image_ids, len(ann_coco_annotation_data['images']), '--image_ids')

    # Annotator annotation run
    # CL: calculate 2D object center coordinates from annotator annotations
    # ENH: get 3D world/reference frame position from annotator annotations 
    # TXT: calculate camera frame origin in W (t_wc) and camera frame orientation in W (R_wc) from annotator annotations
    category_ids = []
    image_fnames = []
    cl_object_center_2D_dicts = []
    enh_object_positions_3D_dict = {}
    txt_C_camera_pose_dict = {}
    ann_image_dicts = ann_coco_annotation_data['images']
    for annotation in ann_coco_annotation_data['annotations']: 
        # get corresponding image
        image = next((image for image in ann_image_dicts if image['id'] == annotation['image_id']), None)
        if image is None:
            print('Error: Image ID not found in images (%s)', annotation['image_id'])
            sys.exit(1)
        image_fname = str(Path(image['file_name']))

        category_id = int(annotation['category_id'])
        image_id = int(annotation['image_id'])

        if image_id in image_ids: # len(annotations) times each image
            # CL: create image specific data - annotator 2D object center-, or better bounding box center coordinates
            object_center_2D = get_image_annotation_object_center(annotation['bbox'])
            cl_object_center_2D_dicts.append({'category_id': category_id, 'object_center_2D': object_center_2D, 'image_fname': image_fname}) 

            # TXT: calculate camera position & rotation in camera frame
            if image_fname not in image_fnames: # 1x each image (= 1 camera body pose (R_wb, t_wb) each image)
                # get orientation of C in W (R_wc)
                # get the camera frame center position in world coordinates (t_wc) (translation from W orign to C origin)
                
                Q_wb = annotation['camera_pose']['quaternion']
                R_wb = np.asarray(quat2rot(Q_wb)) # R_wb = orientation of B (camera body) in W
                t_wb = np.array(annotation['camera_pose']['position']).reshape(1,3) # t_wb position of camera body in world coordinates (where the origin of B is displaced by t_wb from O of W)
                R_wc, t_wc = calc_camera_frame(B_C_fpath, R_wb, t_wb)
                R_wcp = perform_custom_camera_frame_rotations(R_wc)

                txt_C_camera_pose_dict[image_id] = {'position': t_wc, 'rotation': R_wcp, 'image_fname': image_fname}
                image_fnames.append(image_fname)

        # ENH: create static object data (static: 3D object coordinates for the physical model that is not moved during the recording process)
        # object_poses through images slightly deviate, due to the accuracy of the tracking system
        # TODO: Might be an idea to gather every single object_pose from every image/frame and take the mean value 
        if category_id not in category_ids: # len(objects) times (number of objects/categories from the recording)
            enh_object_positions_3D_dict[category_id] = annotation['object_pose']['position']
            category_ids.append(annotation['category_id'])

    # write 2D object annotator annotation center coordinates (.CL columnt/line file)
    init_output_path(save_path + '/cl_ann')
    write_cl(save_path + '/cl_ann', cl_object_center_2D_dicts)
    # write 3D object position, rotation matrix and quaternions (object_pose) from a (physically) static model (.ENH file) 
    write_enh(save_path, enh_object_positions_3D_dict) 
    # write 3D camera position of camera frame 
    write_txt(save_path, txt_C_camera_pose_dict)
    
    # Object detection run 
    # CL: calculate 2D object center coordinates from object detection - 2D BB centers (.cl file)
    cl_object_center_2D_dicts = []
    det_image_dicts = det_coco_annotation_data['images']
    for annotation in det_coco_annotation_data['annotations']:
        # get corresponding image
        image = next((image for image in det_image_dicts if image['id'] == annotation['image_id']), None)
        if image is None:
            print('Error: Image ID not found in images (%s)', annotation['image_id'])
            sys.exit(1)
        image_fname = str(Path(image['file_name']).stem)

        category_id = int(annotation['category_id'])
        image_id = int(annotation['image_id'])

        # CL: create image specific data - object-detector annotation center coordinates
        if image_id in image_ids: # len(annotations) times each image 
            object_center_2D = get_image_annotation_object_center(annotation['bbox'])
            cl_object_center_2D_dicts.append({'category_id': category_id, 'object_center_2D': object_center_2D, 'image_fname': image_fname}) 

    # write 2D object object detection annotation center coordinates (.CL columnt/line file)
    init_output_path(save_path + '/cl_det')
    write_cl(save_path + '/cl_det', cl_object_center_2D_dicts)

    print('Done writing position/pose data for RSG (%s)' % save_path)
    print('Data transformed for images ', image_fnames)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes COCO data format annotation json file and extracts image-specific camera and object position and pose data and saves information in ASCII format')
    parser.add_argument('--ann-annotation-data-path', type=str, required=True,
                        help='(File)path to the COCO data format annotation `*.json` (manually) annotated-generated file, containing absolute 3D camera positions for each image/frame, absolute 3D object positions for each recording (the physical model tracked) and Bboxes (COCO format) for each object in each image/frame')
    parser.add_argument('--det-annotation-data-path', type=str, required=True,
                        help='(File)path to the COCO data format annotation `*.json` Object-detector-generated file containing Bboxes (COCO format) for each detected object in each image/frame')
    parser.add_argument('--image-ids', nargs='*', # nargs: creates a list; 0 or more values expected
                        help='''
                        (optional) ID list of images for which COCO data is extracted and transformed for RSG (e.g., `--image_ids 2 4 8 16 32`);
                        if not passed as an argument data for every image is transformed (assumes consecutive numbering starting with ID 1)
                        ''')
    parser.add_argument('--BCcam-path', type=str, required=True,
                    help='Path to the yaml-file containing the camera rigid body transformation matrix')
    opt = parser.parse_args()
    print(opt)

    transform()