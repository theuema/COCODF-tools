import argparse
from re import I
import sys
import os
import numpy as np
from pathlib import Path

from lib.write_rsg_files import write_cl, write_enh, write_txt 
from lib.base import init_output_path, load_json, get_img_ids_from_arguments, load_B_C, quat2rot, get_image_annotation_object_center

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
    :Saves: the following files to a new directory `rsg/` next to the json file specified by `--annotation-data-path`
         - 0000000Image_ID_3D-pose-cam.txt: (1 file per image/frame, rows #1) the complete pose of the camera (XYZ, rotation matrix, quaternion) for a given image/frame
         - 0000000Image_2D-pos-objects-opt.cl: (1 file per image/frame, rows: #N annotations) the object/category ID + center point (XY) of all objects visible in the image/frame with id Image_ID
         - 0000000Image_2D-pos-objects-det.cl: (1 file per image/frame, rows: #N annotations) the object/category ID + center point (XY) of all objects detected in the image/frame with id Image_ID
         - 3D-pose-objects.enh: (1 file per recording, rows: rows: #N tracked objects) the object/category ID + complete pose of all tracked objects (static objects 'digital twin')
'''

def transform():
    ann_annotation_data_fpath, det_annotation_data_fpath, image_ids, B_C_fpath = opt.ann_annotation_data_path, opt.det_annotation_data_path, opt.image_ids, opt.BCcam_path
    
    # init (file)paths
    try: # annotation_data file path check
        if not os.path.isfile(ann_annotation_data_fpath):
            raise AttributeError('File given to `--opt-annotation-data-path` does not exist.')
        if not os.path.isfile(det_annotation_data_fpath):
            raise AttributeError('File given to `--det-annotation-data-path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    ann_coco_annotation_data = load_json(ann_annotation_data_fpath)
    det_coco_annotation_data = load_json(det_annotation_data_fpath)
    save_path = str(Path(ann_annotation_data_fpath).parents[1] / 'rsg')
    init_output_path(save_path)

    # get image_ids for which annotation data is transformed to RSG data
    image_ids = get_img_ids_from_arguments(image_ids, len(ann_coco_annotation_data['images']), '--image_ids')

    # CL1, ENH
    # calculate 2D object center coordinates from annotator annotations
    # get 3D world plane position from annotator annotations 
    # 2D BoundingBox centers (.cl file) & 3D W position (.enh file)
    category_ids = []
    cl_object_center_2D_dicts = []
    ann_image_dicts = ann_coco_annotation_data['images']
    for annotation in ann_coco_annotation_data['annotations']: 
        # get corresponding image
        image = next((image for image in ann_image_dicts if image['id'] == annotation['image_id']), None)
        if image is None:
            print('Error: Image ID not found in images (%s)', annotation['image_id'])
            sys.exit(1)
        image_name = str(Path(image['file_name']).stem)

        category_id = int(annotation['category_id'])
        image_id = int(annotation['image_id'])

        # CL: create image specific data - annotator 2D object center-, or better bounding box center coordinates
        if image_id in image_ids: # len(annotations) times each image
            object_center_2D = get_image_annotation_object_center(annotation['bbox'])
            cl_object_center_2D_dicts.append({'category_id': category_id, 'object_center_2D': object_center_2D, 'image_name': image_name}) 

        # ENH: create static object data (static: 3D object coordinates for the physical model that is not moved during the recording process)
        # object_poses through images slightly deviate, due to the accuracy of the tracking system
        # TODO: Might be an idea to gather every single object_pose from every image/frame and take the mean value 
        enh_objects_pos_3D_dict = {}
        if category_id not in category_ids: # len(objects) times (number of objects/categories from the recording)
            enh_objects_pos_3D_dict[category_id] = annotation['object_pose']['position']
            category_ids.append(annotation['category_id'])

    # write 2D object annotator annotation center coordinates (.CL columnt/line file)
    write_cl(str(Path(save_path / 'cl_ann')), cl_object_center_2D_dicts)
    # write 3D object position, rotation matrix and quaternions (object_pose) from a (physically) static model (.ENH file) 
    write_enh(save_path, enh_objects_pos_3D_dict) 
   
    # CL2
    # calculate 2D object center coordinates from object detection for subsequent triangulation
    # 2D BB centers (.cl file)
    cl_object_center_2D_dicts = []
    det_image_dicts = det_coco_annotation_data['images']
    for annotation in det_coco_annotation_data['annotations']:
        # get corresponding image
        image = next((image for image in det_image_dicts if image['id'] == annotation['image_id']), None)
        if image is None:
            print('Error: Image ID not found in images (%s)', annotation['image_id'])
            sys.exit(1)
        image_name = str(Path(image['file_name']).stem)

        category_id = int(annotation['category_id'])
        image_id = int(annotation['image_id'])

        # CL: create image specific data - object-detector annotation center coordinates
        if image_id in image_ids: # len(annotations) times each image 
            object_center_2D = get_image_annotation_object_center(annotation['bbox'])
            cl_object_center_2D_dicts.append({'category_id': category_id, 'object_center_2D': object_center_2D, 'image_name': image_name}) 

    # write 2D object object detection annotation center coordinates (.CL columnt/line file)
    write_cl(str(Path(save_path / 'cl_det')), cl_object_center_2D_dicts)
     
    # TXT
    # calculate camera position (= camera plane camera center position) and camera rotation (= rotation from world plane W to camera plane C) from annotator annotations
    # generated file is for comparing with results from the triangulation (.txt)
    image_names = []
    B_C = load_B_C(B_C_fpath)
    for annotation in ann_coco_annotation_data['annotations']: 
        # get corresponding image
        image = next((image for image in ann_image_dicts if image['id'] == annotation['image_id']), None)
        if image is None:
            print('Error: Image ID not found in images (%s)', annotation['image_id'])
            sys.exit(1)
        image_name = str(Path(image['file_name']).stem)

        category_id = int(annotation['category_id'])
        image_id = int(annotation['image_id'])

        # TXT: create image specific data - camera center position in W, camera rotation W->C
        if image_id in image_ids: # len(annotations) times each image 
            if image_name not in image_names: # 1x each image (= camera position (W) and rotation (W->C) for each image)
                # get rotation from world coordinates to camera plane (R_wc)
                # get the camera center position in world coordinates (t_wc)
                    # can be used for 3D/2D projection when building the extrinsic matrix

                # get rotation from W->B, position of the camera body in W, rotation and translation for B->C 
                R_bc = B_C[:3, :3]  # rotation from B->C (from extrinsic calibration) 
                t_bc = B_C[:3, 3:]  # translation from B->C (from extrinsic calibration) 
                Q_wb = annotation['camera_pose']['quaternion']
                t_wb = np.array(annotation['camera_pose']['position']).reshape(1,3) # t_wb position of camera body in world coordinates
                R_wb = quat2rot(Q_wb) # R_wb rotation from world plane to camera (rigid) body plane B
                
                # translate to camera plane
                R_wc = np.matmul(R_wb, R_bc) # R_wc rotation from world plane W to camera plane C 
                t_wc = t_wb.T + np.matmul(R_wb, t_bc) # new camera center in world coordinates (referenced to as "C" when dissecting the camera matrix to intrinsic and extrinsic matrix)

                # write 3D camera position
                write_txt(save_path, image_name, R_wc, t_wc)
                image_names.append(image_name)

    print('Done writing position/pose data for RSG (%s)' % save_path)  #f.write(('%g ' * 5 + '\n') % (cls, *xywh))
    print('Data transformed for images ', image_names)

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
    parser.add_argument("--BCcam-path", type=str, required=True,
                    help="Path to the yaml-file containing the camera rigid body transformation matrix")
    opt = parser.parse_args()
    print(opt)

    transform()