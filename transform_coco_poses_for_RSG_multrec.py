import argparse
import sys
import os
import numpy as np
from pathlib import Path

from lib.write_rsg_files import write_cl, write_enh, write_txt 
from lib.base import init_output_path, load_json, get_img_ids_from_arguments, get_subdir_paths, quat2rot, get_image_annotation_object_center, calc_camera_frame, perform_photogrammetric_camera_frame_rotations, get_perfect_obj_pos_proj, rotate_orientation_upside_down, load_camera_intrinsics, get_projected_point

'''
    :Takes COCO data format annotation json file from recordings specified by `--recordings-path` and `--annotation-json-name`
    :Works with the following file structure:
    ./recordings_path
        ./recordings_path/N/
            ./recordings_path/1/coco_output/annotations/
                ./recordings_path/1/coco_output/annotations/ann-annotation-json-name.json
                ./recordings_path/1/coco_output/detected_images/det-annotation-json-name.json
        ...
    
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
    recordings_path, ann_annotation_json_name, det_annotation_json_name, image_ids, B_C_fpath, Cam_upsidedown, camera_yaml= opt.recordings_path, opt.ann_annotation_json_name, opt.det_annotation_json_name, opt.image_ids, opt.BCcam_path, opt.Cam_upsidedown, opt.camera_yaml
    
    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    for recording_path in recording_paths: 
        # init (file)paths for current recording
        ann_annotation_data_fpath = str(Path(recording_path) / 'output' / 'annotations' / Path(ann_annotation_json_name).with_suffix('')) + '.json'
        det_annotation_data_fpath = str(Path(recording_path) / 'output' / 'detected_images' / Path(det_annotation_json_name).with_suffix('')) + '.json'
        
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
        # ENH: get 3D world plane position from annotator annotations 
        # TXT: calculate camera position (= camera plane camera center position) and camera rotation (= rotation from world plane W to camera plane C) from annotator annotations
        category_ids = []
        image_fnames = []
        cl_object_center_2D_dicts = []
        cl_perfect_obj_pos_proj_dicts = []
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

                # calculate camera position & rotation in camera frame
                # get rotation from world coordinates to camera frame (R_wc)
                # get the camera center position in world coordinates (t_wc)
                Q_wb = annotation['camera_pose']['quaternion']
                R_wb = np.asarray(quat2rot(Q_wb)) # R_wb rotation from world frame to camera (rigid) body frame B
                t_wb = np.array(annotation['camera_pose']['position']).reshape(1,3) # t_wb position of camera body in world coordinates
                R_wc, t_wc = calc_camera_frame(B_C_fpath, R_wb, t_wb)

                # CL: create ground truth object projection
                # Calculate extrinsics matrix & make homogeneous
                # R_wcextr = rotate_orientation_upside_down(R_wc) if Cam_upsidedown else R_wc
                # TODO: using the upper line (upside down correction of camera pose) to compose extrinsic mat does not yield the correct projection! 
                # wrong extrinsic calibration transformation (B_C) expected in function calc_camera_frame
                # if using this in the future, delete if condition line 127;
                extrinsic_mat = np.hstack((R_wc.T, -np.matmul(R_wc.T, t_wc)))
                extrinsic_mat = np.vstack((extrinsic_mat, np.array([0, 0, 0, 1])))
    
                # load intrinsics & make homogeneous
                intrinsics = load_camera_intrinsics(camera_yaml)
                intrinsic_mat = np.vstack((intrinsics['intrinsic_matrix'], np.array([0, 0, 0])))
                intrinsic_mat = np.hstack((intrinsic_mat, np.array([[0], [0], [0], [1]])))
                
                # get point & make homogeneous
                obj_pos = np.append(np.array(annotation['object_pose']['position']), 1).reshape(4,1)

                # project object pos for ground truth 2D C/L projection
                perfect_obj_pos_proj_2D = get_perfect_obj_pos_proj(extrinsic_mat, intrinsic_mat, obj_pos)

                if Cam_upsidedown: # need to rotate on 2D image level due to wrong orientation after 3D rotation; see line 110
                    image_w = int(image['width'])
                    image_h = int(image['height'])
                    xyr = get_projected_point(o=(image_w / 2, image_h / 2), xy=perfect_obj_pos_proj_2D, degree=180)
                    perfect_obj_pos_proj_2D = (round(xyr[0]), round(xyr[1])) # round to nearest integer

                cl_perfect_obj_pos_proj_dicts.append({'category_id': category_id, 'object_center_2D': perfect_obj_pos_proj_2D, 'image_fname': image_fname}) 

                # TXT: write camera frame position and orientation, conform to the photogrammetric system 
                if image_fname not in image_fnames: # 1x each image (= camera position (W) and rotation (W->C) for each image)
                    # generate RSG conform rotation matrix TODO: calculate degrees instead of Rmat
                    R_wcp = perform_photogrammetric_camera_frame_rotations(R_wc, Cam_upsidedown)
                    
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
        # write 2D ground truth object projection center coordinates (.CL columnt/line file)
        init_output_path(save_path + '/cl_gtruh')
        write_cl(save_path + '/cl_gtruh', cl_perfect_obj_pos_proj_dicts)
        # write 3D object position, rotation matrix and quaternions (object_pose) from a (physically) static model (.ENH file) 
        write_enh(save_path, enh_object_positions_3D_dict) 
        # write 3D camera position of camera plane
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
    
    print('Done writing position/pose data for all recordings (%s)' % recording_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes COCO data format annotation json file from several recordings and extracts image-specific camera and object position and pose data and saves information in ASCII format')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/annotations/*.json ... OptiTrack_recordings/N/output/annotations/*.json
                        containing a COCO data format annotation `.json` file specified by `--*-annotation-json-name`
                        ''')
    parser.add_argument('--ann-annotation-json-name', type=str, required=True, help='Annotator (manual) annotation json filename (e.g., `data.json` or just `data`)')
    parser.add_argument('--det-annotation-json-name', type=str, required=True, help='Object-detector annotation json filename (e.g., `xyz.json` or just `xyz`)')
    parser.add_argument('--image-ids', nargs='*', # nargs: creates a list; 0 or more values expected
                        help='''
                        (optional) ID list of images for which COCO data is extracted and transformed for RSG (e.g., `--image_ids 2 4 8 16 32`);
                        if not passed as an argument data for every image is transformed (assumes consecutive numbering starting with ID 1)
                        ''')
    parser.add_argument('--BCcam-path', type=str, required=True,
                        help='Path to the yaml-file containing the camera rigid body transformation matrix')
    parser.add_argument('--Cam-upsidedown', action='store_true', help='Use if the camera was upside-down during the data acquisition process')
    parser.add_argument('--camera-yaml', type=str, help='Path to the yaml-file containing the camera intrinsics')
    opt = parser.parse_args()
    print(opt)

    transform()