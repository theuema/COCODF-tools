'''
    This script is provided by the Institute of Smart Systems Technologies @ AAU 
    for processing ROS-bags captured by their inhouse dronehall tracking system
    https://www.aau.at/intelligente-systemtechnologien/
    Info: This version is a slightly modified version from the original code in line 16, 28, 64 and 467
'''

import argparse
import os
import time
import yaml
import bagpy
from plyfile import PlyData #m
import pandas as pd
import numpy as np
import rosbag
from cv_bridge import CvBridge
import cv2
from glob import glob
import json
from itertools import groupby
import scipy

from lib.base import quat2rot #+

obj_dict = {
    'HORSE': 18,
    'COW': 19,
    'ELEPHANT': 20,
    'BEAR': 21,
    'ZEBRA': 22,
    'GIRA': 23,
    'ORANGE': 49,
    'BANANA': 46,
    'APPLE': 47,
    'TBALL': 11
}


def obj2idx(obj_name):
    "Function return Category Id of Animal as specified in CoCo format."
    return obj_dict[obj_name]


def load_data(camera_path, object_path):
    camera_recording = load_rosbag(camera_path)
    object_poses = load_rosbag(object_path)
    return camera_recording, object_poses


def load_rosbag(path):
    path = os.path.abspath(path)
    bag = bagpy.bagreader(path)
    return bag


def load_pointclouds(path):
    """
    Point CLouds are loaded here. 
    Input: 
        path: Path to Poincloud. 
    Output: Pointcloud (x,y,z)
    """
    pointclouds = PlyData.read(path) #m
    return pointclouds


def load_camera_intrinsics(path):
    """
    Load Intrinsic Matrix and Image size. 
    Input: Path to Intrinsics File
    Output: caemra_intrinsics containing Image size, focal length, principal_point and intrinsics. 
    """
    with open(path) as f:
        camera_intrinsics = yaml.load(f, Loader=yaml.FullLoader)
    camera_intrinsics["intrinsics"] = np.array(camera_intrinsics['camera_matrix']['data']).reshape(3, 3)
    camera_intrinsics["focal_length"] = [camera_intrinsics["intrinsics"][0, 0], camera_intrinsics["intrinsics"][1, 1]]
    camera_intrinsics["principal_point"] = [camera_intrinsics["intrinsics"][0, 2],
                                            camera_intrinsics["intrinsics"][1, 2]]
    camera_intrinsics["image_size"] = [camera_intrinsics["image_height"], camera_intrinsics["image_width"]]

    return camera_intrinsics


def load_rb_cam(path):
    """
    Load Rigid Body CAmera Frame Trafo Matrix a. 
    Input: Path to Transformation MAtrix File
    Output: caemra_intrinsics containing Image size, focal length, principal_point and intrinsics. 
    """
    with open(path) as f:
        Rb_Cam_yaml = yaml.load(f, Loader=yaml.FullLoader)
    Rb_Cam = np.array(Rb_Cam_yaml['transformation']).reshape(4, 4)
    return Rb_Cam


def calcualte_object_pose(object_data):
    """
    Return the average pose of a single object
    Input: 
        object_data: Data Containting Position and rotation of Object relative to Word Frame
    Output: Object 6DoF POsition (rotation, quaternions and translation from Word Frame)
    """
    object_pose = {}
    means = object_data.mean(0)
    object_pose["position"] = np.array(
        [means['pose.pose.position.x'], means['pose.pose.position.y'], means['pose.pose.position.z']])
    object_pose["quaternion"] = [means['pose.pose.orientation.w'], means['pose.pose.orientation.x'],
                                 means['pose.pose.orientation.y'], means['pose.pose.orientation.z']]
    object_pose["rotation"] = quat2rot(object_pose["quaternion"])
    return object_pose

def annotation(object_poses, point_clouds):
    obj = []
    for topic in object_poses.topics:
        object_name = topic[1:topic.find('_rb')]


def center_objects(data, marker_offset=[0, 0, -0.014]):
    """
 
        Read the object pose in world coordinates.

        Input:
        data: object 3D pose bagfile 
        marker_offset: Offset between marker and real object

        Output: 
            rot: object rotation 3X3 rotation matrix relative to word frame
            trans: object 3D position relative to word frame


    """
    object_pose = calcualte_object_pose(data)
    rot = object_pose["rotation"]
    quat = object_pose["quaternion"]
    trans = np.array([object_pose["position"]]) + marker_offset
    return rot, trans, quat


def center_camera(camera_recording, marker_offset=[-0., -0, 0]):
    """
    Read Camera Position and Camera Rotation relative to Word Frame from Bagfile:
    Input: 
            camera_recording: Path to Bagfile
            ct_o
    Output: return Bounding Box [x_c,y_c, w,h] 
    """
    camera_poses = pd.read_csv(camera_recording.message_by_topic('/synchronizer/camera_poses'))

    # Determine Rotation matrix
    camera_q_w = camera_poses["pose.pose.orientation.w"].to_numpy()
    camera_q_x = camera_poses["pose.pose.orientation.x"].to_numpy()
    camera_q_y = camera_poses["pose.pose.orientation.y"].to_numpy()
    camera_q_z = camera_poses["pose.pose.orientation.z"].to_numpy()
    camera_q = np.vstack([camera_q_w, camera_q_x, camera_q_y, camera_q_z]).T  # Camera rotation in quaternions
    camera_rot = np.apply_along_axis(quat2rot, 1, camera_q)  # Camera rotationmatrix

    camera_x = camera_poses["pose.pose.position.x"].to_numpy()
    camera_y = camera_poses["pose.pose.position.y"].to_numpy()
    camera_z = camera_poses["pose.pose.position.z"].to_numpy()
    camera_position = np.vstack([camera_x, camera_y, camera_z]).T  # Camera position
    camera_trans = camera_position + marker_offset

    return camera_rot, camera_q, camera_trans


def calculate_relative_transformation(q_c, q_O, P_c, P_o):
    """
    Calculate relative transformation between Camera and Object
    Input: Rotation of  Camera and Object given in quaternions : q_O, q_c 
           Position of Camera and Object: t_c and t_o 
    Output: return Bounding Box [x_c,y_c, w,h] 
    """
    q_cw = np.array([[q_c[0], q_c[1], q_c[2], q_c[3]],
                     [-q_c[1], q_c[0], q_c[3], -q_c[2]],
                     [-q_c[2], -q_c[3], q_c[0], q_c[1]],
                     [-q_c[3], q_c[2], -q_c[1], q_c[0]]])
    q_cw = q_cw.squeeze(2)
    q_O = q_O.reshape(4, 1)
    q_co = np.matmul(q_cw, q_O)
    R = quat2rot(q_c).squeeze(2)
    P_co_w = P_o - P_c
    P_co_w = P_co_w.reshape(3, 1)
    P = np.matmul(R.T, P_co_w)
    return q_co, P


def calc_bounding_box(x_coord, y_coord):
    """
    Calculate Bounding Box (x_c,y_c,w,h)
    Input: x,y Coordantes of binary mask
    Output: return Bounding Box [x_c,y_c, w,h] 
    """
    return np.rint(np.array([[(np.min(x_coord) + np.max(x_coord)) / 2, (np.min(y_coord) + np.max(y_coord)) / 2],
                             [np.max(x_coord) - np.min(x_coord), np.max(y_coord) - np.min(y_coord)]]))


def xywh2xyxy(bbox):
    x1 = int(bbox[0][0] - bbox[1][0] / 2)
    x2 = int(bbox[0][0] + bbox[1][0] / 2)
    y1 = int(bbox[0][1] - bbox[1][1] / 2)
    y2 = int(bbox[0][1] + bbox[1][1] / 2)
    return [x1, y1, x2, y2]


def xywh2coco(bbox):
    x1 = int(bbox[0][0] - bbox[1][0] / 2)
    y1 = int(bbox[0][1] - bbox[1][1] / 2)
    w = bbox[1][0]
    h = bbox[1][1]
    return [x1, y1, w, h]

def calc_area(x_coord, y_coord):
    return (np.max(x_coord) - np.min(x_coord)) * (np.max(y_coord) - np.min(y_coord))


def bigger_bbox(pix_u, pix_v, image_size):
    pix_u_min = np.amin(pix_u)
    pix_u_max = np.amax(pix_u)
    pix_u_down = pix_u_min - int((pix_u_max - pix_u_min) * 0.05)
    pix_u_upp = pix_u_max + int((pix_u_max - pix_u_min) * 0.05)
    pix_v_min = np.amin(pix_v)
    pix_v_max = np.amax(pix_v)
    pix_v_down = pix_v_min - int((pix_v_max - pix_v_min) * 0.05)
    pix_v_upp = pix_v_max + int((pix_v_max - pix_v_min) * 0.05)
    pix_u_big = np.array([pix_u_down, pix_u_upp]).T
    pix_v_big = np.array([pix_v_down, pix_v_upp]).T
    pix_u_big, pix_v_big = pixel_filter(pix_u_big, pix_v_big, image_size)
    pix_u = np.hstack((pix_u, pix_u_big))
    pix_v = np.hstack((pix_v, pix_v_big))
    bbox = calc_bounding_box(pix_u, pix_v)
    return bbox


def create_extrinsic_matrix(quat, trafo, B_C):
    """
    Calculate The extrinsic Matrix in order to tranform pointcloud into Camera Coordinates. 
    Input: 
        quat: rotation rigid body word frame
        trafo: translation rigid body word frame
        B_C: Rotation and translation matrix from Rigid Body to Camera Frame
    Output: Extrinsic Matrix
    """
    R_bc = B_C[:3, :3]
    t_bc = B_C[:3, 3:]
    # R_bc=np.matmul(rot_z(180),rot_x(90))
    # t_bc=np.array([[0.0013],[0.048],[-0.027]])
    # determine camera-rigid-body rotation 
    Rb_rot = quat2rot(quat).squeeze(2)
    # translate camera-rigid-body frame to match camera frame
    # camera_rot = np.matmul(Rb_rot,rot_z(180))
    # camera_rot=np.matmul(camera_rot,rot_x(90))
    camera_rot = np.matmul(Rb_rot, R_bc)
    trafo = trafo.T + np.matmul(Rb_rot, t_bc)
    # Calculate the extrinsics 
    extrinsics = np.hstack((camera_rot.T, -np.matmul(camera_rot.T, trafo)))

    extrinsics = np.vstack((extrinsics, np.array([0, 0, 0, 1])))
    return extrinsics


def calc_iou(boxA, boxB):
    """
    Calculate Intersection over Union between Masks, In order to decide if Mask should be kept or discarded. 
    Input: 
        boxA: bbox with discarded pixels outside Image 
        boxB: Original bbox 
    Output: Intersection over Union between 
    """
    # determine the (x, y)-coordinates of the intersection rectangle
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])
    # compute the area of intersection rectangle
    interArea = max(0, xB - xA + 1) * max(0, yB - yA + 1)
    # compute the area of both the prediction and ground-truth
    # rectangles
    boxAArea = (boxA[2] - boxA[0] + 1) * (boxA[3] - boxA[1] + 1)
    boxBArea = (boxB[2] - boxB[0] + 1) * (boxB[3] - boxB[1] + 1)
    # compute the intersection over union by taking the intersection
    # area and dividing it by the sum of prediction + ground-truth
    # areas - the interesection area
    iou = interArea / float(boxAArea + boxBArea - interArea)
    # return the intersection over union value
    return iou


def pixel_filter(pix_u, pix_v, image_size):
    # Filter pixel outside the image
    for i, u in enumerate(pix_u):
        if u < 0:
            pix_u[i] = 0
        elif u >= image_size[1]:
            pix_u[i] = image_size[1] - 1
    for i, v in enumerate(pix_v):
        if v < 0:
            pix_v[i] = 0
        elif v >= image_size[0]:
            pix_v[i] = image_size[0] - 1  # If after filttering there is no points left --> return none
    return pix_u, pix_v


def calc_mask(pix_u, pix_v, image_size, iou_thresh=0.5):
    """
    Function to remove Pixels outside Image Size.     
    """
    pix_u = np.array(pix_u, dtype=int)
    pix_v = np.array(pix_v, dtype=int)

    bbox_orig = calc_bounding_box(pix_u, pix_v)
    bbox_orig = xywh2xyxy(bbox_orig)
    pix_u, pix_v = pixel_filter(pix_u, pix_v, image_size)

    bbox = calc_bounding_box(pix_u, pix_v)
    bbox_xyxy = xywh2xyxy(bbox)
    iou = calc_iou(bbox_xyxy, bbox_orig)
    # If iou is above the threshold then keep the bbox/mask/annotation otherwise return None for everything
    if iou < iou_thresh:
        return None, None, None
    # If iou is above the threshold then keep the bbox/mask/annotation otherwise return None for everything
    area = calc_area(pix_u, pix_v)
    # Check if in range of image size
    mask = np.zeros(image_size)
    points = np.vstack([pix_v, pix_u]).T
    mask[points[:, 0], points[:, 1]] = 1
    bbox = bigger_bbox(pix_u, pix_v, image_size)
    return bbox, mask, area


def rot_x(degree):
    rad = np.deg2rad(degree)
    return np.array([[1, 0, 0], [0, np.cos(rad), -np.sin(rad)], [0, np.sin(rad), np.cos(rad)]])


def rot_y(degree):
    rad = np.deg2rad(degree)
    return np.array([[np.cos(rad), 0, np.sin(rad)], [0, 1, 0], [-np.sin(rad), 0, np.cos(rad)]])


def rot_z(degree):
    rad = np.deg2rad(degree)
    return np.array([[np.cos(rad), -np.sin(rad), 0], [np.sin(rad), np.cos(rad), 0], [0, 0, 1]])


def create_mask(camera_intrinsics, point_cloud, extrinsic):
    """
    funcion transform Pointcloud in CAmera Coordinates and then into Image Plane
    Input: 
            Camera_intrinsics: Intrinsic Parameterts
            point_cloud: Point Cloud
            extrinsic: Extrinsic Matrix
    Output: Projected Object into Image Plane, 
            bbox: Bounding Box
            Mask: Binary Mask 
            area: Bounding Box area
    """
    intrinsics_matrix = camera_intrinsics["intrinsics"]
    intrinsics_matrix = np.vstack((intrinsics_matrix, np.array([0, 0, 0])))
    intrinsics_matrix = np.hstack((intrinsics_matrix, np.array([[0], [0], [0], [1]])))
    extrinsics_tf = np.matmul(intrinsics_matrix, extrinsic)
    p = np.matmul(extrinsics_tf, point_cloud)
    u_i = p[0, :] / p[2, :]
    v_i = p[1, :] / p[2, :]
    bbox, mask, area = calc_mask(u_i, v_i, camera_intrinsics["image_size"])
    if mask is None:
        return None, None, None
    mask = scipy.ndimage.binary_fill_holes(mask).astype('uint8')
    kernel = np.ones((9, 9), np.uint8)
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    return mask, bbox, area


def project_marker(camera_intrinsics, marker, extrinsic):
    intrinsics_matrix = camera_intrinsics["intrinsics"]
    intrinsics_matrix = np.vstack((intrinsics_matrix, np.array([0, 0, 0])))
    intrinsics_matrix = np.hstack((intrinsics_matrix, np.array([[0], [0], [0], [1]])))
    extrinsics_tf = np.matmul(intrinsics_matrix, extrinsic)
    marker = np.array([[marker[0, 0], marker[0, 1], marker[0, 2], 1]]).T
    p = np.matmul(extrinsics_tf, marker)
    u_i = p[0, :] / p[2, :]
    v_i = p[1, :] / p[2, :]
    return


def binary_mask_to_rle(binary_mask):
    """
    Encode binary mask as run length in order to store it in Json File. 
    Input: binary Mask
    Output: run length encoded MAsk
    """
    binary_mask = binary_mask[:, :]
    rle = {'counts': [], 'size': list(binary_mask.shape)}
    counts = rle.get('counts')
    for i, (value, elements) in enumerate(groupby(binary_mask.ravel(order='C'))):
        if i == 0 and value == 1:
            counts.append(0)
        counts.append(len(list(elements)))
    return rle


def center_pointcloud(pc, offset):
    """
        Align the Pointcloud with the object coordinate frame.
        The point cloud frame is defined on the bottom, shift the origin of the frame to the marker position.
        Input: 
            pc: Pointcloud (x,y,z,1) in homogeneous coordinates 
            offset: Offset from bottom to marker position.
        Output: 
            pc: rotated pc and aligned with object coordinate system (x,y,z,1) 
    """

    r_x = rot_x(90)  # rotation around 90° in x direction to align swapp the y and z coordinates.
    r_x = np.hstack((r_x, np.array([[0, 0, 0]]).T))
    r_x = np.vstack([r_x, [0, 0, 0, 1]])
    pc = np.matmul(r_x, pc)
    pc[2, :] = pc[2, :] - offset[1]  # subtracte offset, to align coordinate frame with the bottom of the marker.

    # pc[0, :] = -pc[0, :]  # swap x direction of pc's to align the x-axis of pc with the

    p_rot_z = rot_z(
        180)  # Rotate along the z-axis by x ° ,depending on the rotation of the pc relative to coordinate frame
    p_rot_z = np.hstack((p_rot_z, np.array([[0, 0, 0]]).T))
    p_rot_z = np.vstack([p_rot_z, [0, 0, 0, 1]])
    pc = np.matmul(p_rot_z, pc)
    return pc


def create_coco_style_data(images, plc_pths, length1, args):
    """
        Function to create .json File 
        Input: 
            images: Image Path
            plc_pths: PC-paths
            length1: length path, excluding object name
            args: Camera bag file & Object bag file.
        Output: 
                Write .json File. 
    """
    images_dic = []
    annotation = []
    camera_recording, object_poses = load_data(args.camera_path, args.object_path)
    annotation_id = 0
    # Loop over all Objects

    for plc_path in plc_pths:
        posistion_name = plc_path.find('.ply')
        pxx = load_pointclouds(plc_path)['points']['x'].to_numpy()
        pyy = load_pointclouds(plc_path)['points']['y'].to_numpy()
        pzz = load_pointclouds(plc_path)['points']['z'].to_numpy()
        p = np.vstack([pxx, pyy, pzz, np.ones(len(pxx))])  # Poincloud in Homogeneous Coordinates
        # offset = offset_dict[
        #    plc_path[length1["plc_pths"] + 1:posistion_name]]  # Offset from bottom of Object to Marker base
        # p = center_pointcloud(p, offset)  # Transfer Pointcloud to Object Coordinate System
        topic = '/' + plc_path[length1[
                                   "plc_pths"] + 0:posistion_name] + '_rb/vrpn_client/estimated_odometry'  # Topic for Object Position
        data = pd.read_csv(object_poses.message_by_topic(topic))
        rotation_o, translation_o, quat = center_objects(
            data)  # Calculate Object position, assuming that object is not moving.
        # Transform Pointcloud in world Coordinates.
        p_rot_z = np.hstack((rotation_o, np.array([[0, 0, 0]]).T))
        p_rot_z = np.vstack([p_rot_z, [0, 0, 0, 1]])
        p = np.matmul(p_rot_z, p)  # Pointcloud in Word Coordinates
        p[0:3, :] = p[0:3, :] + translation_o.T  # Shift pointcloud to align it's position in World Coordinates
        model_pos = translation_o.flatten().tolist()
        position_o = {'position': model_pos, 'quaternion': quat, 'rotation': rotation_o} #m added rotation matrix # Object 6DoF Position to write in Json File
        _, camera_q, camera_trans = center_camera(camera_recording)  # Get Camera 6 DoF Position
        # For each Object go over all images.
        Matrix_Rb_Cam = load_rb_cam(args.Rb_cam)
        for j, name in enumerate(images):
            relative_q, t_c_o = calculate_relative_transformation(camera_q[j:j + 1].T, np.array(quat),
                                                                  camera_trans[j:j + 1],
                                                                  translation_o)  # Calculate 6DoF postion relative to Camera
            rel_position = t_c_o.flatten().tolist()
            relative_quat = relative_q.flatten().tolist()
            var = int(name[name.find("0"):name.find(".png")])  # Get Image Name
            camera_intrinsics = load_camera_intrinsics(args.camera_yaml)  # Load Intrinsics from yaml File
            extrinsic = create_extrinsic_matrix(camera_q[j:j + 1].T, camera_trans[j:j + 1],
                                                Matrix_Rb_Cam)  # calculate extrinsic matrix
            project_marker(camera_intrinsics, translation_o, extrinsic)
            mask, bbox, area = create_mask(camera_intrinsics, p,
                                           extrinsic)  # Calculate Mask , Bbox and Area of object in Image
            if mask is None:
                print("Obj not annotated")
                continue
            pose_cam = {'position': camera_trans[j:j + 1].flatten().tolist(),
                        'quaternion': camera_q[j:j + 1].flatten().tolist()}
            relative_pose = {'position': rel_position, 'quaternion': relative_quat}
            mask = binary_mask_to_rle(mask)  # encode Mask to write in Json format
            bbox = xywh2coco(bbox)
            annotation_dict = {"image_id": var, 'category_id': obj2idx(plc_path[length1["plc_pths"] + 1:posistion_name]),
                               'id': annotation_id, "object_pose": position_o, "camera_pose": pose_cam,
                               "relative_pose": relative_pose, "iscrowd": 0, "segmentation": mask,
                               "bbox": bbox, 'area': np.float(area)}
            annotation.append(annotation_dict)
            camera_intrinsic = camera_intrinsics["intrinsics"].tolist()
            image_dict = {"file_name": "%08i.png" % var, "id": var, "width": camera_intrinsics["image_width"],
                          "height": camera_intrinsics["image_height"],
                          'camera_intrinsics': camera_intrinsic}
            images_dic.append(image_dict)
            annotation_id += 1
        data = {'images': images_dic, 'annotations': annotation}
    with open(os.path.join(args.annotation_path, 'data.json'), 'w') as outfile:
        json.dump(data, outfile)

    return


def save_images_from_bag(camera_path, image_topic, intrinsics_path, output_path):
    """
    Read bag file containing Images and save images in Output_Path
    Input:
        camera_path: location of bag file containing Images.
        Image_topic: Topic where Images are saved in bagfile.
        output_Path: Location of Path where to save images.
    Output: Save Images in output_path.
    """
    with open(intrinsics_path) as f:
        camera_intrinsics = yaml.load(f, Loader=yaml.FullLoader)
    camera_matrix = np.array(camera_intrinsics['camera_matrix']['data']).reshape(3, 3)
    dist_coefficients = np.array([camera_intrinsics["distortion_coefficients"]["data"]])
    image_size = (camera_intrinsics["image_width"], camera_intrinsics["image_height"])
    mapx, mapy = cv2.initUndistortRectifyMap(camera_matrix, dist_coefficients, None, camera_matrix,
                                             image_size, 5)
    bag = rosbag.Bag(camera_path, "r")
    bridge = CvBridge()
    count = 1
    lossless_compression_factor = 9
    if not os.path.isdir(os.path.join(output_path)):
        os.mkdir(os.path.join(output_path))
    for topic, msg, t in bag.read_messages(topics=[image_topic]):
        cv_img = bridge.imgmsg_to_cv2(msg, desired_encoding="passthrough")
        # cv_img = cv2.rotate(cv_img, cv2.ROTATE_180)
        cv_img = cv2.remap(cv_img, mapx, mapy, cv2.INTER_LINEAR)
        cv2.imwrite(os.path.join(output_path, "%06i.png" % count), cv2.cvtColor(cv_img, cv2.COLOR_RGB2BGR),
                    [cv2.IMWRITE_PNG_COMPRESSION, lossless_compression_factor])
        count += 1
    bag.close()
    return


def process_rosbag(args):
    """
    Process Rosbag
    """
    # Check whether the output folder exists
    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)
        os.makedirs(os.path.join(args.output_dir, 'images/'))
        os.makedirs(os.path.join(args.output_dir, 'annotations/'))
    args.image_path = os.path.join(args.output_dir, 'images/')
    if not os.path.exists(args.image_path):
        os.makedirs(args.image_path)
    args.annotation_path = os.path.join(args.output_dir, 'annotations/')
    if not os.path.exists(args.annotation_path):
        os.makedirs(args.annotation_path)

    if not args.images_extracted:
        start = time.time()
        save_images_from_bag(args.camera_path, args.image_topic, args.camera_yaml, args.image_path)
        images = sorted(glob(os.path.join(args.image_path, '*.png')))
        end = time.time()
        print("{} images extracted in {} seconds".format(len(images), end - start))

    images = sorted(glob(os.path.join(args.image_path, '*.png')))
    plc_pths = sorted(glob(os.path.join(args.pc_path, '*.ply')))
    length = {}
    length["plc_pths"] = len(args.pc_path)
    length["images"] = len(images)
    start = time.time()
    create_coco_style_data(images, plc_pths, length, args)
    end = time.time()
    print("{} images annotated in {} seconds".format(len(images), end - start))
    return


if __name__ == "__main__":
    path = os.path.dirname(os.path.abspath(__file__))
    parser = argparse.ArgumentParser()
    parser.add_argument("--camera_path", type=str, default=os.path.join(path,
                                                                        'MUKISANO_recordings/18_7_21_fullres_frames/third_attempt/CAMERA_fullset_closest_sync.bag'),
                        help="Path to the rosbag to be processed")
    parser.add_argument("--pc_path", type=str, help="Path to the directory containing point clouds",
                        default=os.path.join(path, 'MUKISANO_recordings/PCs_filtered'))
    parser.add_argument("--object_path", type=str, help="Path to the rosbag containing the objects' poses",
                        default=os.path.join(path,
                                             'MUKISANO_recordings/18_7_21_fullres_frames/third_attempt/ANIMAL_fullset.bag'))
    parser.add_argument("--camera_yaml", type=str, help="Path to the yaml-file containing the camera intrinsics",
                        default='intrinsics/calib_file_fullres_drop.yaml')
    parser.add_argument("--Rb_cam", type=str,
                        help="Path to the yaml-file containing the camera Rigid Body Transformaiton matrix",
                        default='transformation_min_rotated_cam_new.yaml')
    parser.add_argument("--output_dir", default=os.path.join(path, 'rgb'), help="Output directory for recorded images.")
    parser.add_argument("--images_extracted", type=bool, default=False,
                        help="Set to False if images need to be extracted, otherwise set to True to not extract them again")
    parser.add_argument("--image_topic", default="/synchronizer/rgb_undistort", help="Image topic.")
    args = parser.parse_args()
    process_rosbag(args)
