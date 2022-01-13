import os
from pathlib import Path
from lib.base import get_image_annotation_object_center, quat2rot

def write_cl(save_path: str, object_center_2D_dicts: list): # save_path must be initialized
    # takes a list of object center dictionaries ({'category_id': category_id, 'object_center_2D': (c, l), 'image_fname': image_fname})
    # sort the list according to category_ids and write/append data as cl-file(s)
    # output is one file for each image_fname containing category_id; object_center_points; comments sorted by ID for use with RSG module

    cl_enc = 'Id;C;L;Status;SigImg\n'
    object_center_2D_dicts = sorted(object_center_2D_dicts, key=lambda d: d['category_id']) # sort list of dictionaries by dictionary value 'category_id' 
    for center_dict in object_center_2D_dicts:
            cl_path = os.path.join(save_path, Path(center_dict['image_fname']).stem + '.cl')
            write_enc = True if not os.path.isfile(cl_path) else False
            with open(cl_path, 'a') as f:
                if write_enc:
                    f.write(cl_enc)
                # write category_id (aka. object_id) and 2D center point
                f.write('%i; ' % center_dict['category_id'])
                # write object center
                rounded_center = tuple([round(x, 2) for x in center_dict['object_center_2D']])
                f.write(('%.2f; %.2f; ') % rounded_center)
                # write RSG comments
                f.write('UM-; 1.000000\n')

def write_enh(save_path: str, object_positions_3D_dict: dict):
    # takes a dictionary of 3D_object_positions {'category_id': [x, y, z]}
    # sort the dictionary by its key 'category_id'
    # save one file for each recording containing category_id; X; Y; Z; comments sorted by ID for use with RSG module

    enh_enc = 'Id;E;N;H;SigE;SigN;SigH;Status;Comment\n'
    sorted_category_id = sorted(object_positions_3D_dict)
    enh_path = os.path.join(save_path, 'GCPs' + '.enh')
    with open(enh_path, 'w') as f:
        f.write(enh_enc)
        for category_id in sorted_category_id:
            # write category_id (aka. object_id) and 3D position
            f.write('%i; ' % category_id)
            # write 3D_object_position
            rounded_pos = tuple([round(x, 4) for x in object_positions_3D_dict[category_id]])
            f.write(('%.4f; %.4f; %.4f; ') % rounded_pos)
            # write RSG comments
            f.write('0.0000000001; 0.0000000001; 0.0000000001; UM-; 0.040 1.0 0.0 0.0 0.0 1.0 0.0\n')

def write_txt(save_path: str, txt_C_camera_pose_dict: dict):
    # takes a dictionary of 3D_camera_positions: {'image_id': {'position': t_wc, 'rotation': R_wc, 'image_fname': image_fname}}
    # sort the dictionary by its key 'image_id'
    # save one file for all images in a recording sorted by ID to compare against RSG calc

    txt_enc = 'ImageId;X;Y;Z;Rm00;Rm01;Rm02;Rm10;Rm11;Rm12;Rm20;Rm21;Rm22;ImageName\n' # "Rm" denotes 3x3 "Rotation matrix Rm01 = row 0 column 1"
    sorted_image_id = sorted(txt_C_camera_pose_dict)
    txt_path = os.path.join(save_path, 'CPOSs' + '.txt')
    with open(txt_path, 'w') as f:
        f.write(txt_enc)
        for image_id in sorted_image_id:
            # write image_id
            f.write('%i; ' % image_id)
            # write 3D position X;Y;Z
            t_wc = txt_C_camera_pose_dict[image_id]['position']
            rounded_t_wc = [round(x, 4) for x in t_wc.T.flatten().tolist()]
            for coordinate in rounded_t_wc:
                f.write('%.4f; ' % coordinate)
            # write rotation matrix RM
            R_wc = txt_C_camera_pose_dict[image_id]['rotation']
            for row in R_wc:
                rounded_row = [round(x, 4) for x in row]
                for rm_element in rounded_row:
                    f.write('%.4f; ' % rm_element)
            # write image_fname
            f.write(txt_C_camera_pose_dict[image_id]['image_fname'] + '\n')
