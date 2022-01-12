import os
from lib.base import get_image_annotation_object_center, quat2rot

def write_cl(save_path: str, object_center_2D_dicts: list):
    # takes a list of object center dictionaries ({'category_id': category_id, 'object_center_2D': (c, l), 'image_name': image_name})
    # sort the list according to category_ids and write/append data as cl-file(s)
    # output is one file for each image_name containing category_id; object_center_points; comments sorted by ID for use with RSG module

    cl_enc = 'Id;C;L;Status;SigImg\n'
    object_center_2D_dicts = sorted(object_center_2D_dicts, key=lambda d: d['category_id']) # sort list of dictionaries by dictionary value 'category_id' 
    for center_dict in object_center_2D_dicts:
            cl_path = os.path.join(save_path, center_dict['image_name'] + '.cl')
            write_enc = True if not os.path.isfile(cl_path) else False
            with open(cl_path, 'a') as f:
                if write_enc:
                    f.write(cl_enc)
                # write category_id (aka. object_id) and 2D center point
                f.write('%s; ' % center_dict['category_id'])
                f.write(('%s; %s; ') % center_dict['object_center_2D'])
                f.write('UM-; 1.000000\n')

def write_enh(save_path: str, objects_pos_3D_dict: dict):
    # takes a dictionary of 3D_object_positions {'category_id': [x, y, z]}
    # sort the dictionary by its key 'category_id'
    # save one file for each recording containing category_id; X; Y; Z; comments sorted by ID for use with RSG module

    enh_enc = 'Id;E;N;H;SigE;SigN;SigH;Status;Comment\n'
    objects_pos_3D_dict = sorted(objects_pos_3D_dict)
    enh_path = os.path.join(save_path, 'GCPs' + '.enh')
    with open(enh_path, 'w') as f:
        f.write(enh_enc)
        for category_id in objects_pos_3D_dict:
            # write category_id (aka. object_id) and 3D position
            f.write('%s; ' % category_id)
            pos = objects_pos_3D_dict[category_id] 
            f.write(('%s; %s; %s; ') % (pos[0], pos[1], pos[2]))
            f.write('0.0000000001; 0.0000000001; 0.0000000001; UM-; 0.040 1.0 0.0 0.0 0.0 1.0 0.0\n')

def write_txt(save_path: str, image_name: str, R_wc, t_wc):
    txt_enc = 'X;Y;Z;Rm00;Rm01;Rm02;Rm10;Rm11;Rm12;Rm20;Rm21;Rm22\n' # "Rm" denotes 3x3 "Rotation matrix Rm01 = row 0 column 1"
    txt_path = os.path.join(save_path, image_name + '.txt')
    with open(txt_path, 'w') as f:
        f.write(txt_enc)
        # write 3D position X;Y;Z
        for coordinate in t_wc.T.flatten().tolist():
            f.write('%s; ' % coordinate)
        # write rotation matrix RM
        for row in R_wc:
            for rm_element in row:
                f.write('%s; ' % rm_element)

'''
:write/append 2D object coordinates of a bbox center:
:encoding:  category_id COLUMN LINE
'''
def write_2D_position(save_fpath: str, bbox: list, mode: str, category_id: int):
    # appends center point information to a file specified by `save_fpath`
    write_enc = False if os.path.isfile(save_fpath) else True

    # write encoding header
    with open(save_fpath, mode) as f:
        if write_enc:
            # write encoding information
            f.write('category_id X Y' + '\n')
        
        center = get_image_annotation_object_center(bbox)
        # write category_id (aka. object_id) and center point
        f.write('%s ' % category_id)
        f.write(('%s %s ' + '\n') % center)

'''
:write/append 3D camera position, rotation matrix and quaternions:
:pose: a dict containing 'position' (XYZ) and 'quaternion' (c xs ys zs)
:encoding: [category_id] X Y Z rm00 rm01 rm02 rm10 rm11 rm12 rm20 rm21 rm22 c xs ys zs
:where: [category_id] is optional
        "rm01" denotes a 3x3 "rotation matrix row 0 column 1"
        "c xs ys zs" denotes the corresponding quaternion
'''
def write_3D_pose(save_fpath: str, pose: dict, mode: str, enc: str = None, category_id: int = None):
    # appends 3D position and pose information to a file specified by `save_fpath`
    write_enc = True if enc is not None else False
    write_id = False if category_id is None else True

    # write encoding header
    with open(save_fpath, mode) as f:
        if write_enc:
            # write encoding information
            f.write('category_id ' + enc + '\n') if write_id else f.write(enc + '\n')
        
        # write category_id (aka. object_id)
        if write_id:
            f.write('%s ' % category_id)

        # write 3D position
        _3D_poss = pose['position']
        for coordinate in _3D_poss:
            f.write('%s ' % coordinate)
        
        # write rotation matrix
        rmatrix_rows = quat2rot(pose['quaternion'])
        # rmatrix_rows = pose['rotation'] #don't directly use rotation to also handle data w/o added rotation matrix
        for row in rmatrix_rows:
            for element in row:
                f.write('%s ' % element)
        
        # write corresponding quaternions
        pose_quats = pose['quaternion']
        for quaternion in pose_quats:
            f.write('%s ' % quaternion)

        f.write('\n')

'''
:write/append 3D position to a file specified by `fpath`
    writes encoding line if `enc` is passed
    writes the an `category_id` before the position of passed 
'''
def write_3D_position(fpath: str, pos: list, mode: str, enc: str = None, category_id: int = None, linebreak: bool = False):
    write_enc = True if enc is not None else False
    write_id = False if category_id is None else True

    with open(fpath, mode) as f:
        if write_enc:
            # write encoding information
            f.write('category_id ' + enc + '\n') if write_id else f.write(enc + '\n')
        
        # write category_id (aka. object_id)
        if write_id:
            f.write('%s ' % category_id)

        # write 3D position
        for coordinate in pos:
            f.write('%s ' % coordinate)
        
        if linebreak:
            f.write('\n')

'''
:write/append 3D orientation to a file specified by `fpath`
'''
def write_3D_orientation_rot(fpath: str, rotM: list, mode: str, linebreak: bool = False): #rotM = [[r00, r01, r02],[r10, r11, r12],[r20, r21, r22]]
    with open(fpath, mode) as f:
        # write rotation matrix
        for row in rotM:
            for element in row:
                f.write('%s ' % element)
        if linebreak:
            f.write('\n')