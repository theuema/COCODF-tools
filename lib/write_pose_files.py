import os
from lib.base import get_image_annotation_object_center

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
:encoding: [category_id] X Y Z rm00 rm01 rm02 rm10 rm11 rm12 rm20 rm21 rm22 c xs ys zs
:where: [category_id] is optional
        "rm01" denotes a 3x3 "rotation matrix row 0 column 1"
        "c xs ys zs" denotes the corresponding quaternion
'''
def write_3D_pose(save_fpath: str, pose: dict, mode: str, category_id: int = None):
    # appends 3D position and pose information to a file specified by `save_fpath`
    write_enc = False if os.path.isfile(save_fpath) else True
    write_id = False if category_id is None else True

    # write encoding header
    with open(save_fpath, mode) as f:
        if write_enc:
            # write encoding information
            enc = 'X Y Z rm00 rm01 rm02 rm10 rm11 rm12 rm20 rm21 rm22 c xs ys zs'
            f.write('category_id ' + enc + '\n') if write_id else f.write(enc + '\n')
        
        # write category_id (aka. object_id)
        if write_id:
            f.write('%s ' % category_id)

        # write 3D position
        _3D_poss = pose['position']
        for coordinate in _3D_poss:
            f.write('%s ' % coordinate)
        
        # write rotation matrix
        rmatrix_rows = pose['rotation']
        for row in rmatrix_rows:
            for element in row:
                f.write('%s ' % element)
        
        # write corresponding quaternions
        pose_quats = pose['quaternion']
        for quaternion in pose_quats:
            f.write('%s ' % quaternion)

        f.write('\n')
