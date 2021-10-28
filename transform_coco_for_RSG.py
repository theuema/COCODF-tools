import argparse
import sys
import os
from pathlib import Path

from numpy.core.records import array

from lib.base import init_output_path, load_json, get_img_ids_from_arguments, get_image_annotation_object_center

def append_2D_position_file(save_fpath: str, bbox: list, category_id: int):
    # appends center point information to a file specified by `save_fpath`
    write_enc = True if os.path.exists(save_fpath) else False

    with open(save_fpath, 'a') as f:
        if write_enc:
            # write encoding information
            f.write('category_id1 X Y \n')
        
        center = get_image_annotation_object_center(bbox)

        # write category_id (aka. object_id) and center point
        f.write('%s ' % category_id)
        f.write(('%g ' + '\n') % center)

def append_3D_pose_file(save_fpath: str, pose: dict):
    # appends 3D position and pose information to a file specified by `save_fpath`
    write_enc = True if os.path.exists(save_fpath) else False

    with open(save_fpath, 'a') as f:
        if write_enc:
            # write encoding information
            f.write('X Y Z rm00 rm01 rm02 rm10 rm11 rm12 rm20 rm21 rm22 c xs ys zs \n')
        
        # write 3D position
        _3D_poss = pose['position']
        for coordinate in _3D_poss:
            f.write('%s ' % coordinate)
        
        # write rotation matrix
        pose_rots = pose['rotation']
        for rmelement in pose_rots:
            f.write('%s ' % rmelement)
        
        # write corresponding quaternions
        pose_quats = pose['quaternion']
        for quaternion in pose_quats:
            f.write('%s ' % quaternion)

        f.write('\n')

def transform():
    annotation_data_fpath, image_ids = opt.annotation_data_path, opt.image_ids
    
    # init (file)paths
    try: # annotation_data file path check
        if not os.path.exists(annotation_data_fpath):
            raise AttributeError('File given to `--annotation-data-path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    coco_annotation_data = load_json(annotation_data_fpath)
    save_path = str(Path(annotation_data_fpath).parent / 'rsg')
    init_output_path(save_path)

    # get image_ids for which coco annotation data is transformed to RSG data
    image_ids = get_img_ids_from_arguments(image_ids, len(coco_annotation_data['images']), '--image_ids')
    # get image_dicts from annotation data
    image_dicts = coco_annotation_data['images']

    category_ids = []
    # go through all annotations and store separate files for each image example
    for annotation in coco_annotation_data['annotations']: 
        # get corresponding image
        image = next(image for image in image_dicts if image['id'] == annotation['image_id'])
        if image is None:
            print('Error: Image ID not found in images (%s)', annotation['image_id'])
            sys.exit(1)
        image_name = str(Path(image['file_name']).stem)

        # create image specific data (non-static: 3D camera position, 2D object coordinates)
        if annotation['image_id'] in image_ids:
            # write 3D camera position, rotation matrix and quaternions (text file)
            # encoding: X Y Z rm00 rm01 rm02 rm10 rm11 rm12 rm20 rm21 rm22 c xs ys zs
            # where: "rm01" denotes the a 3x3 "rotation matrix row 0 column 1"
            #        "c xs ys zs" denotes the corresponding quaternion
            txt_path = save_path + image_name + '_camera_3D_pose' + '.txt'
            append_3D_pose_file(txt_path, annotation['camera_pose'])

            # 2D object coordinates of their center (.CL file)
            # encoding: category_id1 COLUMN LINE
            #           category_id2 COLUMN LINE
            cl_path = save_path + image_name + '_object_2D_pos' + '.CL'
            append_2D_position_file(cl_path, annotation['bbox'], annotation['category_id'])
        
        # create static data (static: 3D object coordinates for the physical model that is not moved during the recording process)
        if annotation['category_id'] not in category_ids:
            # 3D object coordinates (.ENH file) from our static model
            # encoding: category_id1 X1 Y2 Z3 rm00 rm01 rm02 rm10 rm11 rm12 rm20 rm21 rm22 c xs ys zs
            #           category_id2 X1 Y2 Z3 rm00 rm01 rm02 rm10 rm11 rm12 rm20 rm21 rm22 c xs ys zs
            save_fpath = save_path + image_name + '_object_3D_pose' + '.ENH'
            append_3D_pose_file(save_fpath, annotation['object_pose'])
            category_ids.append(annotation['category_id'])


    print('Done writing position/pose data for RSG (%s)' % save_path)  #f.write(('%g ' * 5 + '\n') % (cls, *xywh))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes quaternions stored in a COCO data format annotation file to calculate the rotation matrix, add to the annotation data and save a new file')
    parser.add_argument('--annotation-data-path', type=str, required=True,
                        help='(File)path to the COCO data format annotation `*.json` file, containing 3D camera position and 2D/3D object coordinates, quaternions and rotation matrices')
    parser.add_argument('--image-ids', nargs='*', # nargs: creates a list; 0 or more values expected
                        help='''
                        (optional) ID list of images for which COCO data is extracted and transformed for RSG (e.g., `--image_ids 2 4 8 16 32`);
                        if not passed as an argument data for every image is transformed (assumes consecutive numbering starting with ID 1)
                        ''')
    opt = parser.parse_args()
    print(opt)

    transform()