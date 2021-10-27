import sys
import argparse
import os
import json
from pathlib import Path

from lib.base import load_json, add_rmatrix

def calc_create_new(coco_annotation_data):
    # adds a rotation matrix to `coco_annotation_data` and saves a separate (duplicate + rotation matrix) file
    annotation_data_fpath = opt.annotation_data_path
    coco_annotation_data_rotM = add_rmatrix(coco_annotation_data)

    json_path = str(Path(annotation_data_fpath).parent / Path(annotation_data_fpath).stem) + '_rotM' + Path(annotation_data_fpath).suffix
    with open(json_path, 'w') as outfile:
        json.dump(coco_annotation_data_rotM, outfile)
    print('New annotation file containing quaterions and rotation matrix for poses saved (%s)' % json_path)


def calc_add_existing(coco_annotation_data):
    # adds a rotation matrix to `coco_annotation_data`
    annotation_data_fpath = opt.annotation_data_path
    coco_annotation_data_rotM = add_rmatrix(coco_annotation_data)

    with open(annotation_data_fpath, 'w') as outfile:
        json.dump(coco_annotation_data_rotM, outfile)
    print('Added rotation matrix to existing annotation file (%s)' % annotation_data_fpath)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Uses quaternions stored in a COCO data format annotation file to calculate the rotation matrix, add to the annoation data and save a new file')
    parser.add_argument('--annotation-data-path', type=str, required=True,
                        help='(File)path to the COCO data format annotation `*.json` file, containing `object_pose`, `camera_pose` and `relative_pose` defined by quaternions')
    parser.add_argument('--add-rotation-matrix', action='store_true', help='add rotation matrix to existing `*.json` file given by `--annotation-data-path`')
    opt = parser.parse_args()
    print(opt)

    try:
        if not os.path.exists(opt.annotation_data_path):
            raise AttributeError('File given to `--annotation-data-path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    coco_annotation_data = load_json(opt.annotation_data_path)
    
    # calculate and add rotation matrix to existing data or save a new file including the rotation matrix
    if opt.add_rotation_matrix:
        calc_add_existing(coco_annotation_data) 
    else: 
        calc_create_new(coco_annotation_data)