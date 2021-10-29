import sys
import argparse
import os
import json
from pathlib import Path

from lib.base import load_json, calc_add_rmatrix
'''
    :Takes COCO data format annotation json file specified by `--annotation-data-path`
    :Works with the following json structure:
        images/
        annotations/
            annotations[N]/object_pose/quaterion[4]
            annotations[N]/camera_pose/quaterion[4]
            annotations[N]/relative_pose/quaterion[4]

    :Saves the rotation added annotation data as a new file in the same directory
    :Overwrites the annotation data specified by `--annotation-data-path` with the rotation added annotation data when `--add-rotation-matrix` is passed

    :Resulting rotation data in json structure:
    annotations/
            annotations[N]/object_pose/rotation[3]
                annotations[N]/object_pose/rotation[0]/[3]
                    annotations[N]/object_pose/rotation[0]/[0]/rmatrix_row1_column1_value
                    annotations[N]/object_pose/rotation[0]/[1]/rmatrix_row1_column2_value
                    annotations[N]/object_pose/rotation[0]/[2]/rmatrix_row1_column3_value
                annotations[N]/object_pose/rotation[1]/[3]
                    annotations[N]/object_pose/rotation[1]/[0]/rmatrix_row2_column1_value
                    annotations[N]/object_pose/rotation[1]/[1]/rmatrix_row2_column2_value
                    annotations[N]/object_pose/rotation[1]/[2]/rmatrix_row2_column3_value
                ...
            annotations[N]/camera_pose/rotation[3]
            annotations[N]/relative_pose/rotation[3]
'''
def store():
    # calculate and add rotation matrix to existing data or save a new file including the rotation matrix
    annotation_data_fpath, in_place = opt.annotation_data_path, opt.add_rotation_matrix

    try: # annotation_data file path check
        if not os.path.isfile(annotation_data_fpath):
            raise AttributeError('File given to `--annotation-data-path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    coco_annotation_data = load_json(annotation_data_fpath)
    rmatrix_coco_annotation_data = calc_add_rmatrix(coco_annotation_data)

    # store rotation matrix included annotation data
    if in_place:
        with open(annotation_data_fpath, 'w') as outfile:
            json.dump(rmatrix_coco_annotation_data, outfile)
        print('Added rotation matrix to existing annotation file (%s)' % annotation_data_fpath)
    else: 
        json_path = str(Path(annotation_data_fpath).parent / 'rmatrix_') + str(Path(annotation_data_fpath).name)
        with open(json_path, 'w') as outfile:
            json.dump(rmatrix_coco_annotation_data, outfile)
        print('New annotation file containing quaterions and rotation matrix for poses saved (%s)' % json_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes quaternions stored in a COCO data format annotation file to calculate the rotation matrix, add to the annotation data and save a new file')
    parser.add_argument('--annotation-data-path', type=str, required=True,
                        help='(File)path to the COCO data format annotation `*.json` file, containing `object_pose`, `camera_pose` and `relative_pose` defined by quaternions')
    parser.add_argument('--add-rotation-matrix', action='store_true', help='add rotation matrix to existing `*.json` file given by `--annotation-data-path`')
    opt = parser.parse_args()
    print(opt)

    store()