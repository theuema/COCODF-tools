import sys
import argparse
import os
import json
from pathlib import Path

from lib.base import load_json, calc_add_rmatrix, get_subdir_paths
'''
    :Takes COCO data format annotation json file from recordings specified by `--recordings-path` and `--annotation-json-name`
    :Works with the following file structure:
    ./recordings_path
        ./recordings_path/N/
            ./recordings_path/1/coco_output/annotations/
                ./recordings_path/1/coco_output/annotations/annotation-json-name.json
        ...

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
    recordings_path, annotation_json_name, in_place = opt.recordings_path, opt.annotation_json_name, opt.add_rotation_matrix

    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    for recording_path in recording_paths: 
        # init (file)paths for current recording
        annotation_data_fpath = str(Path(recording_path) / 'output' / 'annotations' / Path(annotation_json_name).with_suffix('')) + '.json'
        try: # annotation_data file path check
            if not os.path.exists(annotation_data_fpath):
                raise AttributeError('Annotation file not found in recording data.')
        except Exception as e:
                print('Exception: {}'.format(str(e)), file=sys.stderr)
                print('File not found (%s)' % annotation_data_fpath)
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

    print('Done calculating rotation matrices for all recordings (%s)' % recording_path)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes quaternions stored in a COCO data format annotation file to calculate the rotation matrix, add to the annotation data and save a new file')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/annotations/*.json ... OptiTrack_recordings/N/output/annotations/*.json
                        containing a COCO data format annotation `.json` file specified by `--annotation-json-name`
                        ''')
    parser.add_argument('--annotation-json-name', type=str, required=True, help='Json annotation filename (e.g., `data.bag` or just `data`)')
    parser.add_argument('--add-rotation-matrix', action='store_true', help='add rotation matrix to existing `*.json` file found in `recordings-path/output/annotations/annotation-json-name.json` ')
    opt = parser.parse_args()
    print(opt)

    store()