import sys
import argparse
import os
import json
from pathlib import Path

from lib.base import load_json, load_mapping, align_coco_annotation_data, add_coco_style_categories, get_annoation_data_fpath, get_subdir_paths
'''
    Exchanges custom category_ids with official COCO dataset category_ids (e.g., the label "horse" is "17", but is "18" in COCO)
    Please see the corresponding `labels/aau_coco.mapping` to see the deviation

    :Takes COCO data format annotation json file specified by `--annotation-data-path`
    :Works with the following json structure:
        images/
        annotations/
            annotations[N]/category_id

    :Either, saves the annotation data as a new file in the same directory
    :Or overwrites category_id data in-place when `--align-in-place` is passed
'''
def fix():
    # calculate and add rotation matrix to existing data or save a new file including the rotation matrix
    recordings_path, mapping_fpath, in_place, deviate = opt.recordings_path, opt.mapping_path, opt.align_in_place, opt.deviate

    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    try: # needs alignment check
        if deviate and mapping_fpath is None:
            raise AttributeError('Missing `mapping_path` to align a deviated dataset.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    # compare recording wise
    for i, recording_path in enumerate(recording_paths):   

        # init filepaths for current recording
        coco_path = os.path.join(recording_path, 'output')
        annotation_data_fpath = get_annoation_data_fpath(coco_path)

        try: # annotation_data file path check
            if not os.path.isfile(annotation_data_fpath):
                raise AttributeError('File given to `--annotation-data-path` does not exist.')
        except Exception as e:
                print('Exception: {}'.format(str(e)), file=sys.stderr)
                sys.exit(1)

        # load mapping, annotation_data and align accordingly
        label_mapping = load_mapping(mapping_fpath)
        coco_annotation_data = load_json(annotation_data_fpath)
        if deviate: # align category_ids according to a pre-defined mapping scheme
            coco_annotation_data_aligned = align_coco_annotation_data(coco_annotation_data, label_mapping)
            # insert "categories" section to data.json (standard for COCO data format)
            coco_annotation_data_aligned = add_coco_style_categories(coco_annotation_data_aligned)
        else:
            # insert "categories" section to data.json (standard for COCO data format)
            coco_annotation_data_aligned = add_coco_style_categories(coco_annotation_data)

        # save aligned annotation_data 
        if in_place:
            with open(annotation_data_fpath, 'w') as outfile:
                json.dump(coco_annotation_data_aligned, outfile)
        else: 
            json_path = str(Path(annotation_data_fpath).parent / 'aligned_labels_') + str(Path(annotation_data_fpath).name)
            with open(json_path, 'w') as outfile:
                json.dump(coco_annotation_data_aligned, outfile)

        # write modifications.txt
        mod_file_path = str((Path(annotation_data_fpath).parents[1]) / 'modifications.txt')
        
        with open(mod_file_path, 'a') as f:
            if deviate:
                f.write('data.json: aligned category_ids (changed custom category_ids to coco category_ids)\n')
            f.write('data.json: added categories section (coco data format definition)\n')

        if deviate:
            print('Done aligning deviated dataset and inserting categories section for recording #%s (%s)' % (i, annotation_data_fpath))
        else:
            print('Done inserting categories section to dataset for recording #%s (%s)' % (i, annotation_data_fpath))

        print('Modifications file written (%s)' % mod_file_path)

    print('Done for all %s recordings.' % len(recording_paths))

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Adds categories section to a json file and optionally aligns deviating category_ids stored in a json file according to a pre-defined mapping scheme')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/ ... OptiTrack_recordings/N/output/ 
                        containing COCO data format annotation/image pairs (annotations/data.json and images/ folder)
                        ''')
    parser.add_argument('--mapping-path', type=str, help='''
                        path to file containing a mapping `category_id new_category_id category_name` corresponding to the category_ids in `annotations/data.json` separated with space (e.g., labels/aau_coco.mapping)
                        ''')
    parser.add_argument('--deviate', action='store_true', help='indicates if the category_ids stored in annotations/data.json deviate from standard COCO category_ids')
    parser.add_argument('--align-in-place', action='store_true', help='in-place alignment of category labels in `*.json` file given by `--annotation-data-path`')
    opt = parser.parse_args()
    print(opt)

    fix()