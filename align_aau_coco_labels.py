import sys
import argparse
import os
import json
from pathlib import Path

from lib.base import load_json, load_mapping, align_coco_annotation_data, add_coco_style_categories, load_custom_labels
'''
    Exchanges custom AAU object labeling with official COCO dataset labels. 
    :Takes COCO data format annotation json file specified by `--annotation-data-path`
    :Works with the following json structure:
        images/
        annotations/
        annotations[N]/category_id

    :Saves the annotation data as a new file in the same directory
    :Overwrites category_id data in-place when `--align-in-place` is passed
'''
def align():
    # calculate and add rotation matrix to existing data or save a new file including the rotation matrix
    annotation_data_fpath, mapping_fpath, in_place = opt.annotation_data_path, opt.mapping_path, opt.align_in_place

    try: # annotation_data file path check
        if not os.path.isfile(annotation_data_fpath):
            raise AttributeError('File given to `--annotation-data-path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)


    # load mapping, annotation_data and align accordingly
    label_mapping = load_mapping(mapping_fpath)
    coco_annotation_data = load_json(annotation_data_fpath)
    coco_annotation_data_aligned = align_coco_annotation_data(coco_annotation_data, label_mapping)
    coco_annotation_data_aligned = add_coco_style_categories(coco_annotation_data_aligned)

    # save aligned annotation_data 
    if in_place:
        with open(annotation_data_fpath, 'w') as outfile:
            json.dump(coco_annotation_data_aligned, outfile)
        print('Aligned category_ids in-place and added categories data section to existing annotation file (%s)' % annotation_data_fpath)
    else: 
        json_path = str(Path(annotation_data_fpath).parent / 'aligned_labels_') + str(Path(annotation_data_fpath).name)
        with open(json_path, 'w') as outfile:
            json.dump(coco_annotation_data_aligned, outfile)
        print('New annotation file containing aligned category_ids and added categories data section saved (%s)' % json_path)

    # write modifications.txt
    mod_file_path = str((Path(annotation_data_fpath).parents[1]) / 'modifications.txt')
    f = open(mod_file_path, 'a')
    f.write('data.json: aligned labeling (changed custom aau to coco labels)\n')
    f.write('data.json: added categories (coco data format definition)\n')
    f.close()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes category_ids stored in a COCO data format annotation file, maps according a predefined mapping and ')
    parser.add_argument('--annotation-data-path', type=str, required=True,
                        help='(File)path to the COCO data format annotation `*.json` file, containing annotations/N/category_id')
    parser.add_argument('--mapping-path', type=str, required=True, help='''
                        path to file containing a mapping `category_id new_category_id category_name` corresponding to the category_ids in `annotations/data.json` separated with space (e.g., labels/aau_coco.mapping)
                        ''')
    parser.add_argument('--align-in-place', action='store_true', help='in-place alignment of category labels in `*.json` file given by `--annotation-data-path`')
    opt = parser.parse_args()
    print(opt)

    align()