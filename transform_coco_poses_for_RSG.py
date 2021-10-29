import argparse
import sys
import os
from pathlib import Path

from lib.write_pose_files import write_2D_position, write_3D_pose
from lib.base import init_output_path, load_json, get_img_ids_from_arguments

def transform():
    annotation_data_fpath, image_ids = opt.annotation_data_path, opt.image_ids
    
    # init (file)paths
    try: # annotation_data file path check
        if not os.path.isfile(annotation_data_fpath):
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
    image_names = []
    # go through all annotations and store separate files for each image example
    for annotation in coco_annotation_data['annotations']: 
        # get corresponding image
        image = next(image for image in image_dicts if image['id'] == annotation['image_id'])
        if image is None:
            print('Error: Image ID not found in images (%s)', annotation['image_id'])
            sys.exit(1)
        image_name = str(Path(image['file_name']).stem)

        # create image specific data (non-static: 3D camera position, 2D object coordinates)
        if annotation['image_id'] in image_ids: # len(annotations) times each image
            # write/append 2D object coordinates of their center (.CL file)
            cl_path = os.path.join(save_path, image_name + '_2D-pos-objects' + '.cl')
            write_2D_position(cl_path, annotation['bbox'], mode='a', category_id=annotation['category_id'])
        
            if image_name not in image_names: # only 1x each image (= 3D camera pose for one frame)
                # write/append 3D camera position, rotation matrix and quaternions for image (text file)
                txt_path = os.path.join(save_path, image_name + '_3D-pose-cam' + '.txt')
                write_3D_pose(txt_path, annotation['camera_pose'], mode='a')
                image_names.append(image_name)

        # write/append static data (static: 3D object coordinates for the physical model that is not moved during the recording process)
        if annotation['category_id'] not in category_ids: # len(objects) times (number of objects/categories from the dataset)
            # write 3D object position, rotation matrix and quaternions from a (physically) static model (.ENH file)
            enh_path = os.path.join(save_path, '3D-pose-objects' + '.enh')
            write_3D_pose(enh_path, annotation['object_pose'], mode='a', category_id=annotation['category_id'])
            category_ids.append(annotation['category_id'])

    print('Done writing position/pose data for RSG (%s)' % save_path)  #f.write(('%g ' * 5 + '\n') % (cls, *xywh))
    print('Transformed for images ', image_names)


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