import argparse
import sys
import os
from pathlib import Path

from lib.write_pose_files import write_2D_position, write_3D_pose
from lib.base import init_output_path, load_json, get_img_ids_from_arguments, get_subdir_paths

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
        images/1...N/file_name
        annotations/
            annotations/1...N/object_pose/position[3] (absolute position of an object in a predefined coordinate system)
            annotations/1...N/object_pose/quaterion[4] (object pose quaternion)
            annotations/1...N/object_pose/rotation[9] (object pose rotation matrix)
            annotations/1...N/camera_pose/position[3] (absolute position of the camera for a particular image/frame in a predefined coordinate system)
            annotations/1...N/camera_pose/quaterion[4] (camera pose quaternion)
            annotations/1...N/camera_pose/rotation[9] (camera pose rotation matrix)
            annotations/1...N/bbox[4] (bbox of the annotated object of the 2D image plane in COCO format)
    :Saves: the following files to a new directory `rsg/` next to the json file specified by `--annotation-data-path`
         - 0000000Image_ID_3D-pose-cam.txt: (1 file per image/frame, rows #1) the complete pose of the camera (XYZ, rotation matrix, quaternion) for a given image/frame
         - 0000000Image_2D-pos-objects.cl: (1 file per image/frame, rows: #N annotations) the object/category ID + center point (XY) of all objects visible in the image/frame with id Image_ID
         - 3D-pose-objects.enh: (1 file per recording, rows: rows: #N tracked objects) the object/category ID + complete pose of all tracked objects (static objects 'digital twin')
'''

def transform():
    recordings_path, annotation_json_name, image_ids = opt.recordings_path, opt.annotation_json_name, opt.image_ids
    
    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)


    for recording_path in recording_paths: 
        # init (file)paths for current recording
        annotation_data_fpath = str(Path(recording_path) / 'output' / 'annotations' / Path(annotation_json_name).with_suffix('')) + '.json'
        try: # annotation_data file path check
            if not os.path.isfile(annotation_data_fpath):
                raise AttributeError('Annotation file not found in recording data.')
        except Exception as e:
                print('Exception: {}'.format(str(e)), file=sys.stderr)
                print('File not found (%s)' % annotation_data_fpath)
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
    
    print('Done writing position/pose data for all recordings (%s)' % recording_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes COCO data format annotation json file from several recordings and extracts image-specific camera and object position and pose data and saves information in ASCII format')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/annotations/*.json ... OptiTrack_recordings/N/output/annotations/*.json
                        containing a COCO data format annotation `.json` file specified by `--annotation-json-name`
                        ''')
    parser.add_argument('--annotation-json-name', type=str, required=True, help='Json annotation filename (e.g., `data.bag` or just `data`)')
    parser.add_argument('--image-ids', nargs='*', # nargs: creates a list; 0 or more values expected
                        help='''
                        (optional) ID list of images for which COCO data is extracted and transformed for RSG (e.g., `--image_ids 2 4 8 16 32`);
                        if not passed as an argument data for every image is transformed (assumes consecutive numbering starting with ID 1)
                        ''')
    opt = parser.parse_args()
    print(opt)

    transform()