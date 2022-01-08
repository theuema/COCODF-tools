import argparse
from re import I
import sys
import os
from pathlib import Path

from lib.base import init_output_path, extract_rosbag_images, load_camera_intrinsics, get_subdir_paths

'''
    Function works with the following filestructure: 
    ./recordings_path
        ./recordings_path/N/
            ./recordings_path/1/image-bag-path.bag
        ...

    :Extracts images from a given ROSbag: specified by `--recordings_path`, `--image-bag-name` and `--image-bag-topic`
    :Performs undistort of images using camera intrinsics: specified by `--camera-yaml` and `--distorted'
    :Saves extracted images to an output directory: specified by `--output-dir`
'''
def process():
    output_path, recordings_path, image_bag_topic, image_bag_name, distorted, camera_yaml = \
        opt.output_path, opt.recordings_path, opt.image_bag_topic, opt.image_bag_name, opt.distorted, opt.camera_yaml

    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    try: # distorted arguments check
        if distorted and camera_yaml is None:
            raise AttributeError('Need camera intrinsics `--camera-yaml` to `--distorted` images.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)


    for recording_path in recording_paths: 
        # init (file)paths for current recording
        image_bag_fpath = str(Path(recording_path / Path(image_bag_name).with_suffix(''))) + '.bag'
        try: 
            if not os.path.isfile(image_bag_fpath):
                raise AttributeError('Filename given to `--image-bag-name` does not exist.')
        except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            print('Missing file (%s)' % image_bag_fpath)
            sys.exit(1)

        # Init (f)paths
        output_path_distorted = str(Path(output_path) / Path(recording_path).name / 'output' / 'distorted_images') 
        output_path_undistorted = str(Path(output_path) / Path(recording_path).name / 'output' / 'images')
        init_output_path(output_path_distorted)
        init_output_path(output_path_undistorted)

        # Process ROSbag
        intrinsics = load_camera_intrinsics(camera_yaml)
        if distorted:
            # undistort and store images
            extract_rosbag_images(image_bag_fpath, image_bag_topic, output_path_undistorted, True, intrinsics)
            # store distorted images
            extract_rosbag_images(image_bag_fpath, image_bag_topic, output_path_distorted, False, intrinsics)
        else:
            # store undistorted images
            extract_rosbag_images(image_bag_fpath, image_bag_topic, output_path_undistorted, False, intrinsics)
            # distort and store distorted images
            extract_rosbag_images(image_bag_fpath, image_bag_topic, output_path_distorted, True, intrinsics)

    print('Done extracting images from ROSbag for multiple recordings (%s)' % (output_path + '/N' + '/output'))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Current processing includes the extraction of images from a given ROSbag')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/image-bag-name.bag ... OptiTrack_recordings/N/image-bag-name.bag
                        bag file name specified by `--image-bag-name`
                        ''')
    parser.add_argument('--image-bag-name', type=str, required=True, help='ROSbag filename (e.g., `image_recording.bag` or just `image_recording`) containing recorded images')
    parser.add_argument('--image-bag-topic', type=str, required=True, help='Topic name in ROSbag containing recorded images')
    #parser.add_argument('--extract-images', action='store_true', help='Extract images')
    parser.add_argument('--distorted', action='store_true', help='Specifies if images in rosbag are distorted images')
    #parser.add_argument('--object-path', type=str, help='Path to the ROSbag containing the objects' poses')
    parser.add_argument('--camera-yaml', type=str, help='Path to the yaml-file containing the camera intrinsics')
    #parser.add_argument('--rb-cam', type=str, help='Path to the yaml-file containing the camera Rigid Body Transformaiton matrix')
    parser.add_argument('--output-path', type=str, required=True, help='Dataset output folder for extracted images (creates folder `images` or `distorted_images`)')

    opt = parser.parse_args()
    print(opt)
    
    process()