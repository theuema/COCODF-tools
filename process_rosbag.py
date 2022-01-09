import argparse
import sys
import os
from pathlib import Path

from lib.base import init_output_path, extract_rosbag_images, load_camera_intrinsics

'''
    :Extracts images from a given ROSbag: specified by `--image-bag-path` and `--image-bag-topic`
    :Performs undistort of images using camera intrinsics: specified by `--camera-yaml` and `--distorted`
    :Saves extracted images to the output directory specified by `--output-dir`
'''
def process():
    output_path, image_bag_name, image_bag_topic, distorted, camera_yaml, recording_path = \
        opt.output_path, opt.image_bag_name, opt.image_bag_topic, opt.distorted, opt.camera_yaml, opt.recording_path

    image_bag_fpath = str(Path(recording_path / Path(image_bag_name).with_suffix(''))) + '.bag'
    try: # ROSbag file exists check
        if not os.path.isfile(image_bag_fpath):
            raise AttributeError('Filename given to `--image-bag-name` does not exist.')
    except Exception as e:
        print('Exception: {}'.format(str(e)), file=sys.stderr)
        print('Missing file (%s)' % image_bag_fpath)
        sys.exit(1)

    try: # distorted arguments check
        if distorted and camera_yaml is None:
            raise AttributeError('Need camera intrinsics `--camera-yaml` to undistort images.')
        if not os.path.isfile(image_bag_fpath):
            raise AttributeError('File given to `--image-bag-path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    # Init (f)paths
    output_path_distorted = str(Path(output_path) / 'output' / 'distorted_images') 
    output_path_undistorted = str(Path(output_path) / 'output' / 'images')
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
        # distort and store images
        extract_rosbag_images(image_bag_fpath, image_bag_topic, output_path_distorted, True, intrinsics)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Current processing includes the extraction of images from a given ROSbag')
    parser.add_argument('--recording-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing the output/ folder and a
                        bag file name specified by `--image-bag-name`
                        ''')
    parser.add_argument('--image-bag-name', type=str, required=True, help='ROSbag filename (e.g., `image_recording.bag` or just `image_recording`) containing recorded images')
    parser.add_argument('--image-bag-topic', type=str, required=True, help='Topic name in ROSbag containing recorded images')
    #parser.add_argument('--extract-images', action='store_true', help='Extract images')
    parser.add_argument('--distorted', action='store_true', help='Specifies if images in rosbag are distorted images')
    #parser.add_argument('--object-path', type=str, help='Path to the ROSbag containing the objects' poses')
    parser.add_argument('--camera-yaml', type=str, help='Path to the yaml-file containing the camera intrinsics')
    #parser.add_argument('--rb-cam', type=str, help='Path to the yaml-file containing the camera Rigid Body Transformaiton matrix')
    parser.add_argument('--output-path', type=str, required=True, help='Output directory for extracted images (creates folder `images` and `distorted_images`)')

    opt = parser.parse_args()
    print(opt)
    
    process()