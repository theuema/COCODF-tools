import argparse
import sys
import os
from pathlib import Path

from lib.base import init_output_path, extract_rosbag_images, load_camera_intrinsics

'''
    :Extracts images from a given ROSbag: specified by `--image-bag-path` and `--image-bag-topic`
    :Performs undistort of images using camera intrinsics: specified by `--camera-yaml` and `--perform-undistort`
    :Saves extracted images to an output directory: specified by `--output-dir`
'''
def process():
    output_path, image_bag_fpath, image_bag_topic, undistort, camera_yaml = \
        opt.output_path, opt.image_bag_path, opt.image_bag_topic, opt.undistort, opt.camera_yaml

    try: # undistort arguments check
        if undistort and camera_yaml is None:
            raise AttributeError('Need camera intrinsics `--camera-yaml` to `--undistort` images.')
        if not os.path.exists(image_bag_fpath):
            raise AttributeError('File given to `--image-bag-path` does not exist.')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    # Init (f)paths
    output_path = str(Path(output_path) / 'images') if undistort else str(Path(output_path) / 'distorted_images')
    init_output_path(output_path)

    # Process ROSbag
    intrinsics = load_camera_intrinsics(camera_yaml)
    extract_rosbag_images(image_bag_fpath, image_bag_topic, output_path, undistort, intrinsics)
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Current processing includes the extraction of images from a given ROSbag')
    parser.add_argument('--image-bag-path', type=str, required=True, help='Path to the ROSbag from which to extract images from')
    parser.add_argument('--image-bag-topic', type=str, required=True, help='Topic name in ROSbag containing recorded images')
    #parser.add_argument('--extract-images', action='store_true', help='Extract images')
    parser.add_argument('--undistort', action='store_true', help='Use camera intrisics to undistort extracted images')
    #parser.add_argument('--object-path', type=str, help='Path to the ROSbag containing the objects' poses')
    parser.add_argument('--camera-yaml', type=str, help='Path to the yaml-file containing the camera intrinsics')
    #parser.add_argument('--rb-cam', type=str, help='Path to the yaml-file containing the camera Rigid Body Transformaiton matrix')
    parser.add_argument('--output-path', type=str, required=True, help='Output directory for extracted images (creates folder `images` or `distorted_images`)')

    opt = parser.parse_args()
    print(opt)
    
    process()