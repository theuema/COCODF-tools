import sys
import argparse
import os
from numpy import random

from lib.base import init_output_path, get_detected_image_paths, get_subdir_paths_sorted_recordings, get_cl_fpaths
from lib.image_plot_gtruth_projections import image_plot_gtruth_projections

'''
    :Reads dataset with multiple COCO data format recordings: from `--recordings_path` with the following filestructure: 
    ./recordings_path
        ./recordings_path/1/coco_output/detected_images/
            ./recordings_path/1/coco_output/detected_images/*.png
        ./recordings_path/1/coco_output/rsg/cl_gtruh
            ./recordings_path/1/coco_output/rsg/cl_gtruh/*.cl
            
    :Plots ground truth projections from corresponding .cl file to a detected image: 
        and saves the results to a new folder `gtruth_proj_images/` in the corresponding ./recordings_path/N/output/ folder
'''
def plot():
    recordings_path, cl_files_path = opt.recordings_path, opt.cl_files_path

    if not os.path.isdir(cl_files_path):
        print('Error: cl directory not found (%s)' % cl_files_path)
        sys.exit(1)

    # get all recording paths
    recording_paths = get_subdir_paths_sorted_recordings(recordings_path)
    cl_files_fpaths = get_cl_fpaths(cl_files_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)
    if not len(cl_files_fpaths):
        print('Error: No ground overwrites to copy found in cl overwrites path (%s)' % cl_files_path)
        sys.exit(1)

    if len(recording_paths) is not len(cl_files_fpaths):
        print('Error: len of recordings paths and cl overwrites fpaths do not match')
        sys.exit(1)

    # plot recording wise
    for rec_i, recording_path in enumerate(recording_paths):   
        # init filepaths for current recording
        coco_path = os.path.join(recording_path, 'output')
        all_detected_img_paths = get_detected_image_paths(coco_path)

        images_output_path = os.path.join(coco_path, 'manual_gtruth_images')
        init_output_path(images_output_path)

        # plot ground truth projections to all images of recording `i`

        for detected_img_path in all_detected_img_paths:
            # corresponding .cl file through sorted list of img and cl paths (see get_detected_image_paths & get_cl_fpaths)
            id_cl_path = cl_files_fpaths[rec_i]

            # get projected coordinates
            f = open(id_cl_path, 'r')
            perfect_proj_dicts = []
            for line in f.readlines()[1:]:
                coord = line.rstrip().split(';')
                # continue at manually defined 'GCP' points
                if 'GCP' in coord[0]:
                    continue
                perfect_proj_dicts.append({'X': round(float(coord[1])), 'Y': round(float(coord[2])), 'ID': coord[0]})

            # X, Y, clabels = ([] for i in range(3))
            # for line in f.readlines()[1:]:
            #     coord = line.rstrip().split(' ')
            #     X.append(float(coord[1]))
            #     Y.append(float(coord[2]))
            #     clabels.append(coord[0])

            # gen colors for #-annotations
            colors = [[random.randint(0, 255) 
                for _ in range(3)] for _ in range(len(perfect_proj_dicts))]

            image_plot_gtruth_projections(detected_img_path, perfect_proj_dicts, colors, out=images_output_path)

            print('Plotted circle at perfect projection for recording #%s)' % rec_i)

    print('Done plotting for all %s recordings.' % len(recording_paths))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Plots ground truth projections from corresponding .cl file to image')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/ ... OptiTrack_recordings/N/output/ 
                        containing COCO data format annotation/image pairs (annotations/data.json and images/ folder)
                        ''')
    parser.add_argument('--cl-files-path', type=str, required=True, 
                        help='''path to manual gtruth cl files (sorted by filename) to plot within recordings
                        ''')
    opt = parser.parse_args()
    print(opt)

    plot()