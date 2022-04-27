import sys
import argparse
import os
from numpy import random

from lib.base import init_output_path, get_detected_image_paths, get_id_cl_path, get_id_img_path, get_subdir_paths, get_cl_fpaths
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
    recordings_path = opt.recordings_path

    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    # plot recording wise
    for rec_i, recording_path in enumerate(recording_paths):   
        # init filepaths for current recording
        coco_path = os.path.join(recording_path, 'output')
        all_detected_img_paths = get_detected_image_paths(coco_path)
        all_cl_paths = get_cl_fpaths(os.path.join(coco_path, 'rsg', 'cl_gtruh'))

        if not len(all_detected_img_paths) or not len(all_cl_paths):
            print('Error: No images or ground truth projections to process found in coco path (%s)' % coco_path)
            return

        images_output_path = os.path.join(coco_path, 'proj_gtruth_images')
        init_output_path(images_output_path)

        # plot ground truth projections to all images of recording `i`

        for i, detected_img_path in enumerate(all_detected_img_paths):
            # corresponding .cl file through sorted list of img and cl paths (see get_detected_image_paths & get_cl_fpaths)
            id_cl_path = all_cl_paths[i] 

            # get projected coordinates
            f = open(id_cl_path, 'r')
            perfect_proj_dicts = []
            for line in f.readlines()[1:]:
                coord = line.rstrip().split(';')
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

            # plot circles of perfect projection

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
    opt = parser.parse_args()
    print(opt)

    plot()