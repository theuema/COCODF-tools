import argparse
import sys
import os
from shutil import copyfile
from pathlib import Path

from lib.base import get_subdir_paths, get_cl_fpaths, init_output_path

def combine():
    recordings_path, output_path, combine_cl_ann, combine_cl_det = opt.recordings_path, opt.output_path, opt.cl_ann, opt.cl_det

    try: # any center points check
        if not combine_cl_ann and not combine_cl_det:
            raise AttributeError('At least pass data from one annotator to combine. (Missing `--cl-ann` or `--cl_det`)')
    except Exception as e:
            print('Exception: {}'.format(str(e)), file=sys.stderr)
            sys.exit(1)

    # get all recording paths
    recording_paths = get_subdir_paths(recordings_path)
    if not len(recording_paths):
        print('Error: No recording directories found (%s)' % recordings_path)
        sys.exit(1)

    init_output_path(output_path)

    for recording_path in recording_paths: 

        # init (file)paths for current recording
        rsg_path = str(Path(recording_path) / 'output' / 'rsg')
        txt_fpath = os.path.join(rsg_path, 'CPOSs.txt')
        enh_fpath = os.path.join(rsg_path, 'GCPs.enh') 
        cl_ann_path = os.path.join(rsg_path, 'cl_ann')
        cl_det_path = os.path.join(rsg_path, 'cl_det')
        
        try: # 3D files exist check
            if not os.path.isfile(txt_fpath):
                raise AttributeError('3D camera position file missing (%s)', txt_fpath)
            if not os.path.isfile(enh_fpath): # detection_data file path check
                raise AttributeError('3D object position file missing (%s)', enh_fpath)
            if combine_cl_ann and not os.path.exists(cl_ann_path):
               raise AttributeError('Missing folder for annotator annotations (%s)', cl_ann_path) 
            if combine_cl_det and not os.path.exists(cl_det_path):
               raise AttributeError('Missing folder for object detector annotations (%s)', cl_det_path) 
        except Exception as e:
                print('Exception: {}'.format(str(e)), file=sys.stderr)
                sys.exit(1)

        if combine_cl_ann:
            # get annotator center point fpaths
            cl_ann_fpaths = get_cl_fpaths(cl_ann_path)
            if not len(cl_ann_fpaths):
                print('Error: No .cl files found (%s)' % cl_ann_path)
                sys.exit(1)
            # init save_path
            save_path = os.path.join(output_path, 'cl_ann_multirec')
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # combine name & copy
            for cl_ann_fpath in cl_ann_fpaths:
                save_fpath = os.path.join(save_path, 'REC' + str(Path(recording_path).stem) + '_' + str(Path(cl_ann_fpath).name))
                copyfile(cl_ann_fpath, save_fpath)

        if combine_cl_det:
            # get object detector center point fpaths
            cl_det_fpaths = get_cl_fpaths(cl_det_path)
            if not len(cl_det_fpaths):
                print('Error: No .cl files found (%s)' % cl_det_path)
                sys.exit(1)
            # init save_path
            save_path = os.path.join(output_path, 'cl_det_multirec') 
            if not os.path.exists(save_path):
                os.makedirs(save_path)
            # combine name & copy
            for cl_det_fpath in cl_det_fpaths:
                save_fpath = os.path.join(save_path, 'REC' + str(Path(recording_path).stem) + '_' + str(Path(cl_det_fpath).name))
                copyfile(cl_det_fpath, save_fpath)

        # copy .txt of recordings into custom one file for the case of only being able to use one image per recording (AAU Musisano project OptiTrack_recordings_jul21*)
        save_fpath = os.path.join(output_path, 'custom' + '_' + str(Path(txt_fpath).name))
        f = open(txt_fpath, 'r')
        pose = f.readlines()[1].replace(" ", "").rstrip().split(';')

        write_enc = True if not os.path.isfile(save_fpath) else False 
        with open(save_fpath, 'a') as f:
            txt_enc = 'ImageId;X;Y;Z;Rm00;Rm01;Rm02;Rm10;Rm11;Rm12;Rm20;Rm21;Rm22;ImageName\n'
            if write_enc:
                f.write(txt_enc)
            f.write(str(Path(recording_path).stem) + '; ')
            for e in pose[1:-1]:
                f.write(e + '; ')
            f.write('%06i.png' % (int(str(Path(recording_path).stem))) + '\n')
        
        # copy .enh files
        save_fpath = os.path.join(output_path, 'REC' + str(Path(recording_path).stem) + '_' + str(Path(enh_fpath).name))
        copyfile(enh_fpath, save_fpath)
    
    print('Done combining multiple recordings for RSG (%s)' % output_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Takes COCO data format annotation json file from several recordings and extracts image-specific camera and object position and pose data and saves information in ASCII format')
    parser.add_argument('--recordings-path', type=str, required=True,
                        help='''
                        path to the dataset directory containing multiple recording folders 1/ ... N/
                        e.g., `--recordings-path ../OptiTrack_recordings` where multiple recording folders exist: 
                        OptiTrack_recordings/1/output/rsg ... OptiTrack_recordings/N/output/rsg
                        the `rsg/` folder containing `GCPs.enh`, `CPOSs.txt` files and the folders `cl_ann/` and/or `cl_det/` with *.cl files
                        ''')
    parser.add_argument('--cl-ann', action='store_true', help='combine center points of the annotator annotated bounding boxes from multiple recordings for RSG')
    parser.add_argument('--cl-det', action='store_true', help='combine center points of the object detector annotated bounding boxes from multiple recordings for RSG')
    parser.add_argument('--output-path', type=str, required=True,
                        help='output path for combined RSG compatible data, generated from multiple recordings')
    opt = parser.parse_args()
    print(opt)

    combine()