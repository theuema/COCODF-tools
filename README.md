# COCO data format plot annotation and detection tools

Tools to plot bounding boxes from COCO data format annotation data and perform object detection on images

***
## Files
***

<br>

### `rotate_coco_180.py`
***Rotates all images for 180 degrees and projects annotation data points of annotation/image pairs in COCO data format to fit the rotated image dataset***

- :Reads COCO data format dataset: from `--coco_path` with the following file structure:

```
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json
```

- :Rotates the bounding boxes for all images as well as the corresponding annotation/data.json: and stores the new rotated dataset to `output_path`


<br>

### `plot_annotations_coco.py` 
***Uses annotation/image pairs in COCO data format to plot bounding boxes and generates annotated images***

- :Reads COCO data format dataset: from `--coco_path` with the following file structure:

```
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json
```

- :Plots bounding boxes from annotation/image pairs: specified by `--image_ids` and saves the results to `--output_path`
- :Plots bounding boxes from annotation/image pairs: specified by `--show_ids` and directly shows the results
 - `--image_ids` and `--show_ids` correspond to the images present in images/ sorted by filename
- :Segmentation mask is visualized: by passing `--segmentation`

<br>

### `detect_coco.py` 
***Runs object detection on images of COCO dataset to plot resulting bounding boxes and generates annotated images***

- :Reads COCO data format dataset: from `--coco_path` with the following file structure: 

```
    ./coco_path
        ./coco_path/images/
            ./coco_path/images/*.png
        ./coco_path/annotations/
            ./coco_path/annotations/data.json
```

- :Runs object detection on images and plots resulting bounding boxes to images: specified by `--image_ids` and saves the results to `--output_path`
- :Runs object detection on images and plots resulting bounding boxes to images: specified by `--show_ids` and directly shows the results
 - `--image_ids` and `--show_ids` correspond to the images present in images/ sorted by filename
- :YOLO implementation taken from: https://github.com/WongKinYiu/ScaledYOLOv4

<br>

### `compare_plot_annotations_coco_scaledyolov4.py` 
***Compares manually created annotations with annotations from object detection and gives a score aboute their divergence.***

- **Info using this script:** There is no need to manually pre-annotate and detect all the data with their corresponding scripts `plot_annotations_coco.py` and `detect_coco.py`
    - `compare_plot_annotations_coco_scaledyolov4.py` combines all process steps and annotates and/or detects if needed given its `image_ids`

<br>

### `*_multrec.py` 
***Multrec variants of all scripts extend their functionality to process multiple recordings.***
***E.g., several images from simple image capturing sessions, image frames of several video recording sessions*** 
***Recording folder must come with the following intended file structure:***

```
    ./recordings_path
        ./recordings_path/1/coco_output/images/
            ./recordings_path/1/coco_output/images/*.png
        ./recordings_path/1/coco_output/annotations/
            ./recordings_path/1/coco_output/annotations/data.json
```

- Output paths mostly fixed to avoid passing too many arguments.


<br>

### `compare_plot_annotations_coco_scaledyolov4_multrec.py`
***Multirecording variant of `compare_plot_annotations_coco_scaledyolov4.py`***
***Compares manually created annotations with annotations from object detection and gives a score aboute their divergence.***

- **Info using this script:** There is no need to manually pre-annotate and detect all the data with their corresponding scripts `plot_annotations_coco_multrec.py` and `detect_coco_multrec.py`
    - `compare_plot_annotations_coco_scaledyolov4_multrec.py` combines all process steps and annotates and/or detects if needed given its `image_ids`

***
## Setup

### Docker *(recommended)*

#### Manual
[//]: # (#TODO: Merge all following steps with ScaledYOLOv4 submodule)

1. **Get COCO data format plot annotation and detection tools**

```bash
cd && \
mkdir ML && cd ML && \
git clone git@github.com:theuema/COCODF-tools.git && \
cd COCODF-tools/
```

2. **Set up ScaledYOLOv4**

*Set up ScaledYOLOv4 using weights: yolov4-p7.pt*
```bash
# cd ~/ML/COCODF-tools/
git clone https://github.com/WongKinYiu/ScaledYOLOv4.git && git checkout yolov4-large && \
cd ScaledYOLOv4/ && \
mkdir -p weights && cd weights && \
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=18fGlzgEJTkUEiBG4hW00pyedJKNnYLP3" -O yolov4-p7.pt && rm -rf /tmp/cookies.txt && \
cd
```

For more information on ScaledYOLOv4, visit: https://github.com/WongKinYiu/ScaledYOLOv4

3. **Create symlinks for imports**

```bash
# cd ~/ML/COCODF-tools/
ln -s ScaledYOLOv4/models models && \
ln -s ScaledYOLOv4/utils utils
```

3. **Install nvidia-docker and start container**

- Install nvidia-docker following: https://github.com/NVIDIA/nvidia-docker
- Start container with the following command (*Change `USER` to your username*):

```bash
nvidia-docker run --gpus all --name pytorch_21.02-py3_Ubuntu-20.04_yolov4_csp -it -v /home/USER/ML:/ML/ --shm-size=64g nvcr.io/nvidia/pytorch:21.02-py3
```

- Check if PyTorch & OpenCV setup works in the container

```bash
python --version && \
python -c "import torch; print(torch.__version__)" && \
python -c "import cv2; print(cv2.__version__)"
```

4. **Install mish-cuda inside container**
- *Info*: if you use different pytorch version, try https://github.com/thomasbrandon/mish-cuda

```bash
cd / && \
git clone https://github.com/JunnYu/mish-cuda && \
cd mish-cuda && \
python setup.py build install
```

5. **Use scripts**

#### Dockerfile *(not complete and currently untested)*

**Usage**

```bash
# points to the location the docker file (docker file naming: pytorch_21.02-py3_ros-noetic-desktop-Ubuntu-20.04_yolov4_csp)
docker build -t pytorch_21.02-py3_ros-noetic-desktop-Ubuntu-20.04_yolov4_csp
```

```dockerfile
FROM nvcr.io/nvidia/pytorch:21.02-py3
LABEL "about"="pytorch:21.02-py3 with ros-noetic-desktop"

# Configure tzdata
ENV TZ=Europe/Vienna
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 'libgl1' is due to 'libGL.so.1' import error when using cv2 from Python
# 'software-properties-common' for 'add-apt-repository' command during ROS installation
# 'lsb-release' for 'lsb_release -sc' command during ROS installation
RUN apt update && apt install -y \
    libgl1 \
    software-properties-common \
    lsb-release 

# install ros-noetic-desktop
RUN add-apt-repository "deb http://archive.ubuntu.com/ubuntu $(lsb_release -sc) main universe restricted multiverse" && \
    sh -c 'echo "deb http://packages.ros.org/ros/ubuntu $(lsb_release -sc) main" > /etc/apt/sources.list.d/ros-latest.list' && \
    curl -s https://raw.githubusercontent.com/ros/rosdistro/master/ros.asc | apt-key add - && \
    apt update && \
    apt install ros-noetic-desktop && \
    echo "source /opt/ros/noetic/setup.bash" >> $HOME/.bashrc && \
    source ~/.bashrc
```