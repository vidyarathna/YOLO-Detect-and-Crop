# YOLO Detect and Crop

This script uses YOLO (You Only Look Once) object detection to identify objects in an image and save each detected object as a separate image file.

## Prerequisites

Before running the script, ensure you have the following installed:

- Python 3.x
- OpenCV (cv2)
- YOLO v3 model files (`yolov3.weights`, `yolov3.cfg`, and `coco.names`)

## Installation

Clone the repository:

```sh
git clone https://github.com/vidyarathna/YOLO-Detect-and-Crop.git
cd YOLO-Detect-and-Crop
```

Install dependencies:

```sh
pip install -r requirements.txt
```

## Usage

To detect objects in an image using YOLO and save them as separate images, run the script `yolo_detect_and_crop.py` with the following command:

```sh
python yolo_detect_and_crop.py -i path/to/your/input/image.jpg -y path/to/yolo-coco -o path/to/output/directory
```

### Arguments

- `-i`, `--image`: Path to the input image.
- `-y`, `--yolo`: Base path to the YOLO directory containing `coco.names`, `yolov3.weights`, and `yolov3.cfg`.
- `-c`, `--confidence`: Minimum probability to filter weak detections (default is 0.5).
- `-t`, `--threshold`: Threshold when applying non-maxima suppression (default is 0.3).
- `-o`, `--output`: Path to the output directory where detected objects will be saved.

### Example

```sh
python yolo_detect_and_crop.py -i ./input_images/living_room.jpg -y ./yolo-coco -o ./output_objects
```

This command will process the `living_room.jpg` image using the YOLO model located in the `./yolo-coco` directory, detect objects, and save each detected object as a separate image in the `./output_objects` directory.

### Script Details

The script performs the following steps:

1. Loads the COCO class labels YOLO was trained on.
2. Loads the YOLO model configuration and weights.
3. Loads the input image and prepares it for detection.
4. Performs object detection using YOLO on the input image.
5. Filters detections based on confidence.
6. Applies non-maxima suppression to remove redundant overlapping boxes.
7. Saves each detected object as a separate image file in the specified output directory.

### Notes

- Ensure the input image is in a format supported by OpenCV.
- Make sure the YOLO directory contains the necessary files (`coco.names`, `yolov3.weights`, `yolov3.cfg`).
- Adjust the confidence and threshold parameters as needed to improve detection accuracy.
