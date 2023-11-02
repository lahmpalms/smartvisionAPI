from ultralytics import YOLO
import cv2
import os


import os

def get_latest_output_image():
    # Define the directory where YOLOv8 saves the output images
    output_directory = 'runs/detect'

    # List all directories (each corresponding to a run)
    run_directories = os.listdir(output_directory)

    # Filter for directories that start with 'predict' (assuming this pattern)
    predict_directories = [d for d in run_directories if d.startswith('predict')]

    if not predict_directories:
        return None  # No 'predictX' directories found

    # Sort the predict directories based on the numeric part and get the latest
    latest_predict_directory = max(predict_directories, key=lambda d: int(d[len('predict'):]) if d[len('predict'):] else -1)

    # Construct the path to the latest output image file (assuming it's named 'output.jpg')
    latest_output_image = os.path.join(output_directory, latest_predict_directory, 'input_image.jpg')

    return latest_output_image

# Usage
latest_image_path = get_latest_output_image()

if latest_image_path:
    print("Latest Output Image Path:", latest_image_path)
else:
    print("No 'predictX' directories found.")




def main(input_image_path):
    # Load the YOLOv8 model
    model = YOLO('yolov8s.pt')

    # Perform YOLOv8 object detection
    results = model.predict(input_image_path, save=True)

    # Save the processed image to an output file
    latest_image_path = get_latest_output_image()

    return latest_image_path

if __name__ == '__main__':
    main("/home/nont/entech/shutterstock_721660498.jpg")
