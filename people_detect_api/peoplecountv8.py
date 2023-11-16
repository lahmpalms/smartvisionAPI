import os
from ultralytics import YOLO
import logging

logging.basicConfig(level=logging.INFO)


def get_latest_output_image():
    output_directory = 'runs/detect'
    run_directories = os.listdir(output_directory)
    predict_directories = [
        d for d in run_directories if d.startswith('predict')]

    if not predict_directories:
        return None  # No 'predict' directories found

    latest_predict_directory = max(predict_directories, key=lambda d: int(
        d[len('predict'):]) if d[len('predict'):] else -1)
    latest_output_image = os.path.join(
        output_directory, latest_predict_directory, 'input_image.jpg')

    return latest_output_image


def main(input_image_path):
    try:
        model = YOLO('yolov8s.pt')

        # Perform YOLOv8 object detection
        results = model.predict(input_image_path, save=True, classes=0)

        logging.info(f"Results saved in: {results[0].save_dir}")

        # Save the processed image to an output file
        latest_image_path = get_latest_output_image()

        data = {
            "results": results[0],
            "latest_image_path": latest_image_path
        }

        if latest_image_path:
            logging.info(f"Latest Output Image Path: {latest_image_path}")
        else:
            logging.warning("No 'predict' directories found.")

        # Delete the input image and the YOLOv8 output directory
        os.remove(input_image_path)

        return data
    except Exception as e:
        logging.error(f"An error occurred: {str(e)}")
        return None
