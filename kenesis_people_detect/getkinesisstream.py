import boto3
import cv2
from ultralytics import YOLO
import supervision as sv


ACCESS_KEY = "AKIATAWAYJAU33F4TVFF"
SECRET_KEY = "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"
STREAM_NAME = "BKS_Dev_Floor"
STREAM_ARN = "arn:aws:kinesisvideo:ap-southeast-1:207636416553:stream/BKS_Dev_Floor/1688639835099"
AWS_REGION = "ap-southeast-1"


def main():
    model = YOLO('yolov8s.pt')

    kv_client = boto3.client("kinesisvideo", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, region_name=AWS_REGION)
    endpoint = kv_client.get_data_endpoint(
        StreamName=STREAM_NAME,
        APIName="GET_HLS_STREAMING_SESSION_URL"
    )['DataEndpoint']

    print(endpoint)

    # Grab the HLS Stream URL from the endpoint
    kvam_client = boto3.client("kinesis-video-archived-media", aws_access_key_id=ACCESS_KEY, aws_secret_access_key=SECRET_KEY, endpoint_url=endpoint, region_name=AWS_REGION)
    url = kvam_client.get_hls_streaming_session_url(
        StreamName=STREAM_NAME,
        PlaybackMode="LIVE"
    )['HLSStreamingSessionURL']

    video = cv2.VideoCapture(url)
    
    codec = cv2.VideoWriter_fourcc(*'XVID')
    fps =int(video.get(cv2.CAP_PROP_FPS))
    cap_width, cap_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))

    save_dir = "staffmove_timed.mp4"
    output = cv2.VideoWriter(save_dir, codec, fps, (cap_width, cap_height), True)   

    results = model.track(source=url, classes=0, device='0',
                          tracker="bytetrack.yaml", stream=True, conf=0.5)

    box_annotator = sv.BoxAnnotator()
    for result in results:

        frame = result.orig_img
        h, w = frame.shape[:2]

        #get the above detections automatically from sv.Detections
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            # pam's fix: detections.tracker_id = np.asarray(result.boxes.id).astype(int) 
            detections.tracker_id=result.boxes.id.cpu().numpy().astype(int)


        #list comprehension to return labels 
        labels = [
            f"#{track_id} {model.model.names[class_id]} {conf:0.2f}" 
            for _, _, conf, class_id, track_id
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        cv2.imshow('track', frame)
        output.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break
    
    output.release()
    video.release()
    cv2.destroyAllWindows

if __name__ == '__main__':
    main()