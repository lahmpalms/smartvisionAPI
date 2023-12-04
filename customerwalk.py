from ultralytics import YOLO
import cv2
import supervision as sv
from supervision.detection.utils import clip_boxes
from dataclasses import replace
import numpy as np
import pymongo
import datetime


LINE_START = sv.Point(640, 0)
LINE_END = sv.Point(640,720)

myclient = pymongo.MongoClient("mongodb://admin:islabac123@18.143.76.245:27017/")
mydb = myclient["people_detect_log"]
mycol = mydb["log"]

def main():
    model = YOLO('yolov8m.pt')

    src = "rtsp://admin:cctv12345@10.10.2.194:554/Streaming/Channels/0102"
    video = cv2.VideoCapture(src)
    fourcc = cv2.VideoWriter_fourcc(*'XVID')  # Define the codec (XVID, MJPEG, etc.)
    
    cap_width, cap_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))   
    out = cv2.VideoWriter('output.avi', fourcc, 20.0, (cap_width, cap_height))
    #stream: used for processing long videos or live feed; uses a generator, which only
    #keeps the results of the current frame in memory, significantly reducing memory consumption.
    results = model.track(source=src, classes=0, device='0',
                          tracker="bytetrack.yaml", stream=True, conf=0.5)

    people = {}
    customer = 0

    polygon_coords = np.array([[31, 288], [230, 288], [173, 117], [25, 116]]) 
    polygon_zone = sv.PolygonZone(polygon=polygon_coords, 
                                  frame_resolution_wh=(cap_width,cap_height), 
                                  triggering_position=sv.Position.CENTER)
    polygon_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.green())

    box_annotator = sv.BoxAnnotator()
    processed_ids = set()
    
    for result in results:

        curr_dt = datetime.datetime.now()

        frame = result.orig_img
        h, w = frame.shape[:2]

        #get the above detections automatically from sv.Detections
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            detections.tracker_id=result.boxes.id.cpu().numpy().astype(int)

        #list comprehension to return labels 
        labels = [
            f"#{track_id} {model.model.names[class_id]} {conf:0.2f}" 
            for _, _, conf, class_id, track_id
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        in_zone = polygon_zone.trigger(detections=detections) #returns if person is in the bbox
        polygon_annotator.annotate(scene=frame,label="Shop Zone")

        #GET CENTROID
        clipped_xyxy = clip_boxes(
            boxes_xyxy=detections.xyxy, frame_resolution_wh=(cap_width, cap_height)
        )
        clipped_detections = replace(detections, xyxy=clipped_xyxy)
        clipped_anchors = np.ceil(clipped_detections.get_anchor_coordinates(anchor=sv.Position.CENTER)).astype(int)

        if detections.tracker_id is not None:
            if people == {}:
                for idx, id in enumerate(detections.tracker_id):
                    people[id] = [[], [], []] #position, centroid
                    people[id][0].append(in_zone[idx])
                    people[id][1] = clipped_anchors[idx]
                    
            else: 
                for idx, id in enumerate(detections.tracker_id):
                    if id in people.keys():
                        people[id][0].append(in_zone[idx])
                        people[id][1] = clipped_anchors[idx]
                        if len(people[id][0]) == 30:
                            for i in range(10):
                                people[id][0].pop(0)
                    if id not in people.keys():
                        people[id] = [[], [], []]
                        people[id][0].append(in_zone[idx])
                        people[id][1] = clipped_anchors[idx]

                for id in list(people): #cannot pop while iterating over dictionary
                    if id not in detections.tracker_id: 
                        people.pop(id)
            
            
                for idx, (id, info) in enumerate(people.items()):
                    if len(info[0]) > 15 and all(info[0]) and id not in processed_ids: 
                        staff_id = id
                        customer += 1
                        processed_ids.add(id)
                        # print ("count: ", customer)
                        # print(f"Customer Enter: {staff_id}")
                        mydict = { "status": "Customer Enter", "time": curr_dt, "count": customer}
                        x = mycol.insert_one(mydict)
        
        cv2.putText(frame, f"Count: {customer}", (11, 60), 0, 0.8, [0, 2550, 0], thickness=2, lineType= cv2.LINE_AA)
        cv2.imshow('Customer Detection', frame)

        out.write(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    out.release()
    cv2.destroyAllWindows

if __name__ == '__main__':
    main()
