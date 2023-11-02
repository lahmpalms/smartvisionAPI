from ultralytics import YOLO
import cv2
import supervision as sv
from supervision.detection.utils import clip_boxes
from dataclasses import replace
import numpy as np
import pymongo


LINE_START = sv.Point(640, 0)
LINE_END = sv.Point(640,720)

myclient = pymongo.MongoClient("mongodb://admin:islabac123@18.143.76.245:27017/")
mydb = myclient["people_detect_log"]
mycol = mydb["log"]

def main():
    model = YOLO('yolov8s.pt')

    src = "rtsp://admin:cctv12345@10.10.2.194:554/Streaming/Channels/0102"
    # src = "/Users/macbook/Desktop/code/py/ml/baksters/entech/footage/staffhelpmove.mp4"
    video = cv2.VideoCapture(src)
    
    cap_width, cap_height = int(video.get(cv2.CAP_PROP_FRAME_WIDTH)), int(video.get(cv2.CAP_PROP_FRAME_HEIGHT))   

    #stream: used for processing long videos or live feed; uses a generator, which only
    #keeps the results of the current frame in memory, significantly reducing memory consumption.
    results = model.track(source=src, classes=0, device='0',
                          tracker="bytetrack.yaml", stream=True, conf=0.5)

    staff_id = None 
    people = {}
    leftzone = False
    status = "In Zone"

    #lower corner [left, right], upper corner [right, left]
    # polygon_coords = np.array([[640,720], [1280,720], [1280,0], [640,0]]) for webcam
    # polygon_coords = np.array([[221, 110], [310, 108], [309, 3], [215, 3]]) 
    polygon_coords = np.array([[51, 288], [183, 274], [173, 117], [25, 116]]) 
    polygon_zone = sv.PolygonZone(polygon=polygon_coords, 
                                  frame_resolution_wh=(cap_width,cap_height), 
                                  triggering_position=sv.Position.CENTER)
    polygon_annotator = sv.PolygonZoneAnnotator(zone=polygon_zone, color=sv.Color.green())

    # polygon_coords2 = np.array([[90, 350], [225, 350], [235, 230], [120, 230]])
    polygon_coords2 = np.array([[345, 254] , [439, 254], [442, 102], [356, 96]])
    polygon_zone2 = sv.PolygonZone(polygon=polygon_coords2, 
                                  frame_resolution_wh=(cap_width,cap_height), 
                                  triggering_position=sv.Position.CENTER)
    polygon_annotator2 = sv.PolygonZoneAnnotator(zone=polygon_zone2, color=sv.Color.red())

    box_annotator = sv.BoxAnnotator()
    for result in results:

        frame = result.orig_img
        h, w = frame.shape[:2]
        # print(h,w)

        # frame = result.plot()

        #get the above detections automatically from sv.Detections
        detections = sv.Detections.from_yolov8(result)

        if result.boxes.id is not None:
            # pam's fix: detections.tracker_id = np.asarray(result.boxes.id).astype(int) 
            detections.tracker_id=result.boxes.id.cpu().numpy().astype(int)
            # cv2.putText(frame, f"ids: {detections.tracker_id}", (11, 140), 0, 0.8, [2550, 0, 0], thickness=2, lineType= cv2.LINE_AA)


        #list comprehension to return labels 
        labels = [
            f"#{track_id} {model.model.names[class_id]} {conf:0.2f}" 
            for _, _, conf, class_id, track_id
            in detections
        ]

        frame = box_annotator.annotate(scene=frame, detections=detections, labels=labels)

        in_zone = polygon_zone.trigger(detections=detections) #returns if person is in the bbox
        in_zone2 = polygon_zone2.trigger(detections=detections) #returns if person is in the bbox
        polygon_annotator.annotate(scene=frame,label="staff zone")
        polygon_annotator2.annotate(scene=frame,label="door zone")

        #GET CENTROID
        clipped_xyxy = clip_boxes(
            boxes_xyxy=detections.xyxy, frame_resolution_wh=(cap_width, cap_height)
        )
        clipped_detections = replace(detections, xyxy=clipped_xyxy)
        clipped_anchors = np.ceil(clipped_detections.get_anchor_coordinates(anchor=sv.Position.CENTER)).astype(int)

        # print(clipped_anchors)

        if detections.tracker_id is not None:
            if people == {}:
                for idx, id in enumerate(detections.tracker_id):
                    people[id] = [[], [], []] #position, centroid
                    people[id][0].append(in_zone[idx])
                    people[id][1] = clipped_anchors[idx]
                    people[id][2].append(in_zone2[idx])
            else: 
                for idx, id in enumerate(detections.tracker_id):
                    if id in people.keys():
                        people[id][0].append(in_zone[idx])
                        people[id][1] = clipped_anchors[idx]
                        people[id][2].append(in_zone2[idx])
                        if len(people[id][0]) == 30:
                            for i in range(10):
                                people[id][0].pop(0)
                    if id not in people.keys():
                        people[id] = [[], [], []]
                        people[id][0].append(in_zone[idx])
                        people[id][1] = clipped_anchors[idx]
                        people[id][2].append(in_zone2[idx])
                        

                for id in list(people): #cannot pop while iterating over dictionary
                    if id not in detections.tracker_id: 
                        people.pop(id)

            
            if staff_id is None:
                for idx, (id, info) in enumerate(people.items()):
                    if len(info[0]) > 15 and all(info[0]): 
                        staff_id = id
                        print(f"STAFF FOUND: {staff_id}")
                        mydict = { "status": "Found Staff", "time": "12:34" }

                        x = mycol.insert_one(mydict)

            else:
                for idx, (id, info) in enumerate(people.items()):
                    if id == staff_id:
                        if not all(info[0]):
                            status = "Staff Left Zone"
                            leftzone = True
                            

                            if leftzone:
                                res = False
                                if(info[2].count(True) >= len(info[2])*0.2):
                                    res = True

                                if res:
                                    status = "Staff Left Shop"
                                    leftzone = False
                            
                        cv2.putText(frame, status, (11, 100), 0, 0.8, [0, 50, 2500], thickness=2, lineType= cv2.LINE_AA)

        cv2.putText(frame, f"staff id: {staff_id}", (11, 60), 0, 0.8, [0, 2550, 0], thickness=2, lineType= cv2.LINE_AA)

        cv2.imshow('track', frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows

if __name__ == '__main__':
    main()
