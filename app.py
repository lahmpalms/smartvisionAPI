import os
import shutil
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pymongo import MongoClient
# Import your detection function
from people_detect_api.people_detect import main, process_video
from typing import Annotated, Generator
import json
import face_recognition
import cv2
import numpy as np
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# Connect to MongoDB
client = MongoClient("mongodb://admin:islabac123@18.143.76.245:27017")
db = client["people_detect_log"]  # Replace with your actual database name
collection = db["log_cloudprocess"]  # Replace with your actual collection name

# Your existing /detect-people/ route


@app.post("/detect-people/")
async def detect_people(image: UploadFile):
    try:
        # Save the uploaded image file
        file_path = "input_image.jpg"
        with open(file_path, "wb") as f:
            f.write(image.file.read())

        # Call your YOLOv8 detection function asynchronously
        output_image_path = main(file_path)
        coordinate = []
        for index, box in enumerate(output_image_path["results"].boxes):
            data = {
                "people_id": index,
                "people_cor": box.xyxy.tolist(),
                "class_id": box.cls.item(),
                "confidence": box.conf.item()
            }
            coordinate.append(data)

        # Store data in MongoDB
        result = collection.insert_one({
            "image_path": output_image_path["latest_image_path"],
            "data": coordinate
        })

        if result.inserted_id:
            # Convert the output image to base64
            with open(output_image_path["latest_image_path"], "rb") as image_file:
                image_base64 = base64.b64encode(
                    image_file.read()).decode("utf-8")

            # Remove the image file after reading
            os.remove(output_image_path["latest_image_path"])

            return JSONResponse(content={
                "data": {
                    "base64_image": image_base64,
                    "coordinate_data": coordinate
                },
                "code": 200,
                "message": "ok",
                "mongodb_inserted_id": str(result.inserted_id)
            })
        else:
            return JSONResponse(content={"error": "Failed to insert data into MongoDB."})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# Your existing /video-detectpeople route


@app.post("/video-detectpeople")
async def get_video_detect_people(video: Annotated[UploadFile, File(description="A video read as UploadFile")]):
    try:
        if not video.content_type == 'video/mp4':
            raise HTTPException(
                status_code=400, detail="Invalid file type. Please upload a video file.")
        else:
            file_path = "input_video.mp4"
        with open(file_path, "wb") as f:
            f.write(video.file.read())
        output_image_path = process_video(file_path)
        coordinate = []
        for index, box in enumerate(output_image_path["results"].boxes):
            data = {
                "people_id": index,
                "people_cor": box.xyxy.tolist(),
                "class_id": box.cls.item(),
                "confidence": box.conf.item()
            }
            coordinate.append(data)

        # Store data in MongoDB
        result = collection.insert_one({
            "image_path": output_image_path["latest_image_path"],
            "data": coordinate
        })
        if result.inserted_id:
            def stream_video():
                try:
                    with open(output_image_path["latest_image_path"], "rb") as file_like:
                        chunk = file_like.read()
                        while chunk:
                            yield chunk
                            chunk = file_like.read()
                finally:
                    # Remove the image file after reading
                    os.remove(output_image_path["latest_image_path"])
            return StreamingResponse(stream_video(), media_type="video/mp4")
        else:
            return JSONResponse(content={"error": "Failed to insert data into MongoDB."})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})

# Your existing /getlogclouddata route


@app.get("/getlogclouddata")
def get_data():
    try:
        results = collection.find()
        data = []
        for res in results:
            res["_id"] = str(res["_id"])
            data.append(res)
        if data:
            return JSONResponse(content={
                "data": data,
                "code": 200,
                "message": "ok",
            })
        else:
            raise HTTPException(
                status_code=404, detail="Data not found in MongoDB.")

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

# New route for face detection


@app.post('/detect_faces/')
async def detect_faces(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        # Convert the image color space from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image_rgb)

        # Convert face locations to x, y coordinates of each corner
        face_coordinates = []
        for face_location in face_locations:
            top, right, bottom, left = face_location
            # Convert to x, y coordinates of each corner
            x_left = left
            y_top = top
            x_right = right
            y_bottom = bottom

            # Append the coordinates to the list
            face_coordinates.append({
                "top_left": {"x": x_left, "y": y_top},
                "top_right": {"x": x_right, "y": y_top},
                "bottom_right": {"x": x_right, "y": y_bottom},
                "bottom_left": {"x": x_left, "y": y_bottom}
            })

            # Draw rectangles around the detected faces
            cv2.rectangle(image_rgb, (left, top),
                          (right, bottom), (0, 255, 0), 2)

        # Encode the resulting image as JPEG binary data
        retval, buffer = cv2.imencode('.jpg', image_rgb)
        image_rgb = buffer.tobytes()

        # Convert the image to base64
        base64_image = base64.b64encode(image_rgb).decode('utf-8')

        # Return the image as a streaming response
        image_response = StreamingResponse(
            content=image_rgb, media_type="image/jpeg")

        # Return the face coordinates and base64 image as a JSON response
        return JSONResponse(content={"face_locations": face_coordinates, "base64_image": base64_image}, status_code=200)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
