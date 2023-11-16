from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
import face_recognition
import cv2
import numpy as np
import base64

app = FastAPI()

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
            cv2.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)

        # Encode the resulting image as JPEG binary data
        retval, buffer = cv2.imencode('.jpg', image_rgb)
        image_jpg = buffer.tobytes()

        # Convert the image to base64
        base64_image = base64.b64encode(image_jpg).decode('utf-8')

        # Return the image as a streaming response
        image_response = StreamingResponse(content=image_jpg, media_type="image/jpeg")

        # Return the face coordinates and base64 image as a JSON response
        return JSONResponse(content={"face_locations": face_coordinates, "base64_image": base64_image}, status_code=200)

    except FileNotFoundError as e:
        raise HTTPException(status_code=404, detail="File not found")
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
