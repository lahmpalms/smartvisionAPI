from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
import face_recognition
import cv2
import base64
import numpy as np

app = FastAPI()

@app.post('/detect_faces/')
async def detect_faces(image: UploadFile = File(...)):
    try:
        contents = await image.read()
        nparr = np.frombuffer(contents, np.uint8)
        image = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        # Define maximum width and height for display
        max_width = 600
        max_height = 800

        # Check the image size and resize if it exceeds the maximum width or height
        if image.shape[1] > max_width or image.shape[0] > max_height:
            image = cv2.resize(image, (max_width, max_height))

        # Convert the image color space from BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Find all face locations in the image
        face_locations = face_recognition.face_locations(image_rgb)

        # Draw rectangles around the detected faces
        for face_location in face_locations:
            top, right, bottom, left = face_location
            cv2.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)

        # Encode the resulting image as a base64 string
        retval, buffer = cv2.imencode('.jpg', image_rgb)
        base64_image = base64.b64encode(buffer).decode('utf-8')

        return JSONResponse(content={'base64_image': base64_image})
    
    except Exception as e:
        return JSONResponse(content={'error': str(e)}, status_code=500)