import os
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from peoplecountv8 import main  # Import your detection function

app = FastAPI()

@app.post("/detect-people/")
async def detect_people(image: UploadFile):
    # Save the uploaded image file
    with open("input_image.jpg", "wb") as f:
        f.write(image.file.read())

    # Call your YOLOv8 detection function
    output_image_path = main("input_image.jpg")  # Modify the detection function

    if output_image_path:
        # Convert the output image to base64
        with open(output_image_path, "rb") as image_file:
            image_base64 = base64.b64encode(image_file.read()).decode("utf-8")

        return JSONResponse(content={"image_base64": image_base64})
    else:
        return JSONResponse(content={"error": "No valid 'predictX' directories found."})

if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
