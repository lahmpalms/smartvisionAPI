import os
import shutil
import base64
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from pymongo import MongoClient
from peoplecountv8 import main, process_video  # Import your detection function
from typing import Annotated, Generator

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb://admin:islabac123@18.143.76.245:27017")
db = client["people_detect_log"]  # Replace with your actual database name
collection = db["log_cloudprocess"]  # Replace with your actual collection name


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
                        chunk = file_like.read(1024)
                        while chunk:
                            yield chunk
                            chunk = file_like.read(1024)
                finally:
                    # Remove the image file after reading
                    os.remove(output_image_path["latest_image_path"])
            return StreamingResponse(stream_video(), media_type="video/mp4")
        else:
            return JSONResponse(content={"error": "Failed to insert data into MongoDB."})

    except Exception as e:
        return JSONResponse(content={"error": str(e)})


def get_data_from_file(file_path: str) -> Generator:
    with open(file=file_path, mode="rb") as file_like:
        yield file_like.read()


if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
