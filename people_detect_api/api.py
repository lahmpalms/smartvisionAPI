import os
import shutil
import base64
from fastapi import FastAPI, File, UploadFile
from fastapi.responses import JSONResponse
from peoplecountv8 import main  # Import your detection function

app = FastAPI()


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
        if output_image_path["latest_image_path"]:
            # Convert the output image to base64
            with open(output_image_path["latest_image_path"], "rb") as image_file:
                image_base64 = base64.b64encode(
                    image_file.read()).decode("utf-8")
                if os.path.exists(output_image_path["latest_image_path"]):
                    os.remove(output_image_path["latest_image_path"])
                else:
                    print(f"File does not exist.")

            return JSONResponse(content={"data": {
                "base64_image": image_base64, "coordinate_data": coordinate},
                "code": 200,
                "message": "ok", })
        else:
            return JSONResponse(content={"error": "No valid 'predict' directories found."})
    except Exception as e:
        return JSONResponse(content={"error": str(e)})
if __name__ == '__main__':
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000, reload=True)
