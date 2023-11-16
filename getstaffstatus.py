from fastapi import FastAPI
from pymongo import MongoClient
from pydantic import BaseModel
from fastapi import HTTPException
from bson.objectid import ObjectId
from datetime import datetime
from typing import List


app = FastAPI()

# MongoDB connection setup
myclient = MongoClient("mongodb://admin:islabac123@18.143.76.245:27017/")
mydb = myclient["people_detect_log"]
mycol = mydb["log"]

class LogItem(BaseModel):
    status: str
    time: str

class LogItemWithID(BaseModel):
    id: str
    log: LogItem

@app.get("/logs/{log_id}", response_model=LogItemWithID)
async def read_log(log_id: str):
    log_data = mycol.find_one({"_id": ObjectId(log_id)})
    if log_data is None:
        raise HTTPException(status_code=404, detail="Log not found")

    log_time = log_data.get("time")
    if log_time and isinstance(log_time, datetime):
        log_data["time"] = log_time.strftime("%Y-%m-%d %H:%M:%S")

    log_id = str(log_data["_id"])  # Convert ObjectId to a string for the ID
    return LogItemWithID(id=log_id, log=LogItem(**log_data))

@app.get("/alllogs", response_model=List[LogItemWithID])
async def get_all_logs():
    log_data = mycol.find({})  # Retrieve all log items
    log_list = []

    for log_item in log_data:
        log_time = log_item.get("time")
        if log_time and isinstance(log_time, datetime):
            log_item["time"] = log_time.strftime("%Y-%m-%d %H:%M:%S")
        log_id = str(log_item["_id"])  # Convert ObjectId to a string for the ID
        log_list.append(LogItemWithID(id=log_id, log=LogItem(**log_item)))

    return log_list