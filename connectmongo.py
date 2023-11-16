import pymongo

myclient = pymongo.MongoClient("mongodb://admin:islabac123@18.143.76.245:27017/")
mydb = myclient["people_detect_log"]
mycol = mydb["log"]

mydict = { "name": "John", "address": "Highway 37" }

x = mycol.insert_one(mydict)