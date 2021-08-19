import time
from camera import gen
import cv2
import uvicorn
from fastapi import File,UploadFile, FastAPI
import mediapipe as mp
from pydantic import BaseModel
import json
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose
app = FastAPI()
class Item(BaseModel):
    user_id: int
    client_type: str
    video: str

@app.post("/counter")
async def count(item:Item):

    video_path=item.video
    user_id=item.user_id
    client_type=item.client_type
    vid= gen(video_path)
    counter = vid[0]
    duration=vid[1]
    fps=vid[2]
    size=vid[3]
    start=vid[4]
    end=vid[5]
    return {"user_id":user_id,
         "client_type":client_type,
         "remarks":"good job continue",
         "exercise_type":"skipping",
         "exercise_sub_type":"skipping",
         "activity_count":counter,
        "activity_duration":duration,
            "activity_interval":"----",
            "video_path":video_path,
            "video_size":size,
            "video_resolution":size,
            "total_duration":duration,
            "position":"front_position",
            "time_to_process": duration,
            "start_time":start,
            "end_time":end,
            "frame_rate": fps
            }

if __name__ == '__main__':
    uvicorn.run(app, host='127.0.0.1', port=5000)
