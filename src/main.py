import typing
import os

import pydantic
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from end2vid import end2vid
from io import BytesIO
from PIL import Image
import numpy as np
import scipy
from pathlib import Path
from pydantic import BaseModel
# from app.conf import settings
import urllib.request
import tempfile
import shutil
import cv2
from datetime import datetime

app = FastAPI()

# settings.post_setup()

def remove_file(path: str) -> None:
    os.unlink(path)

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)), np.uint8())[:,:,::-1]

class Urls(BaseModel):
    audio: str
    image: str

@app.post("/audio2vid_fromurl")
async def audio2vid_fromurl(background_tasks: BackgroundTasks, urls: Urls):

    # if (image.shape[0], image.shape[1]) != (256, 256):
    #     return HTTPException(400, detail="Image Size Error. Expected 256x256.")
    
    tempdir = tempfile.mkdtemp()
    audio_name = 'audio.wav'
    image_name = f'{tempdir[5:]}.jpg'
    urllib.request.urlretrieve(urls.audio, os.path.join(tempdir, audio_name))
    urllib.request.urlretrieve(urls.image, os.path.join(tempdir, image_name))

    audio = scipy.io.wavfile.read(os.path.join(tempdir, audio_name))
    image = cv2.imread(os.path.join(tempdir, image_name))

    shutil.rmtree(tempdir)
    
    start = datetime.now()
    video_path = end2vid(audio, image, audio_name, image_name)
    print(f'time: {datetime.now()-start}')
    background_tasks.add_task(remove_file, video_path)

    response = FileResponse(path=video_path, media_type='video/mp4')
    response.headers["Content-Disposition"] = f"attachment; filename={video_path.split('/')[-1]}"

    return response

@app.post("/audio2vid")
async def audio2vid(background_tasks: BackgroundTasks, files: typing.List[UploadFile] = File(...)):

    if len(files) != 2:
        return HTTPException(400, detail="Input Number Error")

    if files[0].filename[-4:] == '.wav':
        files.reverse()

    if files[0].filename[-4:] != '.jpg' or files[1].filename[-4:] != '.wav':
        return HTTPException(400, detail="Type Error. Expected .jpg and .wav")
    
    image = load_image_into_numpy_array(await files[0].read())
    audio = scipy.io.wavfile.read(BytesIO(await files[1].read()))
    
    if (image.shape[0], image.shape[1]) != (256, 256):
        return HTTPException(400, detail="Image Size Error. Expected 256x256.")

    start = datetime.now()
    video_path = end2vid(audio, image, files[1].filename, files[0].filename)
    print(f'time: {datetime.now()-start}')
    background_tasks.add_task(remove_file, video_path)

    # def iterfile():
    #     with open(video_path, "rb") as video:
    #         yield from video
    # return StreamingResponse(iterfile(), status_code=206, media_type="video/mp4")
    response = FileResponse(path=video_path, media_type='video/mp4')
    response.headers["Content-Disposition"] = f"attachment; filename={video_path.split('/')[-1]}"

    return response
