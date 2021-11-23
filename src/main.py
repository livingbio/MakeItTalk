import typing
import os

import pydantic
from fastapi import FastAPI, File, UploadFile, BackgroundTasks, HTTPException
from fastapi.responses import FileResponse

from end2vid import End2vid
from io import BytesIO
from PIL import Image
import numpy as np
import scipy
from pathlib import Path
from pydantic import BaseModel
from app.conf import settings
import requests
import tempfile
from tempfile import NamedTemporaryFile
import shutil
import cv2
from datetime import datetime
import cloudinary
from cloudinary.uploader import upload as cloudinary_upload

app = FastAPI()

cloudinary.config(
    cloud_name=settings.CLOUDINARY_CLOUD_NAME,
    api_key=settings.CLOUDINARY_API_KEY,
    api_secret=settings.CLOUDINARY_API_SECRET,
    secure=True,
)

def remove_file(path: str) -> None:
    os.unlink(path)

def load_image_into_numpy_array(data):
    return np.array(Image.open(BytesIO(data)), np.uint8())[:,:,::-1]

class Urls(BaseModel):
    audio: str
    image: str
    
end2vid = End2vid()

@app.post("/audio2vid")
async def audio2vid_from_url(background_tasks: BackgroundTasks, urls: Urls):

    # Check the content type of the URL before downloading the content
    try:
        h = requests.head(urls.image, allow_redirects=True)
    except Exception:
        raise HTTPException(400, detail="Invalid URL for image")
    if "image/jpeg" not in h.headers["Content-Type"]:
        raise HTTPException(400, detail="Invalid image file type: expected jpg/jpeg")
    try:
        h = requests.head(urls.audio, allow_redirects=True)
    except Exception:
        raise HTTPException(400, detail="Invalid URL for audio")
    if "audio/wav" not in h.headers["Content-Type"] and "audio/x-wav" not in h.headers["Content-Type"]:
        raise HTTPException(400, detail="Invalid audio file type: expected wav")

    # Download and write files to temporary directory
    resp = requests.get(urls.image)
    with NamedTemporaryFile(delete=False, suffix=".jpg") as image_tmp:
        image_tmp.write(resp.content)

    resp = requests.get(urls.audio)
    with NamedTemporaryFile(delete=False, suffix=".mp4") as audio_tmp:
        audio_tmp.write(resp.content)

    # Load file
    audio = scipy.io.wavfile.read(audio_tmp.name)
    image = cv2.imread(image_tmp.name)

    # Check image size
    if (image.shape[0], image.shape[1]) != (256, 256):
        return HTTPException(400, detail="Invalid image size: expected 256*256")
    
    # Model inference
    start = datetime.now()
    video_path = end2vid.run(audio, image, 'audiooo.wav', image_tmp.name.split('/')[-1])
    print(f'time: {datetime.now()-start}')

    # Upload model output
    upload_resp = cloudinary_upload(
        video_path,
        folder="makeittalk-outputs",
        resource_type="video",
    )
    background_tasks.add_task(remove_file, video_path)
    return {"output_url": upload_resp["url"]}
