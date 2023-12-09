from fastapi import FastAPI, File, UploadFile, HTTPException
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
import imageio

from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

# from domain.question import question_router

#from stream import get_stream_video

app = FastAPI()

origins = [
    "http://localhost:3000",    # 또는 "http://localhost:5173"
]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class ImageData(BaseModel):
    image_str: str

model = load_model("model.h5", compile=False)

@app.post("/uploadfile/")
async def create_upload_file(image_data: ImageData):
    try:
        header, encoded = image_data.image_str.split(",", 1)  
        # print(encoded)
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # 이미지 처리 로직
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        result = model.predict(image_array)
        result_list = result.tolist()[0]
        max_idx, max_value = 0, 0
        for idx, value in enumerate(result_list):
            if max_value < value:
                max_idx = idx
                max_value = value
        return {"predict_class": max_idx + 1, "accuracy": max_value, "2": result_list[1]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")