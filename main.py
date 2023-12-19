from fastapi import FastAPI, HTTPException
from io import BytesIO
from PIL import Image
import base64
import numpy as np
from pydantic import BaseModel
from tensorflow.keras.models import load_model
from starlette.middleware.cors import CORSMiddleware

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

model = load_model("i_model.h5", compile=False)

@app.post("/uploadfile/")
async def create_upload_file(image_data: ImageData):
    try:
        # 원본 이미지를 image에 저장합니다.
        header, encoded = image_data.image_str.split(",", 1)  
        image_bytes = base64.b64decode(encoded)
        image = Image.open(BytesIO(image_bytes)).convert('RGB')

        # 모델에 맞는 인풋 형태로 변형합니다.
        image = image.resize((64, 64))
        image_array = np.array(image) / 255.0
        image_array = np.expand_dims(image_array, axis=0)

        # 이미지를 예측하고 결과를 반환합니다.
        # 반환하는 결과는 1. 예측한 클래스: int, 2. 각 클래스별 예측한 값: List<int>
        result = model.predict(image_array)
        result_list = result.tolist()[0]
        class_0_prob = result_list[0]
        class_1_prob = result_list[1]
        class_2_prob = result_list[2]
        if 0.31 <= class_0_prob < 0.35 and class_1_prob < 0.385 and class_2_prob < 0.316:
            predicted_class = 0
        elif 0.3 <= class_0_prob < 0.355 and 0.35 <= class_1_prob and 0.26 <= class_2_prob < 0.325:
            predicted_class = 1
        elif 0.29 <= class_0_prob < 0.355 and 0.36 <= class_1_prob < 0.39 and 0.28 <= class_2_prob < 0.32:
            predicted_class = 2
        else:
            predicted_class = -1
        return {"predicted_class": predicted_class, "acc_arr": result_list}
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"이미지 처리 실패: {str(e)}")