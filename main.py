from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse

from domain.question import question_router

from stream import get_stream_video

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

app.include_router(question_router.router)

# openCV에서 이미지 불러오는 함수
def video_streaming():
    return get_stream_video()

@app.get("/video")
def video():
    return StreamingResponse(video_streaming(), media_type="multipart/x-mixed-replace; boundary=frame")