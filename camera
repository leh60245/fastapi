import io
from base64 import b64decode
from PIL import Image
from IPython.display import Image as IPImage, display

# Colab에서 웹캠 사용을 위한 JavaScript 코드 실행
def start_webcam():
    js = Javascript('''
        async function startWebcam() {
            const video = document.createElement('video');
            const stream = await navigator.mediaDevices.getUserMedia({ 'video': true });
            video.srcObject = stream;

            // Wait for the video to be loaded
            await video.play();

            // Capture a frame from the video stream
            const canvas = document.createElement('canvas');
            canvas.width = video.videoWidth;
            canvas.height = video.videoHeight;
            canvas.getContext('2d').drawImage(video, 0, 0);

            // Return the captured frame as a JPEG data URL
            return canvas.toDataURL('image/jpeg', 0.8);
        }
    ''')
    display(js)

# 캡처된 이미지를 NumPy 배열로 변환
def capture_webcam():
    data_url = eval_js('startWebcam()')
    binary_data = b64decode(data_url.split(',')[1])
    image = Image.open(io.BytesIO(binary_data))

    # 투명도가 있는 경우 투명도 제거
    if image.mode == 'RGBA':
        image = image.convert('RGB')

    return image

# 웹캠 시작
start_webcam()

# 이미지 캡처
captured_image = capture_webcam()

# 이미지를 JPEG 파일로 저장
captured_image.save('captured_image.jpg', 'JPEG')

# 저장된 이미지 표시
display(IPImage(filename='captured_image.jpg'))

from tensorflow.keras.preprocessing import image
import numpy as np

def preprocess_webcam_image(img):
    img = img.resize((image_height, image_width))
    img_array = image.img_to_array(img)
    img_array = np.expand_dims(img_array, axis=0)
    img_array /= 255.0  # 정규화

    return img_array

# 이미지 전처리
preprocessed_image = preprocess_webcam_image(captured_image)

# 모델로 예측
predictions = model.predict(preprocessed_image)
predicted_class = np.argmax(predictions)

# 예측 결과 출력
print(f"Predicted Class: class_{predicted_class}")
