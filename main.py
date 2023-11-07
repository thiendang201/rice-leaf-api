from fastapi import FastAPI, File, UploadFile
import numpy as np
import cv2
from skimage.feature import hog
import joblib
import json
from fastapi.middleware.cors import CORSMiddleware

svc_model = joblib.load('linear_svc_model.pkl')

with open('pests.json', encoding="utf8") as json_file:
    pests = json.load(json_file)

app = FastAPI()
origins = ["*"]
app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


def getPest(id):
  global pests
  key = str(id)

  if key in pests:
    return pests[key]

  return {}

@app.get('/pest/get')
def get_pest(id):
  for key in pests:
    if pests[key]['ma_sau_benh'] == id:
      return pests[key]

  return {}

@app.post('/predict')
async def rice_leaf_pest_prediction(file: UploadFile = File(...)):
  contents = await file.read()
  jpg_as_np = np.frombuffer(contents, dtype=np.uint8)
  image = cv2.imdecode(jpg_as_np, cv2.IMREAD_COLOR)
  image_shape = (128, 128)

  #Resize ảnh về kích cỡ yêu cầu của mô hình
  image = cv2.resize(image, image_shape, interpolation = cv2.INTER_AREA)
  image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

  #Preprocessing dữ liệu về khoảng 0-1
  image = image/255.0

  orientations = 12
  pixels_per_cell = (8, 8)
  cells_per_block = (2, 2)
  block_norm = 'L2'

  X1 = []
  fd1 = hog(image, orientations=orientations, pixels_per_cell=pixels_per_cell, cells_per_block=cells_per_block,block_norm= block_norm, visualize=False)
  X1.append(fd1)

  rs = svc_model.predict(X1)
  return getPest(rs[0].item())