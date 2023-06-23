from glob import glob
from sklearn.model_selection import train_test_split
import yaml
from ultralytics.yolo.engine.model import YOLO
from ultralytics.vit.rtdetr import model
from ultralytics import yolo

img_list = glob('./dataset/export/images/*.jpg')

print("number of pictures : {0}".format(len(img_list)))

train_img_list, val_img_list = train_test_split(img_list, test_size=0.2, random_state=2000)


print("test img : {0}, val img : {1}".format(len(train_img_list), len(val_img_list)))


with open('./dataset/train.txt', 'w') as f:
  f.write('\n'.join(train_img_list) + '\n')

with open('./dataset/val.txt', 'w') as f:
  f.write('\n'.join(val_img_list) + '\n')
  


with open('./dataset/data.yaml', 'r') as f:
    data = yaml.load(f, Loader=yaml.FullLoader)

print(data)

data['train'] = './dataset/train.txt'
data['val'] = './dataset/val.txt'

with open('./dataset/data.yaml', 'w') as f:
    yaml.dump(data, f)

print("==========================data================================")
print(data)
print("==========================data================================")

print()
print()
print()
print()

model = YOLO('yolov8n.pt')
model.train(data = './dataset/data.yaml', epochs = 1, patience=30, batch=32, imgsz=416)