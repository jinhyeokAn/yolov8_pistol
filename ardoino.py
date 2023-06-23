import cv2
import numpy as np
import serial
from ultralytics import YOLO
from utils import plot_frame_boex

# Open serial communication with Arduino
arduino = serial.Serial('COM4', 9600)  # Replace 'COM3' with the appropriate port and baud rate

model = YOLO("pistol.pt")

# 웹캠 열기
cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640) #해상도 결정
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 360)

while True:
    ret, frame = cap.read()

    # 프레임에 대한 객체 감지 수행
    results = model.predict(frame)

    # 프레임에 객체 감지 결과 표시
    #plot_frame_boex(frame, results[0].boxes.data, score=False)
    plot_frame_boex(frame, results[0].boxes.data, conf=0.8)
    
    # 결과 출력
    cv2.imshow("Object Detection", frame)

    # 만약 pistol이 감지되면 아두이노에 메시지 전송
    for box in results[0].boxes.data:
        if box[-1] == 0:  # Assuming pistol class index is 0
            arduino.write(b'pistol detected\n')
    
    # 'q' 키를 누르면 종료
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

# 작업 완료 후 해제
cap.release()
cv2.destroyAllWindows()
arduino.close()
