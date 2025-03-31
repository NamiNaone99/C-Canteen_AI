import cv2

RTSP_URL = "rtsp://admin:Mobi-10005@192.168.0.99:554/Streaming/Channels/101"

gst_pipeline = (
    f"rtspsrc location={RTSP_URL} ! "
    "rtph264depay ! h264parse ! avdec_h264 ! videoconvert ! appsink"
)
cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)

if not cap.isOpened():
    print("Error: Unable to open video stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame.")
        break

    cv2.imshow("Video Stream", frame)

    # Press 'q' to exit the video stream
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()