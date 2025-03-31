import cv2

# Replace with your RTSP stream URL
rtsp_url = "rtsp://admin:Password@1@192.168.0.67:554/Streaming/Channels/101"

# Open the RTSP stream
cap = cv2.VideoCapture(rtsp_url)

if not cap.isOpened():
    print("Error: Unable to open RTSP stream.")
    exit()

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Unable to read frame from stream.")
        break

    # Display the frame
    cv2.imshow("RTSP Stream", frame)

    # Press 'q' to exit
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close the window
cap.release()
cv2.destroyAllWindows()