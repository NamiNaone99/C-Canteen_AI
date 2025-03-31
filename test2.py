import cv2

# List to store clicked coordinates
clicked_points = []

# Mouse callback function
def mouse_callback(event, x, y, flags, param):
    if event == cv2.EVENT_LBUTTONDOWN:
        clicked_points.append((x, y))
        print(f"Clicked at: ({x}, {y})")
        cv2.circle(resized_image, (x, y), 5, (0, 0, 255), -1)  # Red dot

# Load an image
image_path = r"/mnt/c/Users/nongf/Desktop/CUNEX/experiment/IMG20250226133959.jpg"
image = cv2.imread(image_path)

# Check if the image was loaded successfully
if image is None:
    raise FileNotFoundError(f"Error: Image not found at {image_path}")

# Resize image for display
scale_factor = 0.5  # Reduce to 50% of original size
resized_image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)

# Set up the mouse callback
cv2.namedWindow("Image with Click Coordinates")
cv2.setMouseCallback("Image with Click Coordinates", mouse_callback)

# Display the resized image
while True:
    cv2.imshow("Image with Click Coordinates", resized_image)
    key = cv2.waitKey(1) & 0xFF
    
    if key == ord('s'):  # Save coordinates to file
        with open("clicked_coordinates.txt", "w") as f:
            for point in clicked_points:
                f.write(f"{point[0]},{point[1]}\n")
        print("Coordinates saved.")
    
    if key == ord('q'):  # Quit
        break

cv2.destroyAllWindows()
