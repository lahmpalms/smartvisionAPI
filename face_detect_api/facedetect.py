import face_recognition
import cv2

# Load an image file (in this case, an image containing faces)
image_path = '/home/nont/smartvisionAPI/face_detect_api/20230210_145008.jpg'
image = face_recognition.load_image_file(image_path)

# Define maximum width and height for display
max_width = 600
max_height = 800

# Check the image size and resize if it exceeds the maximum width or height
if image.shape[1] > max_width or image.shape[0] > max_height:
    image = cv2.resize(image, (max_width, max_height))

# Convert the image color space from BGR to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Find all face locations in the image
face_locations = face_recognition.face_locations(image_rgb)

print(f"Found {len(face_locations)} face(s) in the image.")

# Display the image with rectangles around the detected faces
for face_location in face_locations:
    top, right, bottom, left = face_location

    # Draw a rectangle around the face
    cv2.rectangle(image_rgb, (left, top), (right, bottom), (0, 255, 0), 2)

# Show the resized image with detected faces
cv2.imshow('Detected Faces', image_rgb)
cv2.waitKey(0)
cv2.destroyAllWindows()
