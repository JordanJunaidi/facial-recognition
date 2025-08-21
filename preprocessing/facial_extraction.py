import cv2

# Change this to the image you want to extract faces from
TEST_IMAGE = "avengerscast.jpg"

# Load in Haar Cascade Frontal Face algorithm
haar_cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")

# Path for test image
file_name = f"imgs/{TEST_IMAGE}"

# Read the image
img = cv2.imread(file_name)

# Conver the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# Detect the faces
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30)
)

i = 0
for x, y, w, h in faces:
    # Crop image to select only face
    cropped_img = img[y : y + h, x : x + w]

    # Stores each detected face into stored-faces folder
    target_file_name = 'stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_img
    )
    i = i + 1

