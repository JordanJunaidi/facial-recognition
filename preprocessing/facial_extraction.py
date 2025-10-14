import cv2

# Change this to the image you want to extract faces from
TEST_IMAGE = "avengerscast.jpg"

# Load in Haar Cascade Frontal Face algorithm
haar_cascade = cv2.CascadeClassifier("preprocessing/haarcascade_frontalface_default.xml")

# Path for test image
# file_name = f"imgs/{TEST_IMAGE}"
file_name = "preprocessing/imgs/avengerscast.jpg"

# Read the image
img = cv2.imread(file_name)

# Conver the image to grayscale
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_img = cv2.equalizeHist(gray_img)

# Detect the faces
faces = haar_cascade.detectMultiScale(
    gray_img, scaleFactor=1.15, minNeighbors=5, minSize=(30, 30)
)

i = 0
for x, y, w, h in faces:
    # Crop image to select only face
    cropped_img = img[y : y + h, x : x + w]
    cropped_img = cv2.resize(cropped_img, (224,224)) 

    # Stores each detected face into stored-faces folder
    target_file_name = 'preprocessing/stored-faces/' + str(i) + '.jpg'
    cv2.imwrite(
        target_file_name,
        cropped_img
    )
    i = i + 1

