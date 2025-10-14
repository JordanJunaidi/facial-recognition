from PIL import Image
from imgbeddings import imgbeddings
import cv2
from dotenv import load_dotenv
import os
import psycopg2

def crop_face_for_embeddings(path, cascade_path="haarcascade_frontalface_default.xml"):
    img = cv2.imread(path)
    if img is None:
        raise FileNotFoundError(path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)
    cascade = cv2.CascadeClassifier(cascade_path)
    if cascade.empty():
        raise FileNotFoundError(cascade_path)
    faces = cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30,30))
    if len(faces) == 0:
        # fallback: center-crop or return None
        h,w = img.shape[:2]
        s = min(h,w)
        y0 = (h-s)//2; x0 = (w-s)//2
        crop = img[y0:y0+s, x0:x0+s]
    else:
        x,y,w,h = sorted(faces, key=lambda r: r[2]*r[3], reverse=True)[0]
        crop = img[y:y+h, x:x+w]
    crop = cv2.resize(crop, (224,224))
    return Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))

# Change this to the image you want to use
file_name="anthony.jpg"

img = Image.open(f"new_image/imgs/{file_name}")

ibed = imgbeddings()

embedding = ibed.to_embeddings(img)

load_dotenv()
conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),        
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

cur = conn.cursor()
string_representation = "[" + ",".join(str(x) for x in embedding[0].tolist()) + "]"
cur.execute("SELECT * FROM pictures ORDER BY embedding <-> %s LIMIT 1;", (string_representation,))
rows = cur.fetchall()
for row in rows:
    matched_filename = row[0]

    matched_path = os.path.join("preprocessing/stored-faces", matched_filename)
    if os.path.exists(matched_path):
        matched_img = cv2.imread(matched_path)
        cv2.imshow(matched_filename, matched_img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()
    else:
        print("File not found")