from PIL import Image
from imgbeddings import imgbeddings
import cv2
from dotenv import load_dotenv
import os
import psycopg2


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