import psycopg2
import os
from dotenv import load_dotenv
import numpy as np
from imgbeddings import imgbeddings
from PIL import Image

load_dotenv()

print(os.getenv("DB_NAME"))

conn = psycopg2.connect(
    dbname=os.getenv("DB_NAME"),
    user=os.getenv("DB_USER"),        
    password=os.getenv("DB_PASSWORD"),
    host=os.getenv("DB_HOST"),
    port=os.getenv("DB_PORT")
)

cur = conn.cursor()
cur.execute("TRUNCATE TABLE pictures RESTART IDENTITY;")
conn.commit()

for filename in os.listdir("preprocessing/stored-faces"):
    img = Image.open("preprocessing/stored-faces/" + filename)

    ibed = imgbeddings()

    embedding = ibed.to_embeddings(img)
    cur = conn.cursor()
    cur.execute("INSERT INTO pictures values (%s,%s)", (filename, embedding[0].tolist()))
    print(filename)

conn.commit()
cur.close()
conn.close()
