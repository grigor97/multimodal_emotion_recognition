import face_recognition
from PIL import Image
import numpy as np


def face_extraction(pic_path):
    image = face_recognition.load_image_file(pic_path)
    face_locs = face_recognition.face_locations(image)

    if len(face_locs) > 0:
        top, right, bottom, left = face_locs[0]
        face_image = image[top:bottom, left:right]
        pil_image = Image.fromarray(face_image)
    else:
        pil_image = Image.fromarray(image)

    pil_image = pil_image.resize((50, 50))
    return np.array(pil_image)



