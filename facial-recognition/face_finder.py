'''
FaceFinder: Locate a specific person in a group photo using facial recognition.
'''
import cv2
import face_recognition  # pip install face_recognition

# load the “target” face:
known_image = face_recognition.load_image_file("images/target.jpg")
known_encoding = face_recognition.face_encodings(known_image)[0]

# load the group image:
image = face_recognition.load_image_file("images/group.jpg")
face_locations = face_recognition.face_locations(image)
face_encodings = face_recognition.face_encodings(image, face_locations)

# convert to OpenCV format (BGR) to draw boxes:
image_bgr = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

# iterate:
for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
    matches = face_recognition.compare_faces([known_encoding], face_encoding)
    if matches[0]:
        # found the target person
        cv2.rectangle(image_bgr, (left, top), (right, bottom), (255, 0, 0), 2)  # Blue box (BGR)
        cv2.putText(image_bgr, "Target", (left, top-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255,0,0),2)
        break  # if you expect only one appearance

# show or save the result
cv2.imshow("Result", image_bgr)
cv2.waitKey(0)
cv2.destroyAllWindows()
