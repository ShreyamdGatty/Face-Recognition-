import cv2
import csv
from datetime import datetime

# Load Haar Cascade for face detection and LBPH recognizer
face_detect = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
cam = cv2.VideoCapture(0)
rec = cv2.face.LBPHFaceRecognizer_create()
rec.read("recognizer\\trainingData.yml")

# Initialize variables
font = cv2.FONT_HERSHEY_COMPLEX_SMALL
attendance = {}  # Dictionary to store attendance (key: id, value: status)

# Function to save attendance to a CSV file
def save_attendance_to_csv(attendance):
    filename = f"attendance_{datetime.now().strftime('%Y-%m-%d')}.csv"
    with open(filename, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerow(["ID", "Name", "Status", "Timestamp"])
        for id, details in attendance.items():
            writer.writerow([id, details["name"], details["status"], details["time"]])

while True:
    ret, img = cam.read()
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_detect.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x + w, y + h), (255, 0, 0), 2)
        ide, conf = rec.predict(gray[y:y+h, x:x+w])
        
        # Set confidence threshold and recognize face
        if conf < 60:
            if ide == 1:
                name = "Shreya M D"
            elif ide == 2:
                name = "shreya k"
            elif ide == 3:
                name = "vidya"
            elif ide == 4:
                name = "Vishaka"
            elif ide == 5:
                name = "vaishali"
            elif ide == 6:
                name = "pavan"
            elif ide == 7:
                name = "sumana"
            elif ide == 8:
                name = "neha"
            elif ide == 9:
                name ="Lathashree"
        else:
            name = "Unknown"
            ide = None  # Mark as unrecognized
        
        if ide and ide not in attendance:
            # Log attendance if not already recorded
            attendance[ide] = {
                "name": name,
                "status": "Present",
                "time": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            }
        
        cv2.putText(img, name, (x, y + h), font, 2, (255, 255, 255), 2)
    
    cv2.imshow("Face", img)
    if cv2.waitKey(1) == ord('q'):
        break

# Save attendance to CSV and release resources
save_attendance_to_csv(attendance)
cam.release()
cv2.destroyAllWindows()
