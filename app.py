from flask import Flask, request, jsonify
from PIL import Image
import cv2
import os
import csv
import numpy as np
from datetime import datetime
import sqlite3

app = Flask(__name__)

# --------- Setup ---------
def assure_path_exists(path):
    if not os.path.exists(path):
        os.makedirs(path)

def init_database():
    conn = sqlite3.connect("attendance_system.db")
    cursor = conn.cursor()
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS students (
            serial INTEGER PRIMARY KEY,
            id INTEGER,
            name TEXT,
            image_path TEXT
        )
    """)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS attendance (
            id INTEGER,
            name TEXT,
            status TEXT,
            timestamp TEXT
        )
    """)
    conn.commit()
    conn.close()

init_database()

# --------- Face Capture & Training ---------
@app.route('/register', methods=['POST'])
def register_student():
    data = request.json
    Id = str(data.get("id"))
    name = str(data.get("name"))
    serial = str(data.get("serial"))

    if not Id.isnumeric() or not serial.isnumeric() or not name.isalpha():
        return jsonify({"error": "Invalid input"}), 400

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    assure_path_exists("TrainingImage")
    assure_path_exists("StudentDetails")
    face_captured = False

    while True:
        ret, img = cam.read()
        if not ret:
            return jsonify({"error": "Camera not accessible"}), 500

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, 1.1, 4)

        for (x, y, w, h) in faces:
            file_path = os.path.join("TrainingImage", f"{name}.{serial}.{Id}.jpg")
            if os.path.exists(file_path):
                os.remove(file_path)
            cv2.imwrite(file_path, gray[y:y+h, x:x+w])
            face_captured = True
            break

        if face_captured:
            break

    cam.release()
    cv2.destroyAllWindows()

    if not face_captured:
        return jsonify({"error": "No face detected"}), 400

    with open("StudentDetails/StudentDetails.csv", 'a', newline='') as f:
        csv.writer(f).writerow([serial, Id, name])

    conn = sqlite3.connect("attendance_system.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO students (serial, id, name, image_path) VALUES (?, ?, ?, ?)", 
                   (serial, Id, name, file_path))
    conn.commit()
    conn.close()

    train_images()

    return jsonify({"message": "Student registered and model trained"}), 200

def getImagesAndLabels(path):
    faces, Ids = [], []
    for file in os.listdir(path):
        try:
            img_path = os.path.join(path, file)
            img = Image.open(img_path).convert('L')
            image_np = np.array(img, 'uint8')
            id_ = int(file.split('.')[2])
            faces.append(image_np)
            Ids.append(id_)
        except:
            continue
    return faces, Ids

def train_images():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    assure_path_exists("TrainingImageLabel")
    faces, Ids = getImagesAndLabels("TrainingImage")
    if not faces:
        return
    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")

# --------- Recognition ---------
@app.route('/recognize', methods=['POST'])
def recognize_and_mark_attendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("TrainingImageLabel/Trainner.yml")
    except:
        return jsonify({"error": "Model not trained"}), 500

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    conn = sqlite3.connect("attendance_system.db")
    cursor = conn.cursor()
    student_dict = {row[0]: row[1] for row in cursor.execute("SELECT id, name FROM students")}
    assure_path_exists("Attendance")

    cam = cv2.VideoCapture(0)
    marked = False
    response = {}

    while True:
        ret, img = cam.read()
        if not ret:
            break

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = cascade.detectMultiScale(gray, 1.2, 5)

        for (x, y, w, h) in faces:
            id_, confidence = recognizer.predict(gray[y:y+h, x:x+w])
            name = student_dict.get(id_, "Unknown")

            if confidence < 50 and name != "Unknown":
                now = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

                with open("Attendance/Attendance.csv", 'a', newline='') as f:
                    csv.writer(f).writerow([id_, name, "Present", now])

                cursor.execute("INSERT INTO attendance (id, name, status, timestamp) VALUES (?, ?, ?, ?)",
                               (id_, name, "Present", now))
                conn.commit()

                response = {"id": id_, "name": name, "status": "Present", "timestamp": now}
                marked = True
                break

        if marked:
            break

    cam.release()
    cv2.destroyAllWindows()
    conn.close()

    if marked:
        return jsonify(response), 200
    else:
        return jsonify({"message": "No known face recognized"}), 404

# --------- Get Attendance by Date ---------
@app.route('/attendance', methods=['GET'])
def get_attendance_by_date():
    from_date = request.args.get("from")
    to_date = request.args.get("to")

    try:
        datetime.strptime(from_date, "%Y-%m-%d")
        datetime.strptime(to_date, "%Y-%m-%d")
    except:
        return jsonify({"error": "Date format must be YYYY-MM-DD"}), 400

    conn = sqlite3.connect("attendance_system.db")
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, name, status, timestamp FROM attendance
        WHERE DATE(timestamp) BETWEEN ? AND ?
    """, (from_date, to_date))
    records = cursor.fetchall()
    conn.close()

    return jsonify([{"id": r[0], "name": r[1], "status": r[2], "timestamp": r[3]} for r in records])

# --------- Run ---------
if __name__ == "__main__":
    app.run(debug=True)
