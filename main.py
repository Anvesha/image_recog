import tkinter as tk
from tkinter import ttk, messagebox
from PIL import Image
import cv2
import os
import csv
import numpy as np
from datetime import datetime
import sqlite3

# --------- Utilities ---------
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

# --------- One-time Reset for Attendance Table ---------
def reset_attendance_table():
    conn = sqlite3.connect("attendance_system.db")
    conn.execute("DROP TABLE IF EXISTS attendance")
    conn.commit()
    conn.close()
    conn.close()

reset_attendance_table()  # <-- Run once
init_database()

# --------- Image Capture & Training ---------
def TakeImages():
    Id = id_entry.get()
    name = name_entry.get()
    serial = serial_entry.get()

    if not Id.isnumeric() or not name.isalpha() or not serial.isnumeric():
        messagebox.showerror("Error", "Enter valid Serial (number), ID (number), and Name (letters only).")
        return

    cam = cv2.VideoCapture(0)
    detector = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    assure_path_exists("TrainingImage")
    face_captured = False

    while True:
        ret, img = cam.read()
        if not ret:
            messagebox.showerror("Error", "Failed to access camera.")
            return

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = detector.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=4, minSize=(100, 100))

        for (x, y, w, h) in faces:
            file_path = os.path.join("TrainingImage", f"{name}.{serial}.{Id}.jpg")
            if os.path.exists(file_path): os.remove(file_path)
            cv2.imwrite(file_path, gray[y:y + h, x:x + w])
            face_captured = True
            break

        for (x, y, w, h) in faces:
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

        cv2.imshow("Face Capture - Press Q to Exit", img)
        if face_captured or cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cam.release()
    cv2.destroyAllWindows()

    if not face_captured:
        messagebox.showwarning("Warning", "No face detected. Try again.")
        return

    # Save to CSV
    assure_path_exists("StudentDetails")
    with open("StudentDetails/StudentDetails.csv", 'a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([serial, Id, name])

    # Save to DB
    conn = sqlite3.connect("attendance_system.db")
    cursor = conn.cursor()
    cursor.execute("INSERT OR REPLACE INTO students (serial, id, name, image_path) VALUES (?, ?, ?, ?)", 
                   (serial, Id, name, file_path))
    conn.commit()
    conn.close()

    messagebox.showinfo("Success", f"Image saved & student registered!\nID: {Id}, Name: {name}")
    TrainImages()

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

def TrainImages():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    assure_path_exists("TrainingImageLabel")
    faces, Ids = getImagesAndLabels("TrainingImage")

    if not faces:
        messagebox.showerror("Error", "No face data found!")
        return

    recognizer.train(faces, np.array(Ids))
    recognizer.save("TrainingImageLabel/Trainner.yml")

# --------- Recognition & Attendance ---------
def RecognizeAndMarkAttendance():
    recognizer = cv2.face.LBPHFaceRecognizer_create()
    try:
        recognizer.read("TrainingImageLabel/Trainner.yml")
    except:
        messagebox.showerror("Error", "Train the model first!")
        return

    cascade = cv2.CascadeClassifier("haarcascade_frontalface_default.xml")
    conn = sqlite3.connect("attendance_system.db")
    cursor = conn.cursor()
    student_dict = {row[0]: row[1] for row in cursor.execute("SELECT id, name FROM students")}
    assure_path_exists("Attendance")

    cam = cv2.VideoCapture(0)
    marked = False

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

                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(img, f"{name} | Marked", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)
                marked = True
                break  # Stop after first success
            else:
                cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 255), 2)
                cv2.putText(img, "Unknown", (x, y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        cv2.imshow("Recognizing...", img)
        if marked:
            cv2.waitKey(1000)
            break

        if cv2.waitKey(1) == 27:
            break

    cam.release()
    cv2.destroyAllWindows()
    conn.close()

    if marked:
        messagebox.showinfo("Success", "Attendance marked successfully!")
    else:
        messagebox.showwarning("Info", "No known face recognized.")

# --------- GUI ---------
root = tk.Tk()
root.title("Face Recognition Attendance System")
root.geometry("600x400")
root.resizable(False, False)

ttk.Label(root, text="Face Recognition Attendance System", font=("Helvetica", 16, "bold")).pack(pady=20)
form = ttk.Frame(root, padding=10)
form.pack()

ttk.Label(form, text="Serial No:", font=("Helvetica", 12)).grid(row=0, column=0, padx=10, pady=5, sticky="e")
serial_entry = ttk.Entry(form, width=30)
serial_entry.grid(row=0, column=1, pady=5)

ttk.Label(form, text="ID:", font=("Helvetica", 12)).grid(row=1, column=0, padx=10, pady=5, sticky="e")
id_entry = ttk.Entry(form, width=30)
id_entry.grid(row=1, column=1, pady=5)

ttk.Label(form, text="Name:", font=("Helvetica", 12)).grid(row=2, column=0, padx=10, pady=5, sticky="e")
name_entry = ttk.Entry(form, width=30)
name_entry.grid(row=2, column=1, pady=5)

btns = ttk.Frame(root, padding=10)
btns.pack(pady=10)

ttk.Button(btns, text="Take Image", command=TakeImages).grid(row=0, column=0, padx=10)
ttk.Button(btns, text="Mark Attendance", command=RecognizeAndMarkAttendance).grid(row=0, column=1, padx=10)

root.mainloop()
