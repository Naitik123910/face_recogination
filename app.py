import streamlit as st
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
from datetime import datetime

# 📂 Load student images
path = "photos"
images = []
student_names = []
for file in os.listdir(path):
    cur_img = cv2.imread(f"{path}/{file}")
    images.append(cur_img)
    student_names.append(os.path.splitext(file)[0])  # filename without extension

# 🔑 Encode known faces
def find_encodings(images):
    encode_list = []
    for img in images:
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        encodes = face_recognition.face_encodings(img)
        if len(encodes) > 0:
            encode_list.append(encodes[0])
    return encode_list

encode_list_known = find_encodings(images)

# 📝 Attendance function
def mark_attendance(name):
    if not os.path.exists("attendance.csv"):
        df = pd.DataFrame(columns=["Record"])
        df.to_csv("attendance.csv", index=False)

    df = pd.read_csv("attendance.csv")

    if "Record" not in df.columns:
        df = pd.DataFrame(columns=["Record"])

    now = datetime.now()
    date_str = now.strftime("%d/%m/%Y")
    record = f"{name} is present on {date_str}"

    if record in df["Record"].values:
        st.warning(f"⚠️ {name}, you are already present today ✅")
        return  

    df.loc[len(df)] = [record]
    df.to_csv("attendance.csv", index=False)
    st.success(f"✅ Attendance marked: {record}")

# 🎥 Streamlit UI
st.title("🎓 Face Recognition Attendance System")

if st.button("📸 Capture Photo & Mark Attendance", key="capture_once"):
    cap = cv2.VideoCapture(0)

    ret, frame = cap.read()
    if not ret:
        st.error("❌ Camera not detected!")
    else:
        st.image(frame, channels="BGR", caption="Captured Photo")

        # Process frame
        small_frame = cv2.resize(frame, (0,0), None, 0.25, 0.25)
        small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)

        faces_cur_frame = face_recognition.face_locations(small_frame)
        encodes_cur_frame = face_recognition.face_encodings(small_frame, faces_cur_frame)

        if len(encodes_cur_frame) == 0:
            st.error("❌ No face detected. Please try again.")
        else:
            for encodeFace in encodes_cur_frame:
                matches = face_recognition.compare_faces(encode_list_known, encodeFace)
                faceDis = face_recognition.face_distance(encode_list_known, encodeFace)
                matchIndex = np.argmin(faceDis)

                if matches[matchIndex]:
                    name = student_names[matchIndex].upper()
                    mark_attendance(name)
                else:
                    st.error("❌ Unknown face detected!")

    cap.release()
    cv2.destroyAllWindows()
