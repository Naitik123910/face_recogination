import os
import streamlit as st
# Load external CSS
def load_css(file_name):
    with open(file_name) as f:
        st.markdown(f"<style>{f.read()}</style>", unsafe_allow_html=True)

load_css("style.css")

import cv2
import numpy as np
import pandas as pd
from datetime import datetime
import face_recognition
import matplotlib.pyplot as plt

# -------------------------------
# Constants
# -------------------------------
PHOTOS_FOLDER = "photos"
CSV_FILE = "attendance.csv"

# -------------------------------
# Ensure folders and CSV exist
# -------------------------------
if not os.path.exists(PHOTOS_FOLDER):
    os.makedirs(PHOTOS_FOLDER)

if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("Name,Date,Time\n")

# -------------------------------
# Load student images and encodings
# -------------------------------
student_names = []
known_encodings = []

for file in os.listdir(PHOTOS_FOLDER):
    if file.lower().endswith((".jpg", ".jpeg", ".png")):
        name = os.path.splitext(file)[0]
        student_names.append(name)
        img_path = os.path.join(PHOTOS_FOLDER, file)
        image = face_recognition.load_image_file(img_path)
        encodings = face_recognition.face_encodings(image)
        known_encodings.append(encodings[0] if encodings else None)

# -------------------------------
# Function to mark attendance
# -------------------------------
def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    if not df.empty and ((df["Name"] == name) & (df["Date"] == date_str)).any():
        return False  # Already marked today

    with open(CSV_FILE, "a") as f:
        f.write(f"{name},{date_str},{time_str}\n")
    return True  # Newly marked

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Face Recognition Attendance System")

# -------- Take Attendance (Multiple Faces) --------
if st.button("Take Attendance"):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        st.error("Cannot access camera.")
    else:
        ret, frame = cap.read()
        cap.release()

        if not ret:
            st.error("Failed to capture frame from camera.")
        else:
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            face_locations = face_recognition.face_locations(rgb_frame)
            face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

            if len(face_encodings) == 0:
                st.error("No face detected ❌")
            else:
                # Load today's attendance
                try:
                    df_today = pd.read_csv(CSV_FILE)
                    df_today = df_today[df_today["Date"] == datetime.now().strftime("%Y-%m-%d")]
                except Exception:
                    df_today = pd.DataFrame(columns=["Name", "Date", "Time"])

                detected_students = []
                already_marked = []

                for face_encoding in face_encodings:
                    matches = face_recognition.compare_faces(known_encodings, face_encoding)
                    distances = face_recognition.face_distance(known_encodings, face_encoding)
                    best_idx = np.argmin(distances) if len(distances) > 0 else -1

                    if best_idx >= 0 and matches[best_idx]:
                        student_name = student_names[best_idx]
                        if student_name not in df_today["Name"].values:
                            mark_attendance(student_name)
                            detected_students.append(student_name)
                        else:
                            already_marked.append(student_name)

                if detected_students:
                    st.success(f"Attendance marked for: {', '.join(detected_students)} ✅")
                if already_marked:
                    st.warning(f"Already marked today: {', '.join(already_marked)} ⚠")
                if not detected_students and not already_marked:
                    st.error("Unknown face(s) detected ❌")

# -------- Attendance Viewer --------
st.markdown("---")
st.subheader("📒 Attendance Viewer")

# Read CSV safely
try:
    df_all = pd.read_csv(CSV_FILE)
except Exception:
    df_all = pd.DataFrame(columns=["Name", "Date", "Time"])

# Ensure Date column exists
if "Date" in df_all.columns:
    df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
    df_all = df_all.dropna(subset=["Date"])
else:
    df_all["Date"] = pd.to_datetime([])

# Unique dates
unique_dates = sorted(df_all["Date"].dt.date.unique()) if not df_all.empty else []

# -------- Safe placeholder container --------
viewer_container = st.empty()

with viewer_container:
    if unique_dates:
        selected_date = st.select_slider(
            "Select date to view:",
            options=unique_dates,
            value=unique_dates[-1]
        )

        df_selected = df_all[df_all["Date"].dt.date == selected_date]

        if not df_selected.empty:
            st.write(f"Attendance for *{selected_date}*")
            st.dataframe(df_selected.reset_index(drop=True))

            # Pie chart summary
            total_students = len(student_names)
            present = df_selected["Name"].nunique()
            absent = total_students - present

            if total_students > 0:
                fig, ax = plt.subplots()
                ax.pie([present, absent], labels=["Present", "Absent"],
                       autopct="%1.1f%%", startangle=90)
                ax.axis("equal")
                st.pyplot(fig)
        else:
            st.info(f"No records for {selected_date}")
    else:
        st.info("No attendance records yet. Slider will not be displayed.")

# -------- Full attendance history --------
if st.checkbox("Show full attendance history"):
    st.dataframe(df_all.reset_index(drop=True))