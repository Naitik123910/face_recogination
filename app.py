import streamlit as st
import cv2
import numpy as np
import face_recognition
import pandas as pd
import os
from datetime import datetime

PHOTOS_FOLDER = "photos"
CSV_FILE = "attendance.csv"

# -------------------------------
# Load student images
# -------------------------------
images = []
student_names = []

if not os.path.exists(PHOTOS_FOLDER):
    os.makedirs(PHOTOS_FOLDER)

for file in os.listdir(PHOTOS_FOLDER):
    filepath = os.path.join(PHOTOS_FOLDER, file)
    img = cv2.imread(filepath)
    if img is not None:
        images.append(img)
        student_names.append(os.path.splitext(file)[0])

def find_encodings(images):
    encode_list = []
    for img in images:
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        enc = face_recognition.face_encodings(img_rgb)
        if len(enc) > 0:
            encode_list.append(enc[0])
    return encode_list

encode_list_known = find_encodings(images)

# -------------------------------
# Ensure CSV exists
# -------------------------------
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("Name,Date,Time\n")

# -------------------------------
# Mark attendance
# -------------------------------
def mark_attendance(name):
    date_str = datetime.now().strftime("%Y-%m-%d")
    time_str = datetime.now().strftime("%H:%M:%S")

    try:
        df = pd.read_csv(CSV_FILE)
    except Exception:
        df = pd.DataFrame(columns=["Name", "Date", "Time"])

    if not df.empty and ((df["Name"] == name) & (df["Date"] == date_str)).any():
        st.warning(f"{name} already marked present today!")
        return False

    with open(CSV_FILE, "a") as f:
        f.write(f"{name},{date_str},{time_str}\n")

    st.success(f"Attendance marked for {name} ✅")
    return True

# -------------------------------
# Streamlit UI
# -------------------------------
st.title("🎓 Face Recognition Attendance System")

if st.button("Take Attendance"):
    cap = cv2.VideoCapture(0)
    ret, frame = cap.read()
    cap.release()

    if not ret:
        st.error("Failed to access the camera.")
    else:
        face_locs = face_recognition.face_locations(frame)
        encodes_cur = face_recognition.face_encodings(frame, face_locs)

        if not encodes_cur:
            st.error("No face detected. Please try again.")
        else:
            for encode_face, face_loc in zip(encodes_cur, face_locs):
                matches = face_recognition.compare_faces(encode_list_known, encode_face)
                face_dist = face_recognition.face_distance(encode_list_known, encode_face)

                if len(face_dist) > 0:
                    match_index = np.argmin(face_dist)
                    if matches[match_index]:
                        name = student_names[match_index].upper()
                        mark_attendance(name)
                    else:
                        st.error("Unknown face detected ❌")

        # ❌ removed st.image(frame, ...) — no image shown

# -------------------------------
# -------------------------------
# Attendance Viewer
# -------------------------------
import matplotlib.pyplot as plt

st.markdown("---")
st.subheader("📒 Attendance Viewer")

# Ensure file exists
if not os.path.isfile(CSV_FILE):
    with open(CSV_FILE, "w") as f:
        f.write("Name,Date,Time\n")

try:
    df_all = pd.read_csv(CSV_FILE)
except Exception:
    df_all = pd.DataFrame(columns=["Name", "Date", "Time"])

if df_all.empty:
    st.info("No attendance records found yet.")
else:
    # Parse dates
    if "Date" in df_all.columns:
        df_all["Date"] = pd.to_datetime(df_all["Date"], errors="coerce")
        df_all = df_all.dropna(subset=["Date"]).copy()
    else:
        st.error("CSV is missing 'Date' column.")
        st.stop()

    unique_dates = sorted(df_all["Date"].dt.date.unique())

    if len(unique_dates) == 1:
        selected_date = unique_dates[0]
        st.info(f"Showing attendance for **{selected_date}**")
    else:
        selected_date = st.select_slider(
            "Select date to view:",
            options=unique_dates,
            value=unique_dates[-1],  # default latest
        )

    df_selected = df_all[df_all["Date"].dt.date == selected_date].copy()

    if df_selected.empty:
        st.warning(f"No records for {selected_date}")
    else:
        st.write(f"Attendance for **{selected_date}**")
        st.dataframe(df_selected.reset_index(drop=True))

        # -------------------------------
        # Pie Chart for Present vs Absent
        # -------------------------------
        total_students = len(student_names)
        present_students = df_selected["Name"].nunique()
        absent_students = total_students - present_students

        st.write(f"📊 Attendance Summary: {present_students}/{total_students} present")

        if total_students > 0:
            labels = ["Present", "Absent"]
            values = [present_students, absent_students]

            fig, ax = plt.subplots()
            ax.pie(values, labels=labels, autopct="%1.1f%%", startangle=90)
            ax.axis("equal")  # Make it a circle
            st.pyplot(fig)
        else:
            st.warning("No student photos found in 'photos/' folder.")

# Optional: full history
if st.checkbox("Show full attendance history"):
    st.dataframe(df_all.reset_index(drop=True))
