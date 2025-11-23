import face_recognition
import cv2
import numpy as np
import os
import threading
import queue
import time
from datetime import datetime
import customtkinter as ctk
from PIL import Image, ImageTk
import warnings
from CTkMessagebox import CTkMessagebox

# --- Ignore annoying warnings that might pop up ---
warnings.filterwarnings("ignore", category=UserWarning)

# --- Constants and configuration ---
KNOWN_FACES_DIR = "KnownFaces"
ATTENDANCE_LOG_FILE = "Attendance.csv"
RECOGNITION_TOLERANCE = 0.6
FRAME_RESIZE_FACTOR = 0.25
THEME_MODE = "Dark"
THEME_COLOR = "blue"
TOTAL_PHOTOS_REQUIRED = 3  # Human tends to put a comment just to remember magic numbers

# Make sure the folder exists, or else face saving will fail
os.makedirs(KNOWN_FACES_DIR, exist_ok=True)

# Global CTk theme setup
ctk.set_appearance_mode(THEME_MODE)
ctk.set_default_color_theme(THEME_COLOR)


# --- Face Data Management ---
class FaceDataManager:
    """Keeps track of known faces, encodings, and daily attendance."""
    def __init__(self):
        self.known_encodings = []
        self.known_names = []
        self.logged_people = set()
        self.load_today_log()
        self.load_known_faces()

    def load_known_faces(self):
        """Scan the folder and load all face encodings."""
        print("STATUS: Loading known faces... please wait")
        temp_encodings = []
        temp_names = []

        if not os.path.exists(KNOWN_FACES_DIR):
            print("INFO: KnownFaces folder missing, creating new one")
            os.makedirs(KNOWN_FACES_DIR)
            return

        for fname in os.listdir(KNOWN_FACES_DIR):
            if not fname.lower().endswith(('.jpg', '.jpeg', '.png')):
                continue

            # Simplistic name cleanup
            name = os.path.splitext(fname)[0].replace('_', ' ')
            if "_" in name and name.split('_')[-1].isdigit():
                name = "_".join(name.split('_')[:-1])

            fpath = os.path.join(KNOWN_FACES_DIR, fname)

            try:
                img = face_recognition.load_image_file(fpath)
                encs = face_recognition.face_encodings(img)
                if not encs:
                    print(f"WARNING: No face found in {fname}")
                    continue

                temp_encodings.append(encs[0])
                temp_names.append(name)
                print(f"INFO: Loaded {name}")
            except Exception as e:
                print(f"ERROR: Could not load {fname}: {e}")

        self.known_encodings = temp_encodings
        self.known_names = temp_names

    def load_today_log(self):
        """Load today's attendance into memory."""
        self.logged_people.clear()
        today_str = datetime.now().strftime("%Y-%m-%d")

        if not os.path.exists(ATTENDANCE_LOG_FILE):
            return

        try:
            with open(ATTENDANCE_LOG_FILE, 'r') as f:
                header = next(f, None)
                if not header:
                    return

                for line in f:
                    parts = line.strip().split(',')
                    if len(parts) >= 2 and parts[1].startswith(today_str):
                        self.logged_people.add(parts[0])
        except Exception as e:
            print(f"LOG ERROR: Couldn't read attendance CSV: {e}")

    def mark_attendance(self, name):
        """Mark a person as attended if not already logged today."""
        if name not in self.logged_people:
            ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            mode = 'a' if os.path.exists(ATTENDANCE_LOG_FILE) else 'w'
            with open(ATTENDANCE_LOG_FILE, mode) as f:
                if mode == 'w':
                    f.write("Name,Timestamp\n")
                f.write(f"{name},{ts}\n")
            self.logged_people.add(name)
            return True
        return False

    def save_new_face(self, name, frames):
        """Save multiple frames for a new face and reload database."""
        if not frames:
            raise ValueError("No frames to save!")

        for i, frm in enumerate(frames):
            fname = f"{name}_{i+1}.jpg"
            path = os.path.join(KNOWN_FACES_DIR, fname)
            bgr = cv2.cvtColor(frm, cv2.COLOR_RGB2BGR)
            cv2.imwrite(path, bgr)
        # reload encodings
        self.load_known_faces()


# --- Camera Thread ---
class CameraThread(threading.Thread):
    """Handles camera capture and face recognition in a separate thread."""
    def __init__(self, data_manager, frame_queue):
        super().__init__()
        self.data_manager = data_manager
        self.frame_queue = frame_queue
        self.running = True
        self.cap = None
        self.latest_frame = None
        self.daemon = True

    def init_camera(self):
        """Try camera 0 then 1"""
        if self.cap:
            self.cap.release()
        for idx in [0, 1]:
            self.cap = cv2.VideoCapture(idx)
            if self.cap.isOpened():
                print(f"INFO: Camera {idx} opened")
                return True
        print("FATAL: No camera detected")
        self.running = False
        return False

    def run(self):
        if not self.init_camera():
            return
        time.sleep(1)

        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                time.sleep(0.1)
                continue

            self.latest_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Downscale for faster processing
            small = cv2.resize(frame, (0, 0), fx=FRAME_RESIZE_FACTOR, fy=FRAME_RESIZE_FACTOR)
            rgb_small = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)

            faces = face_recognition.face_locations(rgb_small, model="hog")
            encodings = face_recognition.face_encodings(rgb_small, faces)

            names = []
            for enc in encodings:
                name = "Unknown"
                if self.data_manager.known_encodings:
                    matches = face_recognition.compare_faces(self.data_manager.known_encodings, enc, tolerance=RECOGNITION_TOLERANCE)
                    distances = face_recognition.face_distance(self.data_manager.known_encodings, enc)
                    if len(distances) > 0:
                        best_idx = np.argmin(distances)
                        if matches[best_idx]:
                            name = self.data_manager.known_names[best_idx]
                            self.data_manager.mark_attendance(name)
                names.append(name)

            # Draw boxes
            for (top, right, bottom, left), n in zip(faces, names):
                top, right, bottom, left = [int(c / FRAME_RESIZE_FACTOR) for c in (top, right, bottom, left)]
                clr = (0, 255, 0) if n != "Unknown" else (0, 0, 255)
                cv2.rectangle(frame, (left, top), (right, bottom), clr, 2)
                cv2.rectangle(frame, (left, bottom - 30), (right, bottom), clr, cv2.FILLED)
                cv2.putText(frame, n, (left + 6, bottom - 6), cv2.FONT_HERSHEY_DUPLEX, 0.8, (255, 255, 255), 1)

            if not self.frame_queue.full():
                self.frame_queue.put(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))

            time.sleep(0.01)

        if self.cap:
            self.cap.release()

    def stop(self):
        self.running = False

    def take_snapshot(self):
        if self.latest_frame is not None:
            return self.latest_frame.copy()
        return None


# --- Registration Window ---
class RegistrationWindow(ctk.CTkToplevel):
    def __init__(self, master, data_manager, camera_thread):
        super().__init__(master)
        self.data_manager = data_manager
        self.camera_thread = camera_thread

        self.title("Register New User")
        self.geometry("450x350")
        self.transient(master)
        self.grab_set()

        self.captured_frames = []
        self.capture_count = 0
        self.total_captures = TOTAL_PHOTOS_REQUIRED

        frm = ctk.CTkFrame(self)
        frm.pack(padx=20, pady=20, fill="both", expand=True)

        ctk.CTkLabel(frm, text="New User Name (underscores only):", font=ctk.CTkFont(size=14, weight="bold")).pack(pady=(10, 5))
        self.name_entry = ctk.CTkEntry(frm, width=350, placeholder_text="e.g., Jane_Doe")
        self.name_entry.pack(pady=5)

        self.status_var = ctk.StringVar(value=f"Click 'Capture' {self.total_captures} times.\nMove head slightly each shot.")
        ctk.CTkLabel(frm, textvariable=self.status_var, font=ctk.CTkFont(size=12, slant="italic")).pack(pady=(5, 15))

        self.capture_btn = ctk.CTkButton(frm, text=f"Capture Photo 1/{self.total_captures}",
                                         command=self.process_capture, fg_color="#3498DB", hover_color="#2980B9")
        self.capture_btn.pack(pady=(10, 5))

        ctk.CTkButton(frm, text="Close", command=self.destroy, fg_color="#555555").pack(pady=5)

    def process_capture(self):
        """Grab a snapshot from camera and handle multi-photo logic."""
        name = self.name_entry.get().strip()
        if not name or ' ' in name or not name[0].isalpha():
            CTkMessagebox(title="Input Error", message="Name must start with a letter and use underscores.", icon="warning", master=self)
            return

        self.capture_btn.configure(state="disabled")
        snap = self.camera_thread.take_snapshot()
        if snap is None:
            CTkMessagebox(title="Camera Error", message="Camera not ready.", icon="cancel", master=self)
            self.capture_btn.configure(state="normal")
            return

        self.captured_frames.append(snap)
        self.capture_count += 1

        if self.capture_count < self.total_captures:
            self.status_var.set(f"✅ Photo {self.capture_count}/{self.total_captures} captured. Turn head slightly and continue.")
            self.capture_btn.configure(text=f"Capture Photo {self.capture_count+1}/{self.total_captures}", state="normal")
        else:
            self.status_var.set(f"✅ {self.total_captures}/{self.total_captures} photos captured. Processing...")
            self.capture_btn.configure(text="Processing...", state="disabled")
            self.master.after(100, lambda: self.save_all_frames(name))

    def save_all_frames(self, name):
        try:
            self.data_manager.save_new_face(name, self.captured_frames)
            self.master.reload_known_faces()
            CTkMessagebox(title="Success!", message=f"{name} registered with {len(self.captured_frames)} photos.", icon="check", master=self)
            self.destroy()
        except Exception as e:
            CTkMessagebox(title="Error", message=f"Could not save: {e}", icon="cancel", master=self)
            self.capture_btn.configure(state="normal")


# --- Main Application ---
class CTKAttendanceApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("Face Attendance System")
        self.geometry("800x750")
        self.resizable(False, False)

        self.data_manager = FaceDataManager()
        self.frame_queue = queue.Queue(maxsize=1)

        self.camera_thread = CameraThread(self.data_manager, self.frame_queue)
        self.camera_thread.start()

        self.status_text = ctk.StringVar(value=f"STATUS: Ready. {len(self.data_manager.known_names)} users loaded.")
        self.log_count_text = ctk.StringVar()
        self._prev_log_count = len(self.data_manager.logged_people)

        self._setup_ui()
        self.after(10, self._update_frame)

    def reload_known_faces(self):
        try:
            self.data_manager.load_known_faces()
            self.status_text.set(f"Database reloaded. Tracking {len(self.data_manager.known_names)} users.")
        except:
            self.status_text.set("ERROR: Failed to reload database.")

    def open_registration_window(self):
        RegistrationWindow(self, self.data_manager, self.camera_thread)

    def _setup_ui(self):
        main_frame = ctk.CTkFrame(self, corner_radius=10, fg_color="#333333")
        main_frame.pack(padx=20, pady=20, fill="both", expand=True)

        ctk.CTkLabel(main_frame, text="FACIAL ATTENDANCE CHECK-IN", font=ctk.CTkFont(family="Arial", size=22, weight="bold")).pack(pady=(20,10))

        self.video_label = ctk.CTkLabel(main_frame, text="Camera Feed", bg_color="black", width=640, height=480, corner_radius=5)
        self.video_label.pack(pady=10)

        status_frame = ctk.CTkFrame(main_frame, corner_radius=8, fg_color="#3498DB")
        status_frame.pack(pady=(10,5), fill="x", padx=20)

        self.status_frame = status_frame
        ctk.CTkLabel(status_frame, textvariable=self.status_text, font=ctk.CTkFont(size=14, weight="bold"), text_color="white").pack(side="left", padx=15, pady=10)
        ctk.CTkLabel(status_frame, textvariable=self.log_count_text, font=ctk.CTkFont(size=14), text_color="white").pack(side="right", padx=15, pady=10)
        self._update_logged_count_display()

        btn_frame = ctk.CTkFrame(main_frame, fg_color="transparent")
        btn_frame.pack(pady=15)

        ctk.CTkButton(btn_frame, text="➕ REGISTER NEW FACE", command=self.open_registration_window, font=ctk.CTkFont(size=16, weight="bold"), height=40, fg_color="#27AE60", hover_color="#229954").pack(side="left", padx=10)
        ctk.CTkButton(btn_frame, text="EXIT APPLICATION", command=self.on_closing, font=ctk.CTkFont(size=16, weight="bold"), height=40, fg_color="#E74C3C", hover_color="#C0392B").pack(side="right", padx=10)

    def _update_logged_count_display(self):
        self.log_count_text.set(f"TODAY LOGGED: {len(self.data_manager.logged_people)}")

    def _animate_status(self, target_color="#3498DB"):
        try:
            self.status_frame.configure(fg_color=target_color)
            self.after(1000, lambda: self.status_frame.configure(fg_color="#3498DB"))
        except:
            pass

    def _update_frame(self):
        try:
            rgb_frame = self.frame_queue.get_nowait()
            cur_count = len(self.data_manager.logged_people)
            if self._prev_log_count != cur_count:
                self.status_text.set(f"CHECK-IN SUCCESSFUL! Total: {cur_count}")
                self._animate_status("#2ECC71")
                self._prev_log_count = cur_count
                self._update_logged_count_display()

            img = Image.fromarray(rgb_frame).resize((640,480))
            self.ctk_image = ImageTk.PhotoImage(img)
            self.video_label.configure(image=self.ctk_image)
            self.video_label.image = self.ctk_image
        except queue.Empty:
            pass
        except Exception as e:
            print(f"Frame error: {e}")
        self.after(30, self._update_frame)

    def on_closing(self):
        print("INFO: Stopping camera thread...")
        self.camera_thread.stop()
        self.after(500, self.destroy)


if __name__ == "__main__":
    try:
        app = CTKAttendanceApp()
        app.mainloop()
    except Exception as e:
        print(f"CRITICAL FAILURE: {e}")
