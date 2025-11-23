# 📸 FaceRec Attendance System

![Python](https://img.shields.io/badge/Python-3.x-blue?style=for-the-badge&logo=python)
![OpenCV](https://img.shields.io/badge/OpenCV-Computer%20Vision-green?style=for-the-badge&logo=opencv)
![GUI](https://img.shields.io/badge/GUI-CustomTkinter-orange?style=for-the-badge)

A robust, real-time attendance logging application built with Python. This system uses facial recognition to identify registered users via webcam and logs their attendance into a CSV file, wrapped in a modern, dark-themed user interface.

## ✨ Key Features

* **Real-Time Recognition:** Instantly detects and identifies faces from a live video feed.
* **Smart Attendance Logging:** Automatically logs "Name" and "Timestamp" to a CSV file (prevents duplicate entries for the same day).
* **Robust Registration Module:** Features a dedicated registration window that captures **3 different angles** of a user's face to improve recognition accuracy.
* **Modern UI:** Built with `CustomTkinter` for a clean, professional dark-mode interface.
* **Threaded Performance:** Runs video capture and GUI on separate threads to ensure the application never freezes.
* **Live Feedback:** Visual bounding boxes and status updates (Green = Known, Red = Unknown).

## 🛠️ Tech Stack

* **Python 3.x**
* **Face Recognition:** `face_recognition` (dlib)
* **Computer Vision:** `opencv-python`
* **GUI:** `customtkinter`, `CTkMessagebox`
* **Data Handling:** `numpy`, `pandas` (CSV)

## 🚀 Installation & Setup

1.  **Clone the repository**
    ```bash
    git clone [https://github.com/YOUR_USERNAME/REPO_NAME.git](https://github.com/YOUR_USERNAME/REPO_NAME.git)
    cd REPO_NAME
    ```

2.  **Install Dependencies**
    *Note: You may need C++ Build Tools installed (Visual Studio) for `face_recognition` / `dlib`.*
    ```bash
    pip install opencv-python numpy face-recognition customtkinter CTkMessagebox packaging pillow
    ```

3.  **Run the Application**
    ```bash
    python main.py
    ```

## 📖 How to Use

1.  **Start the App:** Run the script to open the main dashboard.
2.  **Register a User:** * Click **"➕ Register New Face"**.
    * Enter the user's name (e.g., `Jane_Doe`).
    * Follow the prompts to capture **3 snapshots** (move your head slightly between shots for better accuracy).
3.  **Mark Attendance:** * Simply stand in front of the camera. 
    * When your face is recognized, the box turns **Green** and your attendance is logged to `Attendance.csv`.

## 📂 Project Structure

├── KnownFaces/        # Stores registered user images
├── Attendance.csv     # Daily attendance logs
├── main.py            # Main application script
└── README.md          # Documentation

## 🤝 Contributing
Contributions are welcome! Please feel free to submit a Pull Request.

## 📄 License
This project is open source and available under the [MIT License](LICENSE).
