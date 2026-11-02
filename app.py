import csv
import os
import threading
import time
import tkinter as tk
from tkinter import ttk, messagebox
from collections import deque, Counter

import cv2
import numpy as np
import mediapipe as mp
from PIL import Image, ImageTk
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib

APP_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_FILE = os.path.join(APP_DIR, "data", "sign_samples.csv")
MODEL_FILE = os.path.join(APP_DIR, "models", "signvision_model.joblib")

ALPHABETS = list("ABCDEFGHIJKLMNOPQRSTUVWXYZ")
WORD_LABELS = ["HELLO", "THANK YOU", "YES", "NO", "PLEASE", "SORRY", "HELP", "GOOD", "STOP", "WATER"]
ALL_LABELS = ALPHABETS + WORD_LABELS

WORD_MEANINGS = {
    "HELLO": "A greeting used to welcome or acknowledge someone.",
    "THANK YOU": "Used to express gratitude.",
    "YES": "An affirmative response.",
    "NO": "A negative response.",
    "PLEASE": "Used to make a polite request.",
    "SORRY": "Used to express apology.",
    "HELP": "Used to ask for assistance.",
    "GOOD": "Used to express that something is good.",
    "STOP": "Used to ask someone to stop.",
    "WATER": "Refers to water."
}

def extract_features(hand_landmarks):
    """Return 63 wrist-relative, scale-normalized hand landmark features."""
    pts = np.array([[p.x, p.y, p.z] for p in hand_landmarks.landmark], dtype=np.float32)
    pts -= pts[0]  # wrist-relative coordinates
    scale = np.max(np.linalg.norm(pts[:, :2], axis=1))
    if scale > 1e-6:
        pts /= scale
    return pts.flatten()

class SignVisionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("SignVision AI | Indian Sign Language Recognition")
        self.root.geometry("1180x760")
        self.root.minsize(980, 650)
        self.root.configure(bg="#0b1220")

        self.mp_hands = mp.solutions.hands
        self.hands = self.mp_hands.Hands(
            static_image_mode=False,
            max_num_hands=1,
            min_detection_confidence=0.65,
            min_tracking_confidence=0.60
        )
        self.mp_draw = mp.solutions.drawing_utils
        self.cap = None
        self.current_features = None
        self.current_prediction = None
        self.current_confidence = 0.0
        self.capture_remaining = 0
        self.capture_label = None
        self.prediction_history = deque(maxlen=9)
        self.model = None
        self.class_names = []
        self.sentence = ""
        self.camera_running = False

        self._style()
        self._build_ui()
        self.load_model()
        self.open_camera()
        self.update_frame()
        self.root.protocol("WM_DELETE_WINDOW", self.close)

    def _style(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except Exception:
            pass
        style.configure("TCombobox", fieldbackground="#172235", background="#172235",
                        foreground="#f8fafc", arrowcolor="#f8fafc")
        style.configure("TButton", font=("Segoe UI", 10, "bold"), padding=8)

    def _build_ui(self):
        header = tk.Frame(self.root, bg="#111c2e", padx=22, pady=15)
        header.pack(fill="x")
        tk.Label(header, text="SIGNVISION AI", bg="#111c2e", fg="#f8fafc",
                 font=("Segoe UI", 22, "bold")).pack(side="left")
        tk.Label(header, text="INDIAN SIGN LANGUAGE • LIVE RECOGNITION PROTOTYPE",
                 bg="#111c2e", fg="#93c5fd", font=("Segoe UI", 9, "bold")).pack(side="left", padx=18)

        body = tk.Frame(self.root, bg="#0b1220", padx=18, pady=16)
        body.pack(fill="both", expand=True)

        left = tk.Frame(body, bg="#111c2e", padx=12, pady=12)
        left.pack(side="left", fill="both", expand=True)
        right = tk.Frame(body, bg="#111c2e", padx=16, pady=14, width=330)
        right.pack(side="right", fill="y", padx=(14, 0))
        right.pack_propagate(False)

        tk.Label(left, text="LIVE CAMERA", bg="#111c2e", fg="#e2e8f0",
                 font=("Segoe UI", 11, "bold")).pack(anchor="w", pady=(0, 8))
        self.video_label = tk.Label(left, bg="#020617", text="Starting webcam...",
                                    fg="#94a3b8", font=("Segoe UI", 14))
        self.video_label.pack(fill="both", expand=True)

        self.camera_status = tk.Label(left, text="Camera: starting", bg="#111c2e",
                                      fg="#fbbf24", font=("Segoe UI", 9))
        self.camera_status.pack(anchor="w", pady=(8, 0))

        tk.Label(right, text="PREDICTION", bg="#111c2e", fg="#93c5fd",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.pred_label = tk.Label(right, text="—", bg="#111c2e", fg="#f8fafc",
                                   font=("Segoe UI", 34, "bold"))
        self.pred_label.pack(anchor="w", pady=(2, 0))
        self.conf_label = tk.Label(right, text="Confidence: —", bg="#111c2e",
                                   fg="#cbd5e1", font=("Segoe UI", 10))
        self.conf_label.pack(anchor="w")
        self.meaning_label = tk.Label(right, text="Show a hand sign to the camera.",
                                      bg="#111c2e", fg="#94a3b8", wraplength=285,
                                      justify="left", font=("Segoe UI", 9))
        self.meaning_label.pack(anchor="w", pady=(8, 14))

        tk.Frame(right, bg="#334155", height=1).pack(fill="x", pady=6)
        tk.Label(right, text="TRAIN YOUR SIGN CLASSES", bg="#111c2e", fg="#93c5fd",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w", pady=(4, 7))
        tk.Label(right, text="Choose a label, show that sign, then capture samples.",
                 bg="#111c2e", fg="#cbd5e1", wraplength=285, justify="left",
                 font=("Segoe UI", 9)).pack(anchor="w")
        self.label_var = tk.StringVar(value="A")
        self.label_combo = ttk.Combobox(right, textvariable=self.label_var,
                                        values=ALL_LABELS, state="readonly", width=24)
        self.label_combo.pack(anchor="w", pady=8)
        self.capture_btn = ttk.Button(right, text="Capture 25 frames", command=self.start_capture)
        self.capture_btn.pack(fill="x", pady=3)
        self.capture_status = tk.Label(right, text="Samples captured: 0",
                                       bg="#111c2e", fg="#cbd5e1", font=("Segoe UI", 9))
        self.capture_status.pack(anchor="w", pady=(4, 8))
        ttk.Button(right, text="Train / Update Model", command=self.train_model).pack(fill="x", pady=3)
        ttk.Button(right, text="Reload Saved Model", command=self.load_model).pack(fill="x", pady=3)

        tk.Frame(right, bg="#334155", height=1).pack(fill="x", pady=9)
        tk.Label(right, text="SENTENCE BUILDER", bg="#111c2e", fg="#93c5fd",
                 font=("Segoe UI", 10, "bold")).pack(anchor="w")
        self.sentence_box = tk.Text(right, height=4, bg="#172235", fg="#f8fafc",
                                    insertbackground="white", relief="flat", wrap="word",
                                    font=("Segoe UI", 10))
        self.sentence_box.pack(fill="x", pady=7)
        row = tk.Frame(right, bg="#111c2e")
        row.pack(fill="x")
        ttk.Button(row, text="Add Prediction", command=self.add_prediction).pack(side="left", expand=True, fill="x")
        ttk.Button(row, text="Clear", command=self.clear_sentence).pack(side="left", padx=(6, 0))

        tk.Frame(right, bg="#334155", height=1).pack(fill="x", pady=9)
        ttk.Button(right, text="Sign Dictionary", command=self.show_dictionary).pack(fill="x")
        self.model_status = tk.Label(right, text="Model: not trained yet",
                                     bg="#111c2e", fg="#fbbf24", font=("Segoe UI", 9),
                                     wraplength=285, justify="left")
        self.model_status.pack(anchor="w", pady=(10, 0))

        footer = tk.Label(self.root,
            text="Prototype note: train with your own ISL samples before relying on predictions. Static hand-pose classification only.",
            bg="#08101d", fg="#94a3b8", font=("Segoe UI", 8), pady=8)
        footer.pack(fill="x")

    def open_camera(self):
        for index in (0, 1):
            cap = cv2.VideoCapture(index, cv2.CAP_DSHOW)
            if cap.isOpened():
                self.cap = cap
                self.camera_running = True
                self.camera_status.config(text=f"Camera: connected (device {index})", fg="#86efac")
                return
            cap.release()
        self.camera_status.config(text="Camera not found. Check webcam permission.", fg="#fca5a5")

    def update_frame(self):
        if self.cap is not None and self.cap.isOpened():
            ok, frame = self.cap.read()
            if ok:
                frame = cv2.flip(frame, 1)
                rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                result = self.hands.process(rgb)
                self.current_features = None

                if result.multi_hand_landmarks:
                    hand = result.multi_hand_landmarks[0]
                    self.mp_draw.draw_landmarks(frame, hand, self.mp_hands.HAND_CONNECTIONS)
                    self.current_features = extract_features(hand)
                    if self.capture_remaining > 0 and self.capture_label:
                        self.save_sample(self.capture_label, self.current_features)
                        self.capture_remaining -= 1
                        self.capture_status.config(
                            text=f"Capturing {self.capture_label}: {25-self.capture_remaining}/25"
                        )
                        if self.capture_remaining == 0:
                            self.capture_status.config(text=f"Finished capturing {self.capture_label}")
                    self.predict_current()
                else:
                    self.prediction_history.clear()
                    self.current_prediction = None
                    self.pred_label.config(text="—")
                    self.conf_label.config(text="Confidence: —")
                    self.meaning_label.config(text="No hand detected. Keep your hand inside the camera view.")

                rgb_show = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                image = Image.fromarray(rgb_show)
                image.thumbnail((820, 560))
                photo = ImageTk.PhotoImage(image=image)
                self.video_label.configure(image=photo, text="")
                self.video_label.image = photo
        self.root.after(20, self.update_frame)

    def save_sample(self, label, features):
        os.makedirs(os.path.dirname(DATA_FILE), exist_ok=True)
        new_file = not os.path.exists(DATA_FILE)
        with open(DATA_FILE, "a", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            if new_file:
                writer.writerow(["label"] + [f"f{i}" for i in range(63)])
            writer.writerow([label] + [float(x) for x in features])

    def start_capture(self):
        if self.current_features is None:
            messagebox.showwarning("No hand detected", "Show your hand clearly in the webcam first.")
            return
        self.capture_label = self.label_var.get()
        self.capture_remaining = 25
        self.capture_status.config(text=f"Capturing {self.capture_label}: 0/25")
        self.capture_btn.config(state="disabled")
        self.root.after(2600, lambda: self.capture_btn.config(state="normal"))

    def train_model(self):
        if not os.path.exists(DATA_FILE):
            messagebox.showwarning("No data", "Capture samples for at least two labels first.")
            return
        try:
            import pandas as pd
            df = pd.read_csv(DATA_FILE)
            counts = df["label"].value_counts()
            usable = counts[counts >= 3].index
            df = df[df["label"].isin(usable)]
            if df["label"].nunique() < 2:
                messagebox.showwarning("More samples needed",
                    "Capture at least 3 samples each for two or more different labels.")
                return
            X = df.drop(columns=["label"]).values.astype(np.float32)
            y = df["label"].astype(str).values
            model = RandomForestClassifier(n_estimators=250, max_depth=18,
                                           class_weight="balanced", random_state=42)
            model.fit(X, y)
            os.makedirs(os.path.dirname(MODEL_FILE), exist_ok=True)
            joblib.dump(model, MODEL_FILE)
            self.model = model
            self.class_names = list(model.classes_)
            self.model_status.config(
                text=f"Model: trained • {len(y)} samples • {len(self.class_names)} classes",
                fg="#86efac"
            )
            messagebox.showinfo("Training complete",
                f"Model trained successfully.\\nSamples: {len(y)}\\nClasses: {len(self.class_names)}\\n"
                "Test predictions with different hand positions and lighting.")
        except Exception as e:
            messagebox.showerror("Training error", str(e))

    def load_model(self):
        if os.path.exists(MODEL_FILE):
            try:
                self.model = joblib.load(MODEL_FILE)
                self.class_names = list(self.model.classes_)
                self.model_status.config(text=f"Model loaded • {len(self.class_names)} classes",
                                          fg="#86efac")
            except Exception as e:
                self.model_status.config(text=f"Could not load model: {e}", fg="#fca5a5")
        else:
            self.model_status.config(text="Model: not trained yet. Capture samples and train.",
                                      fg="#fbbf24")

    def predict_current(self):
        if self.model is None or self.current_features is None:
            return
        try:
            probabilities = self.model.predict_proba([self.current_features])[0]
            idx = int(np.argmax(probabilities))
            label = str(self.model.classes_[idx])
            confidence = float(probabilities[idx])
            self.prediction_history.append(label if confidence >= 0.60 else "?")
            stable = Counter(self.prediction_history).most_common(1)[0][0]
            if stable != "?":
                self.current_prediction = stable
                self.current_confidence = confidence
                self.pred_label.config(text=stable)
                self.conf_label.config(text=f"Confidence score: {confidence*100:.1f}%")
                self.meaning_label.config(text=self.get_meaning(stable))
            else:
                self.pred_label.config(text="Uncertain")
                self.conf_label.config(text=f"Confidence score: {confidence*100:.1f}%")
                self.meaning_label.config(text="Try a clearer hand pose or collect more samples.")
        except Exception:
            pass

    def get_meaning(self, label):
        if label in WORD_MEANINGS:
            return WORD_MEANINGS[label]
        if len(label) == 1 and label.isalpha():
            return f"Alphabet {label} — use this letter for fingerspelling."
        return "Recognized sign label."

    def add_prediction(self):
        if not self.current_prediction:
            messagebox.showinfo("No prediction", "A trained model prediction is not available yet.")
            return
        self.sentence += (" " if self.sentence else "") + self.current_prediction
        self.sentence_box.delete("1.0", "end")
        self.sentence_box.insert("1.0", self.sentence)

    def clear_sentence(self):
        self.sentence = ""
        self.sentence_box.delete("1.0", "end")

    def show_dictionary(self):
        win = tk.Toplevel(self.root)
        win.title("SignVision AI - Sign Dictionary")
        win.geometry("520x620")
        win.configure(bg="#111c2e")
        tk.Label(win, text="SIGN DICTIONARY", bg="#111c2e", fg="#f8fafc",
                 font=("Segoe UI", 17, "bold")).pack(anchor="w", padx=18, pady=16)
        text = tk.Text(win, bg="#172235", fg="#e2e8f0", relief="flat",
                       font=("Segoe UI", 10), wrap="word", padx=12, pady=12)
        text.pack(fill="both", expand=True, padx=18, pady=(0, 18))
        text.insert("end", "ALPHABETS A–Z\\n")
        text.insert("end", "The application can be trained to classify each alphabet hand pose.\\n\\n")
        for letter in ALPHABETS:
            text.insert("end", f"{letter}  —  Alphabet {letter} (fingerspelling label)\\n")
        text.insert("end", "\\nCOMMON WORD LABELS\\n")
        for word, meaning in WORD_MEANINGS.items():
            text.insert("end", f"{word}  —  {meaning}\\n")
        text.insert("end", "\\nImportant: This dictionary explains the labels. It does not teach the exact ISL handshape. "
                           "Use an authoritative ISL learning resource to verify each sign before collecting samples.\\n")
        text.config(state="disabled")

    def close(self):
        try:
            if self.cap is not None:
                self.cap.release()
            self.hands.close()
        except Exception:
            pass
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = SignVisionApp(root)
    root.mainloop()
