import cv2
import mediapipe as mp
import numpy as np
import time
import wmi
import pyautogui
import win32gui
import pygame
import speech_recognition as sr
import threading

# Initialize pygame mixer for audio playback
pygame.mixer.init()

# Load the Rickroll chorus audio file (replace with your actual file path if needed)
AUDIO_FILE = 'rickroll_chorus.mp3'
try:
    pygame.mixer.music.load(AUDIO_FILE)
except:
    print("Warning: Could not load rickroll_chorus.mp3")

# Mediapipe init
mp_face_mesh = mp.solutions.face_mesh
mp_hands = mp.solutions.hands

# Landmarks for eyes and mouth
LEFT_EYE_IDX = [33, 160, 158, 133, 153, 144]
RIGHT_EYE_IDX = [362, 385, 387, 263, 373, 380]
UPPER_LIP = 13
LOWER_LIP = 14
MOUTH_LEFT = 61
MOUTH_RIGHT = 291

# Hand landmarks for fingers (tips and pip joints)
THUMB_TIP = 4
THUMB_IP = 3
INDEX_TIP = 8
INDEX_PIP = 6
MIDDLE_TIP = 12
MIDDLE_PIP = 10
RING_TIP = 16
RING_PIP = 14
PINKY_TIP = 20
PINKY_PIP = 18

class BananaRainEffect:
    def __init__(self):
        self.bananas = []
        self.is_active = False
        self.start_time = None
        self.duration = 2.0  # 2 seconds
        
    def start_banana_rain(self, screen_width, screen_height):
        """Start the banana rain effect"""
        self.bananas = []
        self.is_active = True
        self.start_time = time.time()
        
        # Create initial bananas across the top of the screen
        num_bananas = 15  # Number of banana streams
        for i in range(num_bananas):
            x = (i * screen_width // num_bananas) + np.random.randint(-50, 50)
            self.bananas.append({
                'x': x,
                'y': -30,
                'speed': np.random.uniform(150, 300),  # pixels per second
                'size': np.random.uniform(20, 40)
            })
        
        print("🍌 BANANA RAIN ACTIVATED! 🍌")
    
    def update_and_draw(self, frame, dt):
        """Update banana positions and draw them on the frame"""
        if not self.is_active:
            return frame
        
        h, w = frame.shape[:2]
        
        # Check if effect should end
        if time.time() - self.start_time > self.duration:
            self.is_active = False
            self.bananas = []
            return frame
        
        # Add new bananas randomly during the effect
        if np.random.random() < 0.3:  # 30% chance each frame
            x = np.random.randint(0, w)
            self.bananas.append({
                'x': x,
                'y': -30,
                'speed': np.random.uniform(150, 300),
                'size': np.random.uniform(20, 40)
            })
        
        # Update and draw each banana
        bananas_to_remove = []
        for i, banana in enumerate(self.bananas):
            # Update position
            banana['y'] += banana['speed'] * dt
            
            # Remove bananas that have fallen off screen
            if banana['y'] > h + 50:
                bananas_to_remove.append(i)
                continue
            
            # Draw banana emoji as text (since OpenCV doesn't support emoji rendering well, we'll use 'B')
            font_scale = banana['size'] / 30.0
            
            # Draw banana with slight shadow effect
            shadow_x = int(banana['x'] + 2)
            shadow_y = int(banana['y'] + 2)
            cv2.putText(frame, 'B', (shadow_x, shadow_y), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 0, 0), 3)  # Black shadow
            
            cv2.putText(frame, 'B', (int(banana['x']), int(banana['y'])), cv2.FONT_HERSHEY_SIMPLEX,
                       font_scale, (0, 255, 255), 2)  # Yellow 'B' for banana
        
        # Remove bananas that have fallen off screen
        for i in reversed(bananas_to_remove):
            self.bananas.pop(i)
        
        return frame

class VoiceRecognizer:
    def __init__(self, callback):
        self.recognizer = sr.Recognizer()
        self.microphone = sr.Microphone()
        self.callback = callback
        self.listening = True
        
        # Adjust for ambient noise
        print("Calibrating microphone for ambient noise...")
        with self.microphone as source:
            self.recognizer.adjust_for_ambient_noise(source, duration=2)
        print("Microphone calibrated!")
    
    def listen_continuously(self):
        """Continuously listen for voice commands in a separate thread"""
        def listen_worker():
            while self.listening:
                try:
                    with self.microphone as source:
                        # Listen for audio with timeout
                        audio = self.recognizer.listen(source, timeout=1, phrase_time_limit=3)
                    
                    try:
                        # Recognize speech using Google
                        text = self.recognizer.recognize_google(audio).lower()
                        print(f"Heard: {text}")
                        
                        # Check for trigger word
                        if "banana" in text:
                            self.callback("banana")
                            
                    except sr.UnknownValueError:
                        pass  # Could not understand audio
                    except sr.RequestError as e:
                        print(f"Speech recognition error: {e}")
                        time.sleep(1)
                        
                except sr.WaitTimeoutError:
                    pass  # No audio detected, continue listening
                except Exception as e:
                    print(f"Microphone error: {e}")
                    time.sleep(1)
        
        # Start listening thread
        listen_thread = threading.Thread(target=listen_worker, daemon=True)
        listen_thread.start()
        return listen_thread
    
    def stop_listening(self):
        """Stop the voice recognition"""
        self.listening = False

# WMI brightness controller
def set_brightness(level):
    level = max(0, min(100, level))
    try:
        c = wmi.WMI(namespace='wmi')
        methods = c.WmiMonitorBrightnessMethods()
        for method in methods:
            method.WmiSetBrightness(level, 0)
    except Exception as e:
        print("Brightness set error:", e)

# Calculate EAR for one eye
def eye_aspect_ratio(landmarks, indices, w, h):
    p = landmarks
    v1 = np.linalg.norm(np.array([p[indices[1]].x * w, p[indices[1]].y * h]) - np.array([p[indices[5]].x * w, p[indices[5]].y * h]))
    v2 = np.linalg.norm(np.array([p[indices[2]].x * w, p[indices[2]].y * h]) - np.array([p[indices[4]].x * w, p[indices[4]].y * h]))
    h_dist = np.linalg.norm(np.array([p[indices[0]].x * w, p[indices[0]].y * h]) - np.array([p[indices[3]].x * w, p[indices[3]].y * h]))
    if h_dist == 0:
        return 0
    ear = (v1 + v2) / (2.0 * h_dist)
    return ear

# Mouth open ratio
def mouth_open_ratio(landmarks, w, h):
    top = np.array([landmarks[UPPER_LIP].x * w, landmarks[UPPER_LIP].y * h])
    bottom = np.array([landmarks[LOWER_LIP].x * w, landmarks[LOWER_LIP].y * h])
    left = np.array([landmarks[MOUTH_LEFT].x * w, landmarks[MOUTH_LEFT].y * h])
    right = np.array([landmarks[MOUTH_RIGHT].x * w, landmarks[MOUTH_RIGHT].y * h])

    vertical = np.linalg.norm(bottom - top)
    horizontal = np.linalg.norm(right - left)
    if horizontal == 0:
        return 0
    return vertical / horizontal

# Check if active window title contains "YouTube"
def is_youtube_active():
    try:
        hwnd = win32gui.GetForegroundWindow()
        title = win32gui.GetWindowText(hwnd).lower()
        return "youtube" in title
    except Exception:
        return False

# Detect thumbs up gesture from hand landmarks
def is_thumbs_up(hand_landmarks, w, h):
    lm = hand_landmarks.landmark

    # Thumb: tip above IP joint in y-axis (thumb extended)
    thumb_up = lm[THUMB_TIP].y < lm[THUMB_IP].y

    # Other fingers folded: tip below PIP joints (in y-axis, fingers bent down)
    index_folded = lm[INDEX_TIP].y > lm[INDEX_PIP].y
    middle_folded = lm[MIDDLE_TIP].y > lm[MIDDLE_PIP].y
    ring_folded = lm[RING_TIP].y > lm[RING_PIP].y
    pinky_folded = lm[PINKY_TIP].y > lm[PINKY_PIP].y

    return thumb_up and index_folded and middle_folded and ring_folded and pinky_folded

def main():
    # Initialize banana rain effect
    banana_rain = BananaRainEffect()
    
    # Voice command callback
    def handle_voice_command(command):
        if command == "banana":
            print("🍌 BANANA detected! Starting banana rain...")
            # Get screen dimensions from the current camera frame
            banana_rain.start_banana_rain(640, 480)  # Default camera resolution
    
    # Initialize voice recognition
    voice_recognizer = VoiceRecognizer(handle_voice_command)
    voice_thread = voice_recognizer.listen_continuously()
    
    cap = cv2.VideoCapture(0)
    face_mesh = mp_face_mesh.FaceMesh(max_num_faces=1,
                                      refine_landmarks=True,
                                      min_detection_confidence=0.5,
                                      min_tracking_confidence=0.5)
    hands_detector = mp_hands.Hands(max_num_hands=1,
                                    min_detection_confidence=0.5,
                                    min_tracking_confidence=0.5)

    calibration_frames = 50
    ear_samples = []
    ear_baseline = None
    EAR_CLOSED = 0.15
    mouth_open_threshold = 0.18
    mouth_cooldown = 0

    flicker_active = False
    flicker_start_time = 0

    # Flicker pattern synced to the Rickroll chorus rhythm (seconds per toggle)
    flicker_pattern = [
        0.25, 0.25, 0.2, 0.2, 0.3, 0.3, 0.4,
        0.25, 0.25, 0.25, 0.25, 0.4, 0.5,
        0.2, 0.2, 0.15, 0.15, 0.15, 0.3, 0.3, 0.4, 0.5,
        0.3, 0.3, 0.3, 0.4, 0.4, 0.4,
        0.4, 0.4, 0.3, 0.3, 0.5, 0.6
    ]
    flicker_index = 0
    flicker_timer = 0
    flicker_on = False

    fade_start_time = 0
    fade_duration = 3.0  # seconds to fade to 60%
    fading = False
    current_brightness = 100
    
    # Frame timing for banana rain animation
    prev_time = time.time()

    print("System ready! Voice commands active.")
    print("Say 'banana' to trigger banana rain effect")

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            h, w, _ = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            face_results = face_mesh.process(rgb)
            hands_results = hands_detector.process(rgb)

            now = time.time()
            dt = now - prev_time  # Delta time for smooth animation
            prev_time = now
            
            # Update banana rain dimensions based on actual frame size
            if banana_rain.is_active and len(banana_rain.bananas) > 0:
                # Update any new bananas to use correct screen width
                for banana in banana_rain.bananas:
                    if banana['x'] < 0 or banana['x'] > w:
                        banana['x'] = np.random.randint(0, w)
            
            # Apply banana rain effect if active
            if banana_rain.is_active:
                frame = banana_rain.update_and_draw(frame, dt)

            # Flicker logic synced to rhythm pattern
            if flicker_active:
                if flicker_index < len(flicker_pattern):
                    if now - flicker_timer >= flicker_pattern[flicker_index]:
                        flicker_on = not flicker_on
                        flicker_timer = now
                        if flicker_on:
                            set_brightness(100)
                        else:
                            set_brightness(0)
                        flicker_index += 1
                    # Visual flicker effect overlay (red tint)
                    overlay = frame.copy()
                    alpha = 0.5 if flicker_on else 0
                    cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
                    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)
                else:
                    flicker_active = False
                    fading = True
                    fade_start_time = now
                    flicker_index = 0

            elif fading:
                elapsed = now - fade_start_time
                if elapsed < fade_duration:
                    new_brightness = int(100 - (elapsed / fade_duration) * 40)  # fade to 60%
                    set_brightness(new_brightness)
                    current_brightness = new_brightness
                else:
                    fading = False
                    current_brightness = 60
                    set_brightness(60)

            # Control brightness from eyes when not flickering/fading
            if not flicker_active and not fading:
                if face_results.multi_face_landmarks:
                    landmarks = face_results.multi_face_landmarks[0].landmark

                    left_ear = eye_aspect_ratio(landmarks, LEFT_EYE_IDX, w, h)
                    right_ear = eye_aspect_ratio(landmarks, RIGHT_EYE_IDX, w, h)
                    ear = (left_ear + right_ear) / 2.0

                    if ear_baseline is None:
                        ear_samples.append(ear)
                        if len(ear_samples) >= calibration_frames:
                            ear_baseline = np.mean(ear_samples)
                            print(f"Calibrated EAR baseline: {ear_baseline:.3f}")

                    if ear_baseline:
                        norm_ear = (ear - EAR_CLOSED) / (ear_baseline - EAR_CLOSED + 1e-6)
                        norm_ear = np.clip(norm_ear, 0, 1)
                        brightness = int(norm_ear * 100)
                        set_brightness(brightness)
                        current_brightness = brightness
                    else:
                        brightness = 50
                else:
                    brightness = current_brightness
            else:
                brightness = current_brightness

            # Mouth open detection and tab close (only if not flickering/fading)
            if not flicker_active and not fading:
                if face_results.multi_face_landmarks:
                    landmarks = face_results.multi_face_landmarks[0].landmark
                    mouth_ratio = mouth_open_ratio(landmarks, w, h)

                    if mouth_ratio > mouth_open_threshold and mouth_cooldown == 0:
                        if is_youtube_active():
                            pyautogui.hotkey('ctrl', 'w')
                            mouth_cooldown = 30  # cooldown ~1 sec

                    if mouth_cooldown > 0:
                        mouth_cooldown -= 1
                else:
                    mouth_ratio = 0
            else:
                mouth_ratio = 0

            # Thumbs-up detection triggers rhythmic flicker+fade only if not flickering/fading already
            if hands_results.multi_hand_landmarks and not flicker_active and not fading:
                hand_landmarks = hands_results.multi_hand_landmarks[0]
                if is_thumbs_up(hand_landmarks, w, h):
                    flicker_active = True
                    flicker_start_time = time.time()
                    flicker_timer = flicker_start_time
                    flicker_index = 0
                    flicker_on = False
                    try:
                        pygame.mixer.music.play()
                    except:
                        pass
                    print("Thumbs up detected: starting rhythmic flicker (Rickroll beats).")

            # Voice command to start banana rain with correct dimensions
            if not banana_rain.is_active:
                # Update callback to use actual frame dimensions
                def updated_voice_callback(command):
                    if command == "banana":
                        print("🍌 BANANA detected! Starting banana rain...")
                        banana_rain.start_banana_rain(w, h)
                
                voice_recognizer.callback = updated_voice_callback

            # Display info
            cv2.putText(frame, f"Brightness: {brightness}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (255, 255, 0), 2)
            cv2.putText(frame, f"Mouth Ratio: {mouth_ratio:.2f}", (10, 60), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 0), 2)
            cv2.putText(frame, f"Flicker: {'On' if flicker_active else 'Off'}", (10, 90), cv2.FONT_HERSHEY_SIMPLEX,
                        0.8, (0, 255, 255), 2)
            cv2.putText(frame, f"Voice: Listening for 'banana'", (10, 120), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 0, 255), 2)
            cv2.putText(frame, f"Banana Rain: {'Active' if banana_rain.is_active else 'Ready'}", (10, 150), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (0, 255, 255), 2)

            cv2.imshow("Enhanced Vision Control with Voice", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    except KeyboardInterrupt:
        print("\nShutting down...")
    finally:
        # Cleanup
        voice_recognizer.stop_listening()
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()