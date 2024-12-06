import tkinter as tk
from tkinter import Label, ttk, messagebox, Canvas
from PIL import Image, ImageTk
import cv2
import threading
import time
import random
import mediapipe as mp
from transformers import GPT2LMHeadModel, GPT2Tokenizer
from TTS.api import TTS
from pydub import AudioSegment
import simpleaudio as sa
from queue import Queue
import sounddevice as sd
import numpy as np
from scipy.signal import spectrogram
import torch
import os

# Global variables for better state management
running = True
ghost_activity_level = 0
activity_decay_rate = 0.98
last_phrase_time = time.time()
phrase_cooldown = 15.0
detection_threshold = 0.85
selected_webcam = 0
selected_microphone = 0
voice_queue = Queue()
current_phrase = None 

# Initialize models
print("Loading GPT-2 model...")
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
model = GPT2LMHeadModel.from_pretrained("gpt2").to("cuda" if torch.cuda.is_available() else "cpu")
print("GPT-2 model loaded.")

print("Loading TTS model...")
tts = TTS(model_name="tts_models/en/ljspeech/tacotron2-DCA", progress_bar=False, gpu=torch.cuda.is_available())
print("TTS model loaded.")

# Initialize Mediapipe
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils

def add_ghost_effect(frame, x, y, w, h):
    """Enhanced ghost effects with better noise handling"""
    overlay = frame.copy()
    effect_type = random.choice(['mist', 'orb', 'shadow'])
    
    if effect_type == 'mist':
        # Create misty effect
        alpha = random.uniform(0.3, 0.5)
        cv2.circle(overlay, (x + w//2, y + h//2), 
                  max(w, h)//2, (200, 200, 255), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
        
    elif effect_type == 'orb':
        # Create orb effect
        for r in range(min(w, h)//2, 0, -5):
            alpha = random.uniform(0.1, 0.3)
            color = (200 + random.randint(-20, 20),
                    200 + random.randint(-20, 20),
                    200 + random.randint(-20, 20))
            cv2.circle(overlay, (x + w//2, y + h//2), r, color, -1)
            frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
            
    else:  # shadow
        # Create shadow effect
        alpha = random.uniform(0.2, 0.4)
        cv2.rectangle(overlay, (x, y), (x + w, y + h), 
                     (30, 30, 30), -1)
        frame = cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0)
    
    return frame

def process_webcam_feed(webcam_label, status_var):
    global running, ghost_activity_level
    
    cap = cv2.VideoCapture(selected_webcam, cv2.CAP_DSHOW)
    if not cap.isOpened():
        messagebox.showerror("Error", f"Cannot open camera {selected_webcam}")
        return

    prev_frame = None
    noise_threshold = 15
    
    with mp_face_detection.FaceDetection(min_detection_confidence=0.5) as face_detection:
        while running:
            ret, frame = cap.read()
            if not ret:
                continue

            # Noise analysis
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            if prev_frame is not None:
                frame_diff = cv2.absdiff(gray, prev_frame)
                noise_level = np.mean(frame_diff)
                
                if noise_level > noise_threshold:
                    ghost_activity_level = min(1.0, ghost_activity_level + noise_level/1000)
                    status_var.set("👻 Paranormal Activity Detected!")
                    
                    # Add ambient noise effect
                    noise = np.random.normal(0, 25, frame.shape).astype(np.uint8)
                    frame = cv2.add(frame, noise)

            prev_frame = gray

            # Face detection with ghost effects
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(rgb_frame)

            if results.detections:
                for detection in results.detections:
                    if random.random() > detection_threshold:
                        bbox = detection.location_data.relative_bounding_box
                        x = int(bbox.xmin * frame.shape[1])
                        y = int(bbox.ymin * frame.shape[0])
                        w = int(bbox.width * frame.shape[1])
                        h = int(bbox.height * frame.shape[0])
                        
                        frame = add_ghost_effect(frame, x, y, w, h)
                        ghost_activity_level = min(1.0, ghost_activity_level + 0.1)

            # Update the display
            img = Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
            img_tk = ImageTk.PhotoImage(image=img)
            webcam_label.imgtk = img_tk
            webcam_label.configure(image=img_tk)

            # Decay ghost activity
            ghost_activity_level *= activity_decay_rate
            time.sleep(0.03)

    cap.release()

def synthesize_ghost_voice(phrase):
    try:
        # Generate audio file using TTS
        temp_file = f"ghost_voice_{int(time.time())}.wav"
        tts.tts_to_file(text=phrase, file_path=temp_file)
        
        # Add the file to the playback queue
        voice_queue.put(temp_file)
    except Exception as e:
        print(f"Error in synthesize_ghost_voice: {e}")


def extract_audio_features(indata, frames, time_info, status):
    """Simplified but effective audio processing"""
    global last_phrase_time, ghost_activity_level, current_phrase  # Add global references
    
    if status:
        return
        
    try:
        audio = np.mean(indata, axis=1)
        if np.max(np.abs(audio)) > 0.01:  # Only process if there's significant audio
            f, t, Sxx = spectrogram(audio, 16000)
            if np.any(Sxx):
                # Generate phrase based on audio characteristics
                random_value = np.mean(Sxx)
                current_time = time.time()
                
                if current_time - last_phrase_time >= phrase_cooldown:
                    seed = int(random_value * 10000)
                    phrase = generate_ghostly_phrase(seed)
                    current_phrase.set(phrase)
                    synthesize_ghost_voice(phrase)
                    last_phrase_time = current_time
                    
                # Update activity based on audio
                ghost_activity_level = min(1.0, ghost_activity_level + random_value * 0.1)
                
    except Exception as e:
        print(f"Audio processing error: {e}")

def start_gui():
    global current_phrase
    root = tk.Tk()
    root.title("👻 Ghost Detector 👻")
    root.configure(bg='black')
    
    # Set window size
    root.geometry("1024x768")
    
    # Create main container
    main_container = tk.Frame(root, bg='black')
    main_container.pack(expand=True, fill='both', padx=20, pady=20)
    
    # Status panel
    status_panel = tk.Frame(main_container, bg='#1a1a1a', relief='raised', bd=2)
    status_panel.pack(fill='x', pady=(0, 10))
    
    # Ghost activity meter
    activity_canvas = Canvas(status_panel, height=30, bg='black', highlightthickness=0)
    activity_canvas.pack(fill='x', padx=10, pady=5)
    
    # Current status
    status_var = tk.StringVar(value="📡 Scanning for paranormal activity...")
    status_label = tk.Label(status_panel, textvariable=status_var, 
                          font=("Courier New", 12), bg='#1a1a1a', fg='#00ff00')
    status_label.pack(pady=5)

    # Create webcam feed
    webcam_label = Label(main_container, bg='black')
    webcam_label.pack(expand=True, fill='both')
    
    # Ghost messages panel
    current_phrase = tk.StringVar(value="Awaiting ghostly manifestations...")
    message_label = tk.Label(main_container, textvariable=current_phrase,
                            font=("Courier New", 14), bg='#1a1a1a', fg='#00ff00',
                            wraplength=980)
    message_label.pack(pady=10)
    
    # Controls panel
    controls_frame = tk.Frame(main_container, bg='black')
    controls_frame.pack(fill='x', pady=10)
    
    # Device selection frame
    device_frame = tk.Frame(controls_frame, bg='black')
    device_frame.pack(side='left', padx=10)
    
    # Style for dropdowns
    style = ttk.Style()
    style.theme_use('default')
    style.configure('Ghost.TCombobox',
                    fieldbackground='black',
                    background='#00ff00',
                    foreground='#00ff00')
    
    # Device dropdowns
    webcams = list_webcams()
    tk.Label(device_frame, text="📹 Camera:", font=("Courier New", 10),
            bg='black', fg='#00ff00').pack(side='left', padx=5)
    webcam_dropdown = ttk.Combobox(device_frame, values=webcams,
                                  state="readonly", style='Ghost.TCombobox', width=25)
    webcam_dropdown.current(0)
    webcam_dropdown.pack(side='left', padx=5)
    
    mics = list_microphones()
    tk.Label(device_frame, text="🎤 Audio:", font=("Courier New", 10),
            bg='black', fg='#00ff00').pack(side='left', padx=5)
    mic_dropdown = ttk.Combobox(device_frame, values=mics,
                               state="readonly", style='Ghost.TCombobox', width=30)
    mic_dropdown.current(0)
    mic_dropdown.pack(side='left', padx=5)
    
    # Start button
    start_button = tk.Button(controls_frame, text="👻 Begin Ghost Detection 👻",
                            font=("Courier New", 12, "bold"),
                            command=lambda: start_detector(webcam_label, status_var),
                            bg='#1a1a1a', fg='#00ff00')
    start_button.pack(side='right', padx=10)
    
    # Bind device selection events
    webcam_dropdown.bind("<<ComboboxSelected>>", 
                        lambda e: select_webcam(webcam_dropdown.get()))
    mic_dropdown.bind("<<ComboboxSelected>>",
                     lambda e: select_microphone(mic_dropdown.get()))
    
    def update_activity_meter():
        """Update the ghost activity meter"""
        if running:
            # Get the current width and update meter
            width = activity_canvas.winfo_width()
            meter_width = int(width * ghost_activity_level)
            
            # Choose color based on activity level
            if ghost_activity_level < 0.3:
                color = '#00ff00'
            elif ghost_activity_level < 0.7:
                color = '#ffff00'
            else:
                color = '#ff0000'
            
            # Update meter
            activity_canvas.delete('all')
            activity_canvas.create_rectangle(0, 0, meter_width, 30,
                                          fill=color, width=0)
            
            # Schedule next update
            root.after(50, update_activity_meter)
    
    # Start the activity meter
    update_activity_meter()
    
    def on_closing():
        """Handle window closing"""
        global running
        if messagebox.askokcancel("Quit", "🚪 Close the portal to the spirit realm?"):
            running = False
            root.destroy()
    
    root.protocol("WM_DELETE_WINDOW", on_closing)
    return root

def start_detector(webcam_label, status_var):
    """Start all ghost detection systems"""
    global running
    running = True
    
    # Start webcam thread
    webcam_thread = threading.Thread(target=process_webcam_feed,
                                   args=(webcam_label, status_var),
                                   daemon=True)
    webcam_thread.start()
    
    # Start audio capture
    audio_thread = threading.Thread(target=lambda: start_audio_capture(status_var),
                                  daemon=True)
    audio_thread.start()
    
    # Start voice playback
    voice_thread = threading.Thread(target=voice_playback_thread,
                                  daemon=True)
    voice_thread.start()

def start_audio_capture(status_var):
    """Start audio capture with error handling"""
    try:
        with sd.InputStream(
            samplerate=16000,
            device=selected_microphone,
            channels=1,
            callback=extract_audio_features,
            blocksize=1024
        ) as stream:
            while running:
                time.sleep(0.1)
    except Exception as e:
        print(f"Audio capture error: {e}")
        status_var.set("⚠️ Audio device error - check settings")

def voice_playback_thread():
    """Handle ghost voice playback"""
    while running:
        if not voice_queue.empty():
            voice_file = voice_queue.get()
            try:
                audio = AudioSegment.from_wav(voice_file)
                
                # Add random effects
                if random.random() < 0.5:
                    audio = audio.reverse()
                
                audio = audio.low_pass_filter(1000)
                audio = audio.high_pass_filter(300)
                
                # Play the processed audio
                play_obj = sa.play_buffer(
                    audio.raw_data,
                    num_channels=audio.channels,
                    bytes_per_sample=audio.sample_width,
                    sample_rate=audio.frame_rate
                )
                play_obj.wait_done()
                
                # Clean up
                os.remove(voice_file)
            except Exception as e:
                print(f"Voice playback error: {e}")
        else:
            time.sleep(0.1)

def generate_ghostly_phrase(random_seed):
    """Generate spooky phrases"""
    prompts = [
        "The spirit whispers: ",
        "Through ancient walls: ",
        "From the shadows: ",
        "A ghostly voice: ",
        "The veil parts: "
    ]
    
    torch.manual_seed(random_seed)
    prompt = random.choice(prompts)
    
    # Generate with attention mask
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(model.device)
    attention_mask = torch.ones_like(input_ids)
    
    outputs = model.generate(
        input_ids,
        attention_mask=attention_mask,
        pad_token_id=tokenizer.eos_token_id,
        max_length=20,
        num_return_sequences=1,
        do_sample=True,
        temperature=0.9,
        top_k=40,
        top_p=0.85
    )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

def list_webcams():
    """List available webcams"""
    webcams = []
    try:
        for i in range(10):
            cap = cv2.VideoCapture(i, cv2.CAP_DSHOW)
            if cap.read()[0]:
                webcams.append(f"Camera {i}")
                cap.release()
        return webcams if webcams else ["Camera 0"]
    except Exception as e:
        print(f"Error listing webcams: {e}")
        return ["Camera 0"]

def list_microphones():
    """List available microphones"""
    try:
        devices = sd.query_devices()
        return [f"Microphone {i}" for i, device in enumerate(devices)
                if device['max_input_channels'] > 0]
    except Exception as e:
        print(f"Error listing microphones: {e}")
        return ["Microphone 0"]

def select_webcam(selection):
    """Handle webcam selection"""
    global selected_webcam
    try:
        if "Camera" in selection:
            selected_webcam = int(selection.split(" ")[1])
    except Exception as e:
        print(f"Error selecting webcam: {e}")
        selected_webcam = 0

def select_microphone(selection):
    """Handle microphone selection"""
    global selected_microphone
    try:
        if "Microphone" in selection:
            selected_microphone = int(selection.split(" ")[1])
    except Exception as e:
        print(f"Error selecting microphone: {e}")
        selected_microphone = 0

if __name__ == "__main__":
    root = start_gui()
    root.mainloop()