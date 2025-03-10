import pyaudio
import numpy as np
import librosa
import openl3
import time
import sys

# Bonsai growth stages
BONSAI_STAGES = ["🌱", "🌿", "🌳", "🌲", "🌲🌲", "🌲🌲🌲"]

# Audio settings
FORMAT = pyaudio.paInt16
CHANNELS = 1
RATE = 16000  # 16kHz sample rate
CHUNK = RATE  # 1-second chunks

# Initialize PyAudio
audio = pyaudio.PyAudio()
stream = audio.open(format=FORMAT, channels=CHANNELS, rate=RATE,
                    input=True, frames_per_buffer=CHUNK)

# Function to process audio and get pleasiness score
def get_pleasiness(audio_data):
    # Convert byte data to NumPy array
    audio_np = np.frombuffer(audio_data, dtype=np.int16).astype(np.float32) / 32768.0

    # Extract OpenL3 embeddings
    emb, _ = openl3.get_audio_embedding(audio_np, sr=RATE, content_type="music")

    # Simple score: Use mean of embeddings (higher might indicate richer sounds)
    pleasiness_score = np.mean(emb)

    return pleasiness_score

# Bonsai growth logic
def update_bonsai(score, stage):
    if score > 0.1:  # Threshold (tweak as needed)
        stage = min(stage + 1, len(BONSAI_STAGES) - 1)
    return stage

# CLI Real-time Loop
bonsai_stage = 0
print("\n🎤 Listening... Speak or play sound 🎶")

try:
    while True:
        audio_data = stream.read(CHUNK, exception_on_overflow=False)
        score = get_pleasiness(audio_data)
        bonsai_stage = update_bonsai(score, bonsai_stage)

        # Clear previous line and print bonsai
        sys.stdout.write(f"\r{BONSAI_STAGES[bonsai_stage]} (Pleasiness: {score:.3f}) ")
        sys.stdout.flush()

        time.sleep(1)  # Process every second

except KeyboardInterrupt:
    print("\n🌿 Bonsai growth complete! Exiting...")
    stream.stop_stream()
    stream.close()
    audio.terminate()
