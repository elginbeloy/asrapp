import os
import time
import warnings
from queue import Queue
import numpy as np
import sounddevice as sd
import torch
from termcolor import colored
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import pipeline, logging
from argparse import ArgumentParser
from sys import getsizeof

# pynput for hotkey detection
from pynput.keyboard import Listener, KeyCode

# Hide warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()

# Parameters
CHUNK_DURATION_SECONDS = 2
SAMPLE_RATE = 16000
CHUNK_FRAMES = CHUNK_DURATION_SECONDS * SAMPLE_RATE
CHUNKS_PER_REFINEMENT = 8
SEGMENT_SILENT_CHUNKS = 2
VAD_THRESHOLD = 0.5
# https://huggingface.co/openai/whisper-base
MODEL_TO_USE = "openai/whisper-small.en"
DEVICE_TO_USE = "cpu"
if torch.cuda.is_available():
    DEVICE_TO_USE = "cuda:0"
# MPS (Apple Silicon) is only available in PyTorch 1.12+
if getattr(torch.backends, "mps", None):
    if torch.backends.mps.is_available():
        DEVICE_TO_USE = "mps"

# Load Silero VAD
vad_model = load_silero_vad()
def is_speech_chunk(chunk_data):
    tensor_data = torch.from_numpy(chunk_data).float()
    if len(tensor_data.shape) == 1:
        tensor_data = tensor_data.unsqueeze(0)
    with torch.no_grad():
        speech_ts = get_speech_timestamps(
            tensor_data,
            vad_model,
            sampling_rate=SAMPLE_RATE,
            threshold=VAD_THRESHOLD
        )
    return len(speech_ts) > 0

# A queue to collect audio from the sounddevice callback
audio_queue = Queue()
def audio_callback(indata, frames, time_info, status):
    audio_queue.put(indata.copy())

def get_audio_text(audio_data, asr_pipeline):
    return asr_pipeline({"array": audio_data, "sampling_rate": SAMPLE_RATE})["text"]

def get_byte_str(byte_count):
    if byte_count > 1_000_000_000:
        val = round(byte_count / 100_000_000) / 10
        return f"{val}GB"
    elif byte_count > 1_000_000:
        val = round(byte_count / 100_000) / 10
        return f"{val}MB"
    elif byte_count > 1000:
        val = round(byte_count / 100) / 10
        return f"{val}KB"
    else:
        return f"{byte_count}B"

def print_data(master_buffer_size, segment_buffer_size):
    master_byte_str = get_byte_str(master_buffer_size)
    segment_byte_str = get_byte_str(segment_buffer_size)
    print("\nBuffer Sizes:", end="  ")
    print(colored(f"{master_byte_str} master", "green"), end=" | ")
    print(colored(f"{segment_byte_str} segment", "blue"))

def print_banner(recording):
    """Print the hotkey banner and current recording status."""
    print(colored("[S] Stop & process the master buffer", "red"), " | ",
          colored("[R] Toggle Recording", "green"))
    status_text = "● Recording" if recording else "○ Paused"
    status_color = "green" if recording else "yellow"
    print(colored(f"Status: {status_text}", status_color, attrs=["bold"]))
    print("-" * 50)

def print_transcripts(full_transcript, segment_transcript, partial_transcript, recording):
    """Clear screen, print banner, then transcripts."""
    os.system("clear")
    print_banner(recording)
    # Show the transcripts in different colors
    print(colored(full_transcript, "white", attrs=["bold"]), end=" ")
    print(colored(segment_transcript, "blue"), end=" ")
    print(colored(partial_transcript, "yellow"))

def record(verbose=False, finish_on_silence=False):
    print("Loading Whisper pipeline...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_TO_USE,
        device=DEVICE_TO_USE
    )
    print("Pipeline loaded.\n")

    # Tracking data and buffers
    master_buffer = []
    segment_buffer = []
    partial_buffer = []  # for a single chunk
    full_transcript = ""
    segment_transcript = ""
    partial_transcript = ""
    frames_collected = 0
    chunk_count = 0
    consecutive_silent_chunks = 0

    # Stop/recording flags
    stop = False
    recording = True

    # Define the listener callback
    def on_press(key):
        nonlocal stop, recording
        if isinstance(key, KeyCode):
            if key.char == 's':
                stop = True
                # Stop listener when S is pressed
                return False
            elif key.char == 'r':
                recording = not recording

    listener = Listener(on_press=on_press)
    listener.start()

    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    ):
        # Main loop to process audio
        while not stop:
            # Process audio only if we're currently recording
            if recording and not audio_queue.empty():
                block = audio_queue.get()
                partial_buffer.append(block)
                segment_buffer.append(block)
                master_buffer.append(block)
                frames_collected += block.shape[0]

                # Once we have enough frames for a chunk
                if frames_collected >= CHUNK_FRAMES:
                    frames_collected = 0
                    chunk_data = np.concatenate(partial_buffer, axis=0).squeeze()
                    partial_buffer = []

                    if is_speech_chunk(chunk_data):
                        chunk_count += 1
                        consecutive_silent_chunks = 0
                        # Refine segment transcript with the entire segment buffer
                        if chunk_count > CHUNKS_PER_REFINEMENT:
                            chunk_count = 0
                            all_audio_data = np.concatenate(segment_buffer, axis=0).squeeze()
                            partial_transcript = ""
                            segment_transcript = get_audio_text(all_audio_data, asr_pipeline)
                        else:
                            # Transcribe just this chunk into partial_transcript
                            partial_transcript += get_audio_text(chunk_data, asr_pipeline)

                    else:
                        consecutive_silent_chunks += 1
                        if consecutive_silent_chunks >= SEGMENT_SILENT_CHUNKS:
                            # After enough silent chunks, close out this segment
                            consecutive_silent_chunks = 0
                            chunk_count = 0
                            if (len(segment_transcript) > 1 or len(partial_transcript) > 1):
                                full_transcript += " " + segment_transcript
                                full_transcript += " " + partial_transcript
                            segment_buffer = []
                            segment_transcript = ""
                            partial_transcript = ""

                    # Clear screen, print banner & transcripts
                    print_transcripts(full_transcript, segment_transcript, partial_transcript, recording)
                    if verbose:
                        print_data(getsizeof(master_buffer), getsizeof(segment_buffer))

            # Small pause so we don't hog the CPU
            time.sleep(0.001)

    # Once 'S' is pressed, we're done
    print("\nStopped by user.")
    # Optionally get a final transcription from the entire audio buffer
    if len(master_buffer) > 0:
        all_audio_data = np.concatenate(master_buffer, axis=0).squeeze()
        full_transcript = get_audio_text(all_audio_data, asr_pipeline)

    with open("memory.txt", "w", encoding="utf-8") as file:
        file.write(full_transcript)

    print("Final transcription saved to memory.txt\n")
    return full_transcript

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("-v", "--verbose", action="store_true")
    args = parser.parse_args()
    record(verbose=args.verbose)
