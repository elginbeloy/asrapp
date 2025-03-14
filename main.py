import os
import warnings
from queue import Queue
import numpy as np
import sounddevice as sd
import torch
from termcolor import colored
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import pipeline, logging


# Hide warnings
warnings.filterwarnings("ignore")
logging.set_verbosity_error()


# Parameters
CHUNK_DURATION_SECONDS = 2
SAMPLE_RATE = 16000
CHUNK_FRAMES = CHUNK_DURATION_SECONDS * SAMPLE_RATE
CHUNKS_PER_REFINEMENT = 8
SILENT_CHUNKS_TO_FORGET = 3
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


# Silero VAD: Detect if a chunk of audio contains speech
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


# Queue to store audio data frames
audio_queue = Queue()
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())


def print_transcripts(full_transcript, segment_transcript, partial_transcript):
    os.system("clear")
    print(full_transcript, end=" ")
    print(colored(segment_transcript, "blue"))
    print(colored(partial_transcript, "yellow"))


def main():
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
    partial_buffer = [] # for a single chunk
    full_transcript = ""
    segment_transcript = ""
    partial_transcript = ""
    frames_collected = 0
    chunk_count = 0
    consecutive_silent_chunks = 0

    # Start recording
    with sd.InputStream(
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32",
        callback=audio_callback
    ):
        print(colored("Recording!", "green", attrs=["bold"]))
        print("Press Ctrl+C to stop.")
        try:
            while True:
                block = audio_queue.get()
                partial_buffer.append(block)
                segment_buffer.append(block)
                master_buffer.append(block)
                frames_collected += block.shape[0]

                # Process chunk once we get enough frames
                if frames_collected >= CHUNK_FRAMES:
                    frames_collected = 0
                    chunk_data = np.concatenate(
                      partial_buffer, axis=0).squeeze()
                    partial_buffer = []

                    if is_speech_chunk(chunk_data):
                        chunk_count += 1
                        consecutive_silent_chunks = 0
                        if chunk_count > CHUNKS_PER_REFINEMENT:
                            chunk_count = 0
                            all_audio_data = np.concatenate(
                              segment_buffer, axis=0).squeeze()
                            partial_transcript = ""
                            segment_transcript = asr_pipeline({
                              "array": all_audio_data,
                              "sampling_rate": SAMPLE_RATE,
                            })["text"]
                        else:
                            partial_transcript += asr_pipeline({
                              "array": chunk_data,
                              "sampling_rate": SAMPLE_RATE
                            })["text"]
                        print_transcripts(full_transcript, segment_transcript, partial_transcript)
                    else:
                        consecutive_silent_chunks += 1
                        if consecutive_silent_chunks >= SILENT_CHUNKS_TO_FORGET:
                            # After long silence break out segment so we dont re-refine
                            consecutive_silent_chunks = 0
                            if len(segment_transcript) > 1:
                              # partial transcript may still have some not put into segment 
                              # since we only refine every N chunks
                              full_transcript += " " + segment_transcript + partial_transcript
                            segment_buffer = []
                            segment_transcript  = ""
                            partial_transcript = ""
                            print_transcripts(full_transcript, segment_transcript, partial_transcript)
                            print("(new segment cuz of long silence)")

        except KeyboardInterrupt:
            print("\nStopped by user.")
            # Final transcript from the entire audio
            # TODO: Does this really help compared to just last active chunks?
            if len(master_buffer) > 0:
                all_audio_data = np.concatenate(master_buffer, axis=0).squeeze()
                refined_result = asr_pipeline({"array": all_audio_data, "sampling_rate": SAMPLE_RATE})
                full_transcript = refined_result["text"]
            pass

    # Write final transcript to file
    with open("memory.txt", "w", encoding="utf-8") as file:
        file.write(full_transcript)
    print("Final transcription saved to memory.txt\n")


if __name__ == "__main__":
  main()
