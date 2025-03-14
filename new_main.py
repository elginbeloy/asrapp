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


def print_transcripts(full_transcript, partial_transcript):
    os.system("clear")
    print(full_transcript, end=" ")
    print(colored(partial_transcript, "blue"))


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
    full_transcript = ""
    partial_buffer = []
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
                              master_buffer, axis=0).squeeze()
                            partial_transcript = ""
                            full_transcript = asr_pipeline({
                              "array": all_audio_data,
                              "sampling_rate": SAMPLE_RATE,
                            })["text"]
                        else:
                            partial_transcript += asr_pipeline({
                              "array": chunk_data,
                              "sampling_rate": SAMPLE_RATE
                            })["text"]
                        print_transcripts(full_transcript, partial_transcript)
                    else:
                        consecutive_silent_chunks += 1
                        if consecutive_silent_chunks >= SILENT_CHUNKS_TO_FORGET:
                            # TODO:
                            consecutive_silent_chunks = 0
                            print("could refine and forget")
                            # REFINE AND FORGET
                            # After enough silence we reset the master buffer.
                            # This is because theres no risk of cutting off
                            # in between words so these set of chunks can be
                            # processed seperately and concatenated.
                            # TODO: should not be 1 but CHUNK_FRAMES is too big?
                            # print(len(master_buffer))
                            #  if len(master_buffer) >= 1:
                            #    master_buffer = []
                            #    chunk_count = 0
                            #    all_audio_data = np.concatenate(
                            #      master_buffer, axis=0).squeeze()
                            #    partial_transcript = ""
                            #    full_transcript = asr_pipeline({
                            #      "array": all_audio_data,
                            #      "sampling_rate": SAMPLE_RATE,
                            #    })["text"]
                            #    print_transcripts(full_transcript, partial_transcript)

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
