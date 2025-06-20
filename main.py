import os
import warnings
from queue import Queue
import numpy as np
import sounddevice as sd
import torch
from termcolor import colored
from silero_vad import load_silero_vad, get_speech_timestamps
from transformers import pipeline, logging
from argparse import ArgumentParser
from sys import stdin, getsizeof
import tty
import termios
from select import select

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


def get_audio_text(audio_data, asr_pipeline):
    return asr_pipeline({
        "array": audio_data,
        "sampling_rate": SAMPLE_RATE,
    })["text"]


def get_byte_str(bytes):
  if bytes > 1000000000:
      bytes = round(bytes / 100000000)/10
      return f"{bytes}GB"
  elif bytes > 1000000:
      bytes = round(bytes / 100000)/10
      return f"{bytes}MB"
  elif bytes > 1000:
      bytes = round(bytes / 100)/10
      return f"{bytes}KB"
  else:
      return f"{bytes}B"


def print_data(
    master_buffer_size,
    segment_buffer_size,
):
    master_byte_str = get_byte_str(master_buffer_size)
    segment_byte_str = get_byte_str(segment_buffer_size)
    print("\nBuffer Sizes:", end="  ")
    print(colored(f"{master_byte_str} master", "green"), end=" | ")
    print(colored(f"{segment_byte_str} segment", "blue"))


def print_transcripts(
    full_transcript,
    segment_transcript,
    partial_transcript
):
    os.system("clear")
    print(colored(full_transcript, "white", attrs=["bold"]), end=" ")
    print(colored(segment_transcript, "blue"), end=" ")
    print(colored(partial_transcript, "yellow"))


def record(verbose=False, cutoff_on_silence=False):
    # Load the model
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
        print("Press R to stop.")

        # File descriptor for stdin
        fd = stdin.fileno()
        # Save the original terminal settings
        old_settings = termios.tcgetattr(fd)

        try:
            tty.setcbreak(fd)
            while True:
                if stdin in select([stdin], [], [], 0)[0]:
                    ch = stdin.read(1)
                    if ch == "r":
                        if len(master_buffer) > 0:
                            all_audio_data = np.concatenate(
                                master_buffer,
                                axis=0
                            ).squeeze()
                            full_transcript = get_audio_text(
                                all_audio_data,
                                asr_pipeline
                            )
                        return full_transcript

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
                        # Refine segment transcript using segment audio
                        if chunk_count > CHUNKS_PER_REFINEMENT:
                            chunk_count = 0
                            all_audio_data = np.concatenate(
                              segment_buffer, axis=0).squeeze()
                            partial_transcript = ""
                            segment_transcript = get_audio_text(
                                all_audio_data,
                                asr_pipeline
                            )
                        # Transcribe last chunk into partial_transcript
                        else:
                            partial_transcript += get_audio_text(
                                chunk_data,
                                asr_pipeline
                            )
                        print_transcripts(
                            full_transcript,
                            segment_transcript,
                            partial_transcript
                        )
                        if verbose:
                            print_data(
                                getsizeof(master_buffer),
                                getsizeof(segment_buffer)
                            )
                    else:
                        consecutive_silent_chunks += 1
                        if consecutive_silent_chunks >= SEGMENT_SILENT_CHUNKS:
                            # If cutoff on silence finish and return master
                            if cutoff_on_silence:
                                if len(master_buffer) > 0:
                                    all_audio_data = np.concatenate(
                                        master_buffer,
                                        axis=0
                                    ).squeeze()
                                    full_transcript = get_audio_text(
                                        all_audio_data,
                                        asr_pipeline
                                    )
                                return full_transcript

                            # After long silence break out a segment
                            # into full_transcript which we dont re-refine
                            consecutive_silent_chunks = 0
                            chunk_count = 0
                            if (len(segment_transcript) > 1 or \
                                len(partial_transcript) > 1):
                                full_transcript += " " + segment_transcript
                                full_transcript += " " + partial_transcript
                            segment_buffer = []
                            segment_transcript  = ""
                            partial_transcript = ""
                            print_transcripts(
                                full_transcript,
                                segment_transcript,
                                partial_transcript
                            )
                            if verbose:
                                print_data(
                                    getsizeof(master_buffer),
                                    getsizeof(segment_buffer)
                                )
        except Exception as e:
            print(e)
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
            if len(master_buffer) > 0:
                all_audio_data = np.concatenate(
                    master_buffer,
                    axis=0
                ).squeeze()
                full_transcript = get_audio_text(
                    all_audio_data,
                    asr_pipeline
                )
            return full_transcript

if __name__ == "__main__":
  parser = ArgumentParser()
  parser.add_argument("-v", "--verbose", action="store_true")
  args = parser.parse_args()
  record(verbose=args.verbose)
