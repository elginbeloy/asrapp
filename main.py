import os
import sounddevice as sd
import queue
import numpy as np
import warnings

# Sizes available here:
# https://huggingface.co/openai/whisper-base
MODEL_TO_USE = "openai/whisper-small.en"

warnings.filterwarnings("ignore")

from transformers import pipeline, logging
logging.set_verbosity_error()

audio_queue = queue.Queue()

# Callback to add audio data to processing queue
def audio_callback(indata, frames, time, status):
    audio_queue.put(indata.copy())

def main():
    # Load pipeline
    print("Loading Whisper pipeline...")
    asr_pipeline = pipeline(
        "automatic-speech-recognition",
        model=MODEL_TO_USE,
        device="cuda:0"  # "cpu" if no GPU
    )
    print("Pipeline loaded.\n")

    # Parameters
    chunk_duration_seconds   = 3
    sample_rate        = 16000
    chunk_frames       = chunk_duration_seconds * sample_rate
    # Number of chunks before processing entire to refine
    CHUNKS_PER_REFINEMENT = 6

    # Buffer variables
    partial_buffer     = []
    master_buffer      = []
    frames_collected   = 0
    partial_transcript = ""
    full_transcript    = ""
    chunk_count        = 0

    # Start recording
    with sd.InputStream(
      samplerate=sample_rate,
      channels=1,
      dtype="float32",
      callback=audio_callback):
        print("Recording. Press Ctrl+C to stop.\n")
        try:
            while True:
                # Fetch audio from queue in small blocks
                block = audio_queue.get()
                partial_buffer.append(block)
                master_buffer.append(block)
                frames_collected += block.shape[0]

                # If we've collected enough frames for a chunk…
                if frames_collected >= chunk_frames:
                    frames_collected = 0
                    # Concatenate the partial chunk
                    chunk_data = np.concatenate(partial_buffer, axis=0).squeeze()
                    partial_buffer = []

                    # Partial ASR for quick feedback
                    partial_result = asr_pipeline({"array": chunk_data, "sampling_rate": sample_rate})
                    partial_transcript = partial_result["text"]
                    full_transcript += " " + partial_transcript
                    chunk_count += 1
                    os.system("clear")
                    print(full_transcript)

                    # Every N chunks, refine by reprocessing the entire audio
                    if chunk_count >= CHUNKS_PER_REFINEMENT:
                        chunk_count = 0
                        all_audio_data = np.concatenate(
                          master_buffer, axis=0).squeeze()
                        refined_result = asr_pipeline(
                            {
                              "array": all_audio_data,
                              "sampling_rate": sample_rate
                            }
                        )
                        refined_transcript = refined_result["text"]
                        full_transcript = refined_transcript

                        os.system("clear")
                        print(full_transcript)

        except KeyboardInterrupt:
            print("\nStopped by user.")
            # Refine transcript before saving and exiting
            all_audio_data = np.concatenate(
              master_buffer, axis=0).squeeze()
            refined_result = asr_pipeline(
                {
                  "array": all_audio_data,
                  "sampling_rate": sample_rate
                }
            )
            refined_transcript = refined_result["text"]
            full_transcript = refined_transcript
            pass

    # Write final transcript to file
    with open("memory.txt", "w", encoding="utf-8") as file:
        file.write(full_transcript)

    print("Final transcription saved to memory.txt\n")
    return full_transcript

if __name__ == "__main__":
    main()
