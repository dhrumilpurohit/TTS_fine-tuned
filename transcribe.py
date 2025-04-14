import os
import argparse
import pandas as pd
import subprocess
import torch
import sys
import io
import codecs
from tqdm import tqdm
from transformers import AutoProcessor, AutoModelForCTC
import soundfile as sf
import warnings

# Suppress warnings
warnings.filterwarnings("ignore")

# Ensure proper Unicode handling
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')


def convert_audio(input_file, output_file=None):
    """Convert audio to 16kHz mono WAV format using ffmpeg."""
    if output_file is None:
        base_dir = os.path.dirname(input_file)
        basename = os.path.basename(input_file)
        name, _ = os.path.splitext(basename)
        output_file = os.path.join(base_dir, f"{name}_16k.wav")

    command = [
        "ffmpeg", "-y", "-i", input_file,
        "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        "-loglevel", "error",  # Suppress ffmpeg output
        output_file
    ]

    try:
        subprocess.run(command, check=True, capture_output=True)
        return output_file
    except subprocess.CalledProcessError as e:
        print(f"Error converting {input_file}: {e.stderr.decode('utf-8')}")
        return None
    except FileNotFoundError:
        print("Error: ffmpeg not found. Please install ffmpeg and ensure it's in your PATH.")
        return None


def load_gujarati_model(device):
    """Load Gujarati speech recognition model."""
    model_choices = [

        "addy88/wav2vec2-punjabi-stt",  # Another option

    ]

    for model_name in model_choices:
        try:
            print(f"Loading model: {model_name}")
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForCTC.from_pretrained(model_name).to(device)
            print(f"Successfully loaded model: {model_name}")
            return processor, model
        except Exception as e:
            print(f"Error loading {model_name}: {str(e)}")
            continue

    raise ValueError("Could not load any suitable Gujarati speech recognition model")


def transcribe_gujarati_audio(input_directory, output_csv, use_gpu=True):
    """
    Transcribe Gujarati audio files and save results to a CSV file.

    Args:
        input_directory: Directory containing audio files
        output_csv: Path to save output CSV
        use_gpu: Whether to use GPU if available
    """
    # Set device
    device = "cuda" if use_gpu and torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")

    try:
        processor, model = load_gujarati_model(device)
    except ValueError as e:
        print(e)
        return

    # Find all audio files
    audio_files = []
    for root, _, files in os.walk(input_directory):
        for file in files:
            if file.lower().endswith(('.wav', '.mp3', '.flac', '.ogg', '.m4a')):
                audio_files.append(os.path.join(root, file))

    print(f"Found {len(audio_files)} Gujarati audio files")
    results = []

    # Process each file
    for audio_path in tqdm(audio_files, desc="Processing files"):
        try:
            # Convert to 16kHz mono WAV
            converted_path = convert_audio(audio_path)
            if not converted_path or not os.path.exists(converted_path):
                print(f"Could not convert {audio_path}")
                results.append({
                    "audio_path": audio_path,
                    "transcription": "[CONVERSION_ERROR]"
                })
                continue

            # Load audio
            try:
                audio_input, sample_rate = sf.read(converted_path)

                # Ensure proper audio format
                if len(audio_input.shape) > 1:
                    audio_input = audio_input.mean(axis=1)  # Convert to mono if stereo

                # Normalize audio
                if audio_input.dtype != 'float32':
                    audio_input = audio_input.astype('float32') / 32768.0

                # Resample if needed
                if sample_rate != 16000:
                    import librosa
                    audio_input = librosa.resample(audio_input, orig_sr=sample_rate, target_sr=16000)
            except Exception as e:
                print(f"Error loading {audio_path}: {str(e)}")
                results.append({
                    "audio_path": audio_path,
                    "transcription": "[AUDIO_LOAD_ERROR]"
                })
                continue

            # Process through model
            try:
                input_values = processor(
                    audio_input,
                    sampling_rate=16000,
                    return_tensors="pt",
                    padding=True
                ).input_values.to(device)

                with torch.no_grad():
                    logits = model(input_values).logits

                # Get transcription
                predicted_ids = torch.argmax(logits, dim=-1)
                transcription = processor.batch_decode(predicted_ids)[0]
            except Exception as e:
                print(f"Error processing {audio_path}: {str(e)}")
                transcription = "[TRANSCRIPTION_ERROR]"

            # Store result
            results.append({
                "audio_path": audio_path,
                "transcription": transcription.strip()
            })

            # Clean up temporary file
            if os.path.exists(converted_path) and converted_path != audio_path:
                os.remove(converted_path)

        except Exception as e:
            print(f"Fatal error processing {audio_path}: {str(e)}")
            import traceback
            traceback.print_exc()

    # Save results
    if not results:
        print("No results to save.")
        return

    df = pd.DataFrame(results)

    # Enhanced saving with multiple formats to ensure proper unicode handling for Gujarati
    try:
        # Method 1: Write to TSV with explicit BOM and tab separator
        with codecs.open(output_csv.replace('.csv', '.tsv'), 'w', encoding='utf-8-sig') as f:
            f.write("audio_path\ttranscription\n")  # Header with tab separator
            for _, row in df.iterrows():
                # Clean the path to avoid tab issues
                clean_path = row['audio_path'].replace('\t', ' ')
                # Write with tab separator
                f.write(f"{clean_path}\t{row['transcription']}\n")

        print(f"Transcriptions saved to {output_csv.replace('.csv', '.tsv')} (TSV format)")



        # Method 4: Save CSV with utf-8-sig encoding for proper BOM
        df.to_csv(output_csv, index=False, encoding='utf-8-sig')
        print(f"Saved as CSV file with UTF-8-BOM: {output_csv}")

    except Exception as e:
        print(f"Error saving output files: {str(e)}")

    # Output sample results
    print("\nSample Gujarati Transcriptions:")
    for i, row in df.head(min(3, len(df))).iterrows():
        print(f"File: {os.path.basename(row['audio_path'])}")
        print(f"Text: {row['transcription']}")
        print("---")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Transcribe Gujarati audio files")
    parser.add_argument("--input_dir", required=True, help="Directory with audio files")
    parser.add_argument("--output_csv", default="punjabi_transcriptions.csv", help="Output CSV file")
    parser.add_argument("--use_cpu", action="store_true", help="Force CPU usage")

    args = parser.parse_args()

    transcribe_gujarati_audio(
        input_directory=args.input_dir,
        output_csv=args.output_csv,
        use_gpu=not args.use_cpu
    )
