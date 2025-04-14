import os
import torch
import pandas as pd
import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
from transformers import VitsModel, VitsTokenizer

import librosa
import torch.nn as nn
import torch.optim as optim
from tqdm.auto import tqdm
import json
import argparse
import random
from pathlib import Path
import soundfile as sf
import time

# Check GPU availability and enforce GPU usage
if not torch.cuda.is_available():
    raise RuntimeError("GPU is required but not available. Please make sure CUDA is installed and a GPU is available.")

device = torch.device("cuda")
print(f"Using device: {device}")

# Configure memory optimization for limited VRAM
torch.cuda.empty_cache()
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"

# Base directories
BASE_DIR = os.path.join(os.getcwd(), "dataset")
OUTPUT_DIR = "indian_tts_models-final"

# Training parameters
BATCH_SIZE = 2
EPOCHS = 5
LEARNING_RATE = 1e-6
TRAIN_RATIO = 0.9  # 90% for training, 10% for validation
SAMPLING_RATE = 16000
MAX_TEXT_LENGTH = 200
GRADIENT_ACCUMULATION_STEPS = 4
# Set a fixed length for all audio samples
TARGET_AUDIO_LENGTH = 100000  # About 6.25 seconds @ 16kHz

# Define the languages and their corresponding Meta F5 model IDs
LANGUAGES = {
    "hi": {
        "name": "Hindi",
        "model_id": "facebook/mms-tts-hin",
        "data_dir": os.path.join(BASE_DIR, "dataset-hi"),
        "metadata_file": "metadata_hi.csv",
        "audio_dir": "wavs",
    },
    "bn": {
        "name": "Bengali",
        "model_id": "facebook/mms-tts-ben",
        "data_dir": os.path.join(BASE_DIR, "dataset-bn"),
        "metadata_file": "metadata_bn.csv",
        "audio_dir": "wavs",
    },
    "gu": {
        "name": "Gujarati",
        "model_id": "facebook/mms-tts-guj",
        "data_dir": os.path.join(BASE_DIR, "dataset-gj"),
        "metadata_file": "metadata_gj.csv",
        "audio_dir": "wavs",
    },
    "kn": {
        "name": "Kannada",
        "model_id": "facebook/mms-tts-kan",
        "data_dir": os.path.join(BASE_DIR, "dataset-kn"),
        "metadata_file": "metadata_kn.csv",
        "audio_dir": "wavs",
    },
    "mr": {
        "name": "Marathi",
        "model_id": "facebook/mms-tts-mar",
        "data_dir": os.path.join(BASE_DIR, "dataset-mr"),
        "metadata_file": "metadata_mr.csv",
        "audio_dir": "wavs",
    },
    "pa": {
        "name": "Punjabi",
        "model_id": "facebook/mms-tts-pan",
        "data_dir": os.path.join(BASE_DIR, "dataset-pj"),
        "metadata_file": "metadata_pn.csv",
        "audio_dir": "wavs",
    },
    "ta": {
        "name": "Tamil",
        "model_id": "facebook/mms-tts-tam",
        "data_dir": os.path.join(BASE_DIR, "dataset-ta"),
        "metadata_file": "metadata_ta.csv",
        "audio_dir": "wavs",
    },
    "te": {
        "name": "Telugu",
        "model_id": "facebook/mms-tts-tel",
        "data_dir": os.path.join(BASE_DIR, "dataset-tl"),
        "metadata_file": "metadata_tl.csv",
        "audio_dir": "wavs",
    },
    "ur": {
        "name": "Urdu",
        "model_id": "facebook/mms-tts-urd",
        "data_dir": os.path.join(BASE_DIR, "dataset-ur"),
        "metadata_file": "metadata_ur.csv",
        "audio_dir": "wavs",
    }
}


# Dataset class for TTS fine-tuning
class TTSDataset(Dataset):
    def __init__(self, df, tokenizer, audio_dir, language_code):
        self.df = df
        self.tokenizer = tokenizer
        self.audio_dir = audio_dir
        self.language_code = language_code
        self.sampling_rate = SAMPLING_RATE
        self.target_length = TARGET_AUDIO_LENGTH

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        try:
            row = self.df.iloc[idx]

            # Using 'audio_file' column as specified in your metadata
            audio_filename = row['audio_file']

            # Validate text is a string - this is critical as we hit an error here
            text = row.get('text', '')
            if not isinstance(text, str):
                # Convert to string if it's not already a string
                try:
                    text = str(text)
                except:
                    # If conversion fails, use a placeholder
                    text = "placeholder text"
                print(f"Warning: Non-string text at index {idx} converted to: '{text}'")

            # Make sure text is not empty
            if not text.strip():
                text = "empty text"
                print(f"Warning: Empty text at index {idx}, using placeholder")

            # Construct full audio path
            audio_file = os.path.join(self.audio_dir, audio_filename)

            # Check if file exists
            if not os.path.exists(audio_file):
                # Try looking in different locations
                possible_paths = [
                    audio_file,
                    os.path.join(self.audio_dir, "wavs", audio_filename),
                    os.path.join(self.audio_dir, audio_filename),
                    audio_filename if os.path.isabs(audio_filename) else None
                ]

                for path in possible_paths:
                    if path and os.path.exists(path):
                        audio_file = path
                        break
                else:
                    raise FileNotFoundError(f"Audio file not found: {audio_filename}")

            # Tokenize text - this is where the original error occurred
            try:
                tokenized = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_LENGTH)
            except Exception as e:
                print(f"Error tokenizing text '{text}' at index {idx}: {e}")
                # Try with a simple placeholder text that should tokenize without issues
                text = "placeholder text"
                tokenized = self.tokenizer(text, return_tensors="pt", padding="max_length", max_length=MAX_TEXT_LENGTH)

            # Load and process audio
            waveform, sr = librosa.load(audio_file, sr=self.sampling_rate, mono=True)

            # Handle waveform length - trim or pad to target length
            if len(waveform) > self.target_length:
                waveform = waveform[:self.target_length]
            elif len(waveform) < self.target_length:
                padded_waveform = np.zeros(self.target_length)
                padded_waveform[:len(waveform)] = waveform
                waveform = padded_waveform

            # Optional: Normalize waveform
            # In your TTSDataset.__getitem__ method, replace the normalization with:
            waveform = librosa.util.normalize(waveform)
            # Add this to clip extreme values
            waveform = np.clip(waveform, -1.0, 1.0)

            # Convert to float32 tensor
            waveform = torch.tensor(waveform, dtype=torch.float32)

            return {
                "input_ids": tokenized.input_ids.squeeze(),
                "attention_mask": tokenized.attention_mask.squeeze(),
                "waveform": waveform,
                "audio_path": audio_file,
                "text": text,
                "language_code": self.language_code
            }
        except Exception as e:
            print(f"Error processing sample {idx}: {e}")
            # Return a fallback/dummy sample instead of raising an exception
            # This prevents the entire batch from failing
            fallback_text = "fallback text"
            fallback_tokenized = self.tokenizer(fallback_text, return_tensors="pt", padding="max_length",
                                                max_length=MAX_TEXT_LENGTH)
            fallback_waveform = torch.zeros(self.target_length, dtype=torch.float32)

            return {
                "input_ids": fallback_tokenized.input_ids.squeeze(),
                "attention_mask": fallback_tokenized.attention_mask.squeeze(),
                "waveform": fallback_waveform,
                "audio_path": "fallback_path",
                "text": fallback_text,
                "language_code": self.language_code
            }


# Modified collate function to filter out None values and handle issues
def collate_fn(batch):
    # Filter out any None or problematic samples
    valid_batch = [sample for sample in batch if sample is not None]

    # If all samples were filtered out, create a dummy batch
    if len(valid_batch) == 0:
        return None

    # Prepare lists for batch
    input_ids = []
    attention_masks = []
    waveforms = []
    texts = []
    language_codes = []

    for sample in valid_batch:
        try:
            input_ids.append(sample['input_ids'])
            attention_masks.append(sample['attention_mask'])
            texts.append(sample['text'])
            language_codes.append(sample['language_code'])
            waveforms.append(sample['waveform'])
        except Exception as e:
            print(f"Error in collate_fn with sample: {e}")
            continue

    # Check if we have any valid samples left
    if len(input_ids) == 0:
        return None

    # Stack tensors
    try:
        input_ids = torch.stack(input_ids)
        attention_masks = torch.stack(attention_masks)
        waveforms = torch.stack(waveforms)
    except Exception as e:
        print(f"Error stacking tensors in collate_fn: {e}")
        return None

    return {
        "input_ids": input_ids,
        "attention_mask": attention_masks,
        "waveforms": waveforms,
        "texts": texts,
        "language_codes": language_codes
    }


# Function to train a single language model
def train_language_model(lang_code):
    lang_info = LANGUAGES[lang_code]
    lang_name = lang_info["name"]
    model_id = lang_info["model_id"]
    data_dir = lang_info["data_dir"]
    metadata_file = os.path.join(data_dir, lang_info["metadata_file"])
    audio_dir = os.path.join(data_dir, lang_info["audio_dir"])
    lang_output_dir = os.path.join(OUTPUT_DIR, lang_code)

    # Create output directory if it doesn't exist
    os.makedirs(lang_output_dir, exist_ok=True)

    print(f"\n{'=' * 50}")
    print(f"Starting training for {lang_name} ({lang_code})")
    print(f"{'=' * 50}")

    # Print directory information
    print(f"Data directory: {os.path.abspath(data_dir)}")
    print(f"Metadata file: {os.path.abspath(metadata_file)}")
    print(f"Audio directory: {os.path.abspath(audio_dir)}")

    try:
        # Load metadata
        print("Loading metadata...")
        df = pd.read_csv(metadata_file)
        print(f"Dataset size: {len(df)} samples")
        print(f"Columns in metadata file: {df.columns.tolist()}")
        print(df.head())

        # Preprocessing: validate and clean text column
        print("Cleaning dataset text...")
        # Convert text column to string and handle NaN values
        df['text'] = df['text'].astype(str)
        # Remove rows with empty text
        df = df[df['text'].str.strip() != '']
        # Remove rows with text that's just 'nan'
        df = df[~df['text'].str.lower().isin(['nan', 'none', 'null'])]
        print(f"Dataset size after text cleaning: {len(df)} samples")

        # Load model and tokenizer
        print(f"Loading model and tokenizer for {lang_name}...")
        tokenizer = VitsTokenizer.from_pretrained(model_id)
        model = VitsModel.from_pretrained(model_id, low_cpu_mem_usage=True)

        # Model optimization for memory efficiency
        # Use mixed precision instead of pure FP16
        from torch.amp import autocast, GradScaler
        scaler = GradScaler()

        # Move model to device
        model = model.to(device)

        # Filter dataset to only include valid audio files
        valid_rows = []
        for idx, row in tqdm(df.iterrows(), desc="Validating audio files", total=len(df)):
            audio_filename = row['audio_file']
            audio_file = os.path.join(audio_dir, audio_filename)

            # Check if file exists in different possible locations
            file_exists = os.path.exists(audio_file)
            if not file_exists:
                possible_paths = [
                    os.path.join(audio_dir, "wavs", audio_filename),
                    os.path.join(audio_dir, audio_filename),
                    audio_filename if os.path.isabs(audio_filename) else None
                ]
                file_exists = any(path and os.path.exists(path) for path in possible_paths)

            if file_exists:
                valid_rows.append(idx)

        filtered_df = df.loc[valid_rows]
        print(f"Original dataset: {len(df)} samples")
        print(f"Filtered dataset: {len(filtered_df)} samples with valid audio files")

        if len(filtered_df) == 0:
            print(f"No valid audio files found for {lang_name}. Skipping training.")
            return False

        # Create full dataset
        full_dataset = TTSDataset(filtered_df, tokenizer, audio_dir, lang_code)

        # Split dataset into train and validation
        train_size = int(TRAIN_RATIO * len(full_dataset))
        val_size = len(full_dataset) - train_size
        train_dataset, val_dataset = random_split(full_dataset, [train_size, val_size])

        print(f"Training set: {len(train_dataset)} samples")
        print(f"Validation set: {len(val_dataset)} samples")

        # Create DataLoaders with robust error handling
        train_dataloader = DataLoader(
            train_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=True,
            num_workers=0,  # Reduced workers for better stability
            pin_memory=True  # Added for faster data transfer to GPU
        )

        val_dataloader = DataLoader(
            val_dataset,
            batch_size=BATCH_SIZE,
            collate_fn=collate_fn,
            shuffle=False,
            num_workers=0,  # Reduced workers for better stability
            pin_memory=True  # Added for faster data transfer to GPU
        )

        # Training setup
        optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE)
        scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS)

        # Save best model parameters
        best_val_loss = float('inf')
        best_epoch = 0

        # Properly define the loss computation function to handle tensor shape mismatches
        def compute_loss(pred_waveform, target_waveform):
            # Make sure both are on the same device
            if pred_waveform.device != target_waveform.device:
                target_waveform = target_waveform.to(pred_waveform.device)

            # Resize if needed (your existing code)
            if pred_waveform.size(-1) != target_waveform.size(-1):
                if pred_waveform.size(-1) > target_waveform.size(-1):
                    pred_waveform = pred_waveform[..., :target_waveform.size(-1)]
                else:
                    padding_size = target_waveform.size(-1) - pred_waveform.size(-1)
                    pred_waveform = torch.nn.functional.pad(pred_waveform, (0, padding_size))



            # Ensure both are of the same dtype
            target_waveform = target_waveform.to(pred_waveform.dtype)

            # Check for NaN values
            if torch.isnan(pred_waveform).any() or torch.isnan(target_waveform).any():
                print("Warning: NaN values detected in waveforms before loss calculation")
                # Replace NaNs with zeros
                pred_waveform = torch.nan_to_num(pred_waveform)
                target_waveform = torch.nan_to_num(target_waveform)

            # L1 loss is often more stable than MSE for audio
            return nn.L1Loss()(pred_waveform, target_waveform)

        # Training loop
        for epoch in range(EPOCHS):
            # Training
            model.train()
            train_loss = 0
            train_batches_processed = 0
            train_start_time = time.time()

            train_progress_bar = tqdm(train_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Train]")
            for step, batch in enumerate(train_progress_bar):
                # Skip None batches (from collate_fn error handling)
                if batch is None:
                    continue

                try:
                    # Move data to GPU
                    input_ids = batch['input_ids'].to(device)
                    attention_mask = batch['attention_mask'].to(device)
                    waveforms = batch['waveforms'].to(device)

                    # Zero the gradients
                    optimizer.zero_grad()

                    # Forward pass
                    outputs = model(
                        input_ids=input_ids,
                        attention_mask=attention_mask,
                        speaker_id=torch.zeros(input_ids.shape[0], dtype=torch.long).to(device),  # default speaker
                        return_dict=True
                    )

                    # Calculate loss using the proper function
                    loss = compute_loss(outputs.waveform, waveforms)

                    # Apply gradient accumulation
                    loss = loss / GRADIENT_ACCUMULATION_STEPS
                    scaler.scale(loss).backward()

                    # Add after loss.backward() but before optimizer.step()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)

                    # Add this function and call it at key points in your training loop
                    def check_for_nan_tensors(model, step, location="unknown"):
                        for name, param in model.named_parameters():
                            if torch.isnan(param).any() or torch.isinf(param).any():
                                print(f"NaN or Inf detected in {name} at {location}, step {step}")
                                return True
                        return False

                    # Call this before and after forward pass, after loss calculation, and after backward pass


                    # Step optimizer only after accumulating gradients
                    if (step + 1) % GRADIENT_ACCUMULATION_STEPS == 0:
                        scaler.step(optimizer)
                        scaler.update()
                        optimizer.zero_grad()

                    train_loss += loss.item() * GRADIENT_ACCUMULATION_STEPS
                    train_batches_processed += 1
                    train_progress_bar.set_postfix({"loss": loss.item() * GRADIENT_ACCUMULATION_STEPS})

                except Exception as e:
                    print(f"Error in training batch {step}: {e}")
                    continue

            # Avoid division by zero
            if train_batches_processed == 0:
                print("No valid training batches processed in this epoch. Skipping to next epoch.")
                continue

            avg_train_loss = train_loss / train_batches_processed
            train_time = time.time() - train_start_time

            # Validation
            model.eval()
            val_loss = 0
            val_batches_processed = 0
            val_start_time = time.time()

            val_progress_bar = tqdm(val_dataloader, desc=f"Epoch {epoch + 1}/{EPOCHS} [Val]")
            with torch.no_grad():
                for batch in val_progress_bar:
                    # Skip None batches
                    if batch is None:
                        continue

                    try:
                        # Move data to GPU
                        input_ids = batch['input_ids'].to(device)
                        attention_mask = batch['attention_mask'].to(device)
                        waveforms = batch['waveforms'].to(device)

                        # Forward pass
                        with autocast("cuda"):
                            outputs = model(
                                input_ids=input_ids,
                                attention_mask=attention_mask,
                                speaker_id=torch.zeros(input_ids.shape[0], dtype=torch.long).to(device),
                                return_dict=True
                            )
                            loss = compute_loss(outputs.waveform, waveforms)
                            loss = loss / GRADIENT_ACCUMULATION_STEPS

                        # Calculate validation loss
                        loss = compute_loss(outputs.waveform, waveforms)
                        val_loss += loss.item()
                        val_batches_processed += 1
                        val_progress_bar.set_postfix({"loss": loss.item()})
                    except Exception as e:
                        print(f"Error in validation batch: {e}")
                        continue

            # Avoid division by zero
            if val_batches_processed == 0:
                print("No valid validation batches processed in this epoch. Using previous best val loss.")
                avg_val_loss = best_val_loss
            else:
                avg_val_loss = val_loss / val_batches_processed

            val_time = time.time() - val_start_time

            # Update learning rate
            scheduler.step()

            # Print epoch summary
            print(
                f"Epoch {epoch + 1}/{EPOCHS} - Train Loss: {avg_train_loss:.4f} ({train_time:.2f}s), Val Loss: {avg_val_loss:.4f} ({val_time:.2f}s)")

            # Save checkpoint
            checkpoint_path = os.path.join(lang_output_dir, f"checkpoint_epoch_{epoch + 1}.pt")
            torch.save({
                'epoch': epoch + 1,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'scheduler_state_dict': scheduler.state_dict(),
                'train_loss': avg_train_loss,
                'val_loss': avg_val_loss,
            }, checkpoint_path)

            # Save best model
            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                best_epoch = epoch + 1
                best_model_path = os.path.join(lang_output_dir, "best_model.pt")
                torch.save(model.state_dict(), best_model_path)
                print(f"New best model saved with validation loss: {best_val_loss:.4f}")

        # After training - save final model
        print(f"Training complete for {lang_name}")
        print(f"Best model was from epoch {best_epoch} with validation loss: {best_val_loss:.4f}")

        # Save final model and tokenizer in HuggingFace format
        final_model_path = os.path.join(lang_output_dir, "final_model")
        model.save_pretrained(final_model_path)
        tokenizer.save_pretrained(final_model_path)
        print(f"Final model saved to {final_model_path}")

        # Generate a sample audio using the fine-tuned model
        try:
            sample_text = filtered_df.iloc[0]['text']
            sample_file = os.path.join(lang_output_dir, "sample_output.wav")

            print(f"Generating sample audio for: '{sample_text}'")

            # Use model to generate sample
            inputs = tokenizer(sample_text, return_tensors="pt").to(device)
            with torch.no_grad():
                output = model.generate_speech(inputs["input_ids"])

            # Save the audio
            waveform = output.cpu().numpy()
            sf.write(sample_file, waveform, SAMPLING_RATE)
            print(f"Sample audio saved to {sample_file}")

        except Exception as e:
            print(f"Error generating sample audio: {e}")

        return True

    except Exception as e:
        print(f"Error training model for {lang_name}: {e}")
        import traceback
        traceback.print_exc()
        return False


# Create a simple inference script template
def create_inference_script():
    script = """
import torch
from transformers import VitsModel, VitsTokenizer
import soundfile as sf
import argparse
import os

def synthesize_speech(text, lang_code, output_file):
    # Check for GPU
    if not torch.cuda.is_available():
        print("Warning: GPU not available. Using CPU instead, which might be slow.")
        device = torch.device("cpu")
    else:
        device = torch.device("cuda")

    # Load fine-tuned model and tokenizer
    model_dir = f"indian_tts_models/{lang_code}/final_model"

    if not os.path.exists(model_dir):
        print(f"Error: Model not found for language code '{lang_code}'")
        return False

    # Load tokenizer and model
    tokenizer = VitsTokenizer.from_pretrained(model_dir)
    model = VitsModel.from_pretrained(model_dir)

    # Move to GPU
    model = model.to(device)

    # Tokenize text
    inputs = tokenizer(text, return_tensors="pt").to(device)

    # Generate speech
    with torch.no_grad():
        output = model.generate_speech(inputs["input_ids"])

    # Save audio
    waveform = output.cpu().numpy()
    sf.write(output_file, waveform, samplerate=16000)
    print(f"Speech synthesized and saved to {output_file}")
    return True

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Indian Languages TTS Synthesis")
    parser.add_argument("--text", type=str, required=True, help="Text to synthesize")
    parser.add_argument("--lang", type=str, required=True, help="Language code (hi, bn, gu, kn, ml, mr, pa, ta, te, ur)")
    parser.add_argument("--output", type=str, default="output.wav", help="Output audio file path")

    args = parser.parse_args()
    synthesize_speech(args.text, args.lang, args.output)
"""

    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Write the script
    with open(os.path.join(OUTPUT_DIR, "synthesize.py"), "w") as f:
        f.write(script)

    print(f"Inference script created at {os.path.join(OUTPUT_DIR, 'synthesize.py')}")


# Main function to manage training of all languages
def main():
    # Verify GPU is available
    if not torch.cuda.is_available():
        raise RuntimeError("This script requires a GPU to run. Please make sure your GPU is properly set up.")

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Save language configuration
    with open(os.path.join(OUTPUT_DIR, "language_config.json"), "w") as f:
        json.dump(LANGUAGES, f, indent=4)

    # Parse arguments
    parser = argparse.ArgumentParser(description="Train TTS models for Indian languages")
    parser.add_argument("--langs", type=str, default="all",
                        help="Comma-separated language codes to train (default: all)")
    parser.add_argument("--epochs", type=int, default=5,
                        help="Number of training epochs (default: 5)")
    parser.add_argument("--batch_size", type=int, default=2,
                        help="Batch size (default: 2)")
    parser.add_argument("--lr", type=float, default=1e-5,
                        help="Learning rate (default: 1e-5)")

    args = parser.parse_args()

    # Update global parameters based on args
    global EPOCHS, BATCH_SIZE, LEARNING_RATE
    EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    LEARNING_RATE = args.lr

    # Print GPU info
    print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    print(f"CUDA Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

    # Determine which languages to train
    langs_to_train = list(LANGUAGES.keys()) if args.langs == "all" else args.langs.split(",")

    # Create inference script
    create_inference_script()

    # Record results
    results = {}

    # Train each language model
    for lang_code in langs_to_train:
        if lang_code in LANGUAGES:
            print(f"\nPreparing to train {LANGUAGES[lang_code]['name']} model...")
            success = train_language_model(lang_code)
            results[lang_code] = "Success" if success else "Failed"
        else:
            print(f"Warning: Language code '{lang_code}' not recognized. Skipping.")
            results[lang_code] = "Not recognized"

    # Print summary
    print("\n" + "=" * 60)
    print("Training Summary")
    print("=" * 60)
    for lang_code, status in results.items():
        if lang_code in LANGUAGES:
            print(f"{LANGUAGES[lang_code]['name']} ({lang_code}): {status}")
        else:
            print(f"Unknown language ({lang_code}): {status}")

    print("\nTo synthesize speech with your trained models, use:")
    print(
        f"python {os.path.join(OUTPUT_DIR, 'synthesize.py')} --text \"Your text here\" --lang LANG_CODE --output output.wav")
    print("Where LANG_CODE is one of:", ", ".join(LANGUAGES.keys()))


if __name__ == "__main__":
    main()
