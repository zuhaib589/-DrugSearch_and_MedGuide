import pytesseract
from PIL import Image
import io
import torch
import torchaudio
from transformers import WhisperProcessor, WhisperForConditionalGeneration

# ------------------------------
# OCR with pytesseract
# ------------------------------
def extract_text_from_image(uploaded_file):
    image = Image.open(uploaded_file)
    text = pytesseract.image_to_string(image)
    return text

def clean_ocr_text(text):
    """
    Clean OCR text to make it safe for regex search.
    """
    import re
    cleaned = re.sub(r"[^a-zA-Z0-9\s]", "", text)
    return cleaned.strip()


# ------------------------------
# Voice to text with Whisper-Tiny via Hugging Face Transformers
# ------------------------------
processor = WhisperProcessor.from_pretrained("openai/whisper-tiny")
model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-tiny")
model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language="en", task="transcribe")

def speech_to_text_whisper(audio_file):
    # Load audio
    audio_input, sampling_rate = torchaudio.load(audio_file)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        audio_input = resampler(audio_input)

    input_features = processor(audio_input.squeeze(), sampling_rate=16000, return_tensors="pt").input_features
    predicted_ids = model.generate(input_features)
    transcription = processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
    return transcription

