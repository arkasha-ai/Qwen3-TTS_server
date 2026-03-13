import os
import shutil
import time
import torch
from fastapi import FastAPI, UploadFile, File, HTTPException, Form, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from typing import Optional
import soundfile as sf
from pydub import AudioSegment, silence
import io
import magic
import logging
import numpy as np

logging.basicConfig(level=logging.INFO)

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

device = "cuda:0" if torch.cuda.is_available() else "cpu"

# Initialize Qwen3-TTS model
from qwen_tts import Qwen3TTSModel

logging.info(f"Loading Qwen3-TTS model on {device}...")
model = Qwen3TTSModel.from_pretrained(
    "Qwen/Qwen3-TTS-12Hz-1.7B-Base",
    device_map=device,
    dtype=torch.bfloat16,
)
logging.info("Qwen3-TTS model loaded successfully")

# Initialize Whisper for transcription (Qwen3-TTS doesn't have built-in transcription)
import whisper

logging.info("Loading Whisper model for transcription...")
whisper_model = whisper.load_model("base", device=device)
logging.info("Whisper model loaded successfully")

output_dir = 'outputs'
os.makedirs(output_dir, exist_ok=True)

resources_dir = 'resources'
os.makedirs(resources_dir, exist_ok=True)

# Default reference audio and text for base_tts
default_ref_audio = None
default_ref_text = "Some call me nature, others call me mother nature."
default_voice_prompt = None

# Cache for voice data: {"{voice}__{emotion}": {"processed_audio": path, "ref_text": str, "prompt": object}}
voice_cache = {}


def convert_to_wav(input_path: str, output_path: str):
    """Convert any audio format to WAV using pydub."""
    audio = AudioSegment.from_file(input_path)
    audio = audio.set_channels(1)  # Convert to mono
    audio = audio.set_frame_rate(24000)  # Set sample rate
    audio.export(output_path, format='wav')


def transcribe_audio(audio_path: str) -> str:
    """Transcribe audio using Whisper."""
    result = whisper_model.transcribe(audio_path)
    return result["text"].strip()


def detect_leading_silence(audio, silence_threshold=-42, chunk_size=10):
    """Detect silence at the beginning of the audio."""
    trim_ms = 0
    while audio[trim_ms:trim_ms + chunk_size].dBFS < silence_threshold and trim_ms < len(audio):
        trim_ms += chunk_size
    return trim_ms


def remove_silence_edges(audio, silence_threshold=-42):
    """Remove silence from the beginning and end of the audio."""
    start_trim = detect_leading_silence(audio, silence_threshold)
    end_trim = detect_leading_silence(audio.reverse(), silence_threshold)
    duration = len(audio)
    return audio[start_trim:duration - end_trim]


def process_reference_audio(reference_file: str) -> tuple[str, str]:
    """
    Process reference audio: clip to max 15s and transcribe.
    Returns (processed_audio_path, transcription).
    """
    temp_short_ref = f'{output_dir}/temp_short_ref.wav'
    aseg = AudioSegment.from_file(reference_file)

    # 1. try to find long silence for clipping
    non_silent_segs = silence.split_on_silence(
        aseg, min_silence_len=1000, silence_thresh=-50, keep_silence=1000, seek_step=10
    )
    non_silent_wave = AudioSegment.silent(duration=0)
    for non_silent_seg in non_silent_segs:
        if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
            logging.info("Audio is over 15s, clipping short. (1)")
            break
        non_silent_wave += non_silent_seg

    # 2. try to find short silence for clipping if 1. failed
    if len(non_silent_wave) > 15000:
        non_silent_segs = silence.split_on_silence(
            aseg, min_silence_len=100, silence_thresh=-40, keep_silence=1000, seek_step=10
        )
        non_silent_wave = AudioSegment.silent(duration=0)
        for non_silent_seg in non_silent_segs:
            if len(non_silent_wave) > 6000 and len(non_silent_wave + non_silent_seg) > 15000:
                logging.info("Audio is over 15s, clipping short. (2)")
                break
            non_silent_wave += non_silent_seg

    aseg = non_silent_wave

    # 3. if no proper silence found for clipping
    if len(aseg) > 15000:
        aseg = aseg[:15000]
        logging.info("Audio is over 15s, clipping short. (3)")

    aseg = remove_silence_edges(aseg) + AudioSegment.silent(duration=50)
    aseg.export(temp_short_ref, format='wav')

    # Transcribe the short clip
    ref_text = transcribe_audio(temp_short_ref)
    logging.info(f'Reference text transcribed from first 15s: {ref_text}')

    return temp_short_ref, ref_text


def apply_speed(audio_data: np.ndarray, sr: int, speed: float) -> np.ndarray:
    """Apply speed adjustment to audio using time stretching."""
    if speed == 1.0:
        return audio_data
    
    try:
        import librosa
        # Time stretch: speed > 1 = faster, speed < 1 = slower
        return librosa.effects.time_stretch(audio_data, rate=speed)
    except ImportError:
        logging.warning("librosa not installed, speed adjustment not available")
        return audio_data


def generate_speech_with_prompt(text: str, voice_prompt, speed: float = 1.0) -> tuple[np.ndarray, int]:
    """Generate speech using cached voice clone prompt (optimized)."""
    import time
    start_time = time.time()
    
    # Set fixed seed for reproducible output
    torch.manual_seed(42)
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Auto",
        voice_clone_prompt=voice_prompt,
    )
    
    audio_data = wavs[0]
    
    # Apply speed adjustment if needed
    if speed != 1.0:
        audio_data = apply_speed(audio_data, sr, speed)
    
    generation_time = time.time() - start_time
    audio_duration = len(audio_data) / sr
    logging.info(f"Generation completed in {generation_time:.2f}s (audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)")
    
    return audio_data, sr


def generate_speech(text: str, ref_audio_path: str, ref_text: str, speed: float = 1.0) -> tuple[np.ndarray, int]:
    """Generate speech using Qwen3-TTS voice cloning (non-cached fallback)."""
    import time
    start_time = time.time()
    
    # Set fixed seed for reproducible output
    torch.manual_seed(42)
    
    # Load reference audio as numpy array
    ref_audio_data, ref_sr = sf.read(ref_audio_path)
    
    wavs, sr = model.generate_voice_clone(
        text=text,
        language="Auto",
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
    )
    
    audio_data = wavs[0]
    
    # Apply speed adjustment if needed
    if speed != 1.0:
        audio_data = apply_speed(audio_data, sr, speed)
    
    generation_time = time.time() - start_time
    audio_duration = len(audio_data) / sr
    logging.info(f"Generation completed in {generation_time:.2f}s (audio duration: {audio_duration:.2f}s, RTF: {generation_time/audio_duration:.2f}x)")
    
    return audio_data, sr


def audio_to_wav_bytes(audio_data: np.ndarray, sr: int) -> io.BytesIO:
    """Convert numpy audio array to WAV bytes."""
    buffer = io.BytesIO()
    sf.write(buffer, audio_data, sr, format='WAV')
    buffer.seek(0)
    return buffer


def find_voice_reference(voice: str, emotion: str = "neutral") -> Optional[str]:
    """
    Find reference audio file for a voice+emotion combo.
    
    Search order:
    1. resources/{voice}/{emotion}.wav (new format)
    2. resources/{voice}/neutral.wav (fallback emotion)
    3. resources/{voice}.wav or resources/{voice}.* (legacy format)
    """
    # New format: resources/{voice}/{emotion}.wav
    voice_dir = os.path.join(resources_dir, voice)
    if os.path.isdir(voice_dir):
        emotion_file = os.path.join(voice_dir, f"{emotion}.wav")
        if os.path.isfile(emotion_file):
            return emotion_file
        # Fallback to neutral if requested emotion not found
        if emotion != "neutral":
            neutral_file = os.path.join(voice_dir, "neutral.wav")
            if os.path.isfile(neutral_file):
                logging.info(f"Emotion '{emotion}' not found for voice '{voice}', falling back to 'neutral'")
                return neutral_file
        # Try any file in the directory
        for f in os.listdir(voice_dir):
            if f.endswith('.wav'):
                logging.info(f"Using first available emotion '{f}' for voice '{voice}'")
                return os.path.join(voice_dir, f)

    # Legacy format: resources/{voice}.wav or resources/{voice}.*
    # First try WAV
    legacy_wav = os.path.join(resources_dir, f"{voice}.wav")
    if os.path.isfile(legacy_wav):
        return legacy_wav

    # Then try any matching file
    for f in os.listdir(resources_dir):
        name_without_ext = os.path.splitext(f)[0]
        if name_without_ext == voice and os.path.isfile(os.path.join(resources_dir, f)):
            file_path = os.path.join(resources_dir, f)
            # Convert to WAV if needed
            if not f.lower().endswith('.wav'):
                wav_path = os.path.join(output_dir, 'ref_converted.wav')
                convert_to_wav(file_path, wav_path)
                return wav_path
            return file_path

    return None


def get_cache_key(voice: str, emotion: str = "neutral") -> str:
    """Build cache key from voice and emotion."""
    return f"{voice}__{emotion}"


def get_or_create_voice_cache(voice: str, reference_file: str, emotion: str = "neutral") -> dict:
    """
    Get cached voice data or create new cache entry.
    Caches: processed audio path, transcription, and voice clone prompt.
    This avoids repeated Whisper transcription on every request.
    """
    global voice_cache
    
    cache_key = get_cache_key(voice, emotion)
    
    if cache_key in voice_cache:
        logging.info(f"Using cached voice data for: {cache_key}")
        return voice_cache[cache_key]
    
    logging.info(f"Creating voice cache for: {cache_key}")
    
    # Process reference audio (clip to 15s, remove silence)
    processed_ref, ref_text = process_reference_audio(reference_file)
    
    # Create reusable voice clone prompt
    ref_audio_data, ref_sr = sf.read(processed_ref)
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
    )
    
    # Store in cache
    voice_cache[cache_key] = {
        "processed_audio": processed_ref,
        "ref_text": ref_text,
        "prompt": voice_prompt,
        "audio_data": ref_audio_data,
        "sample_rate": ref_sr,
    }
    
    logging.info(f"Voice cache created for: {cache_key} (transcription: '{ref_text[:50]}...')")
    return voice_cache[cache_key]


def invalidate_voice_cache(voice: str, emotion: Optional[str] = None):
    """Invalidate cache entries for a voice. If emotion is None, invalidate all emotions."""
    global voice_cache
    keys_to_delete = []
    prefix = f"{voice}__"
    if emotion:
        key = get_cache_key(voice, emotion)
        if key in voice_cache:
            keys_to_delete.append(key)
    else:
        keys_to_delete = [k for k in voice_cache if k.startswith(prefix)]
    for k in keys_to_delete:
        del voice_cache[k]
        logging.info(f"Cleared voice cache: {k}")


def list_voices() -> dict[str, list[str]]:
    """
    List all available voices and their emotions.
    Scans both new format (directories) and legacy format (flat files).
    """
    voices = {}
    
    for entry in os.listdir(resources_dir):
        entry_path = os.path.join(resources_dir, entry)
        
        if os.path.isdir(entry_path):
            # New format: directory with emotion files
            emotions = []
            for f in sorted(os.listdir(entry_path)):
                if f.endswith('.wav'):
                    emotions.append(os.path.splitext(f)[0])
            if emotions:
                voices[entry] = emotions
        elif os.path.isfile(entry_path):
            # Legacy format: flat file
            name = os.path.splitext(entry)[0]
            if name not in voices:
                voices[name] = ["neutral"]
    
    return voices


@app.on_event("startup")
async def startup_event():
    """Warmup the model on startup."""
    global default_ref_audio, default_voice_prompt
    
    # Check if we have a default voice file
    ref = find_voice_reference("default_en")
    if ref:
        default_ref_audio = ref
    
    # Warmup with demo_speaker0 if available
    demo_ref = find_voice_reference("demo_speaker0")
    if demo_ref:
        logging.info("Warming up model with demo_speaker0...")
        test_text = "This is a test sentence generated by the Qwen3-TTS API."
        try:
            await synthesize_speech(test_text, "demo_speaker0")
            logging.info("Warmup complete")
        except Exception as e:
            logging.warning(f"Warmup failed: {e}")


# ─── Voice management endpoints ───

@app.get("/voices/")
async def get_voices():
    """List all available voices and their emotions."""
    return {"voices": list_voices()}


@app.post("/voices/{name}/emotions/{emotion}")
async def upload_voice_emotion(name: str, emotion: str, file: UploadFile = File(...)):
    """Upload a reference audio for a specific voice + emotion."""
    try:
        contents = await file.read()

        allowed_extensions = {'wav', 'mp3', 'flac', 'ogg'}
        max_file_size = 5 * 1024 * 1024

        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type. Allowed: wav, mp3, flac, ogg")

        if len(contents) > max_file_size:
            raise HTTPException(status_code=400, detail="File size over 5MB limit.")

        temp_file = io.BytesIO(contents)
        file_format = magic.from_buffer(temp_file.read(), mime=True)
        if 'audio' not in file_format:
            raise HTTPException(status_code=400, detail="Invalid file content.")

        voice_dir = os.path.join(resources_dir, name)
        os.makedirs(voice_dir, exist_ok=True)

        # Save original
        temp_path = os.path.join(voice_dir, f"{emotion}_orig.{file_ext}")
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Convert to WAV
        wav_path = os.path.join(voice_dir, f"{emotion}.wav")
        convert_to_wav(temp_path, wav_path)

        # Remove temp original if different from wav
        if temp_path != wav_path and os.path.exists(temp_path):
            os.remove(temp_path)

        invalidate_voice_cache(name, emotion)

        return {"message": f"Voice '{name}' emotion '{emotion}' uploaded successfully."}
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error uploading voice emotion: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.delete("/voices/{name}")
async def delete_voice(name: str):
    """Delete a voice and all its emotions."""
    voice_dir = os.path.join(resources_dir, name)
    deleted = False

    # Delete directory format
    if os.path.isdir(voice_dir):
        shutil.rmtree(voice_dir)
        deleted = True

    # Delete legacy flat files
    for f in os.listdir(resources_dir):
        if os.path.splitext(f)[0] == name and os.path.isfile(os.path.join(resources_dir, f)):
            os.remove(os.path.join(resources_dir, f))
            deleted = True

    if not deleted:
        raise HTTPException(status_code=404, detail=f"Voice '{name}' not found.")

    invalidate_voice_cache(name)
    return {"message": f"Voice '{name}' deleted."}


@app.delete("/voices/{name}/emotions/{emotion}")
async def delete_voice_emotion(name: str, emotion: str):
    """Delete a specific emotion from a voice."""
    voice_dir = os.path.join(resources_dir, name)
    emotion_file = os.path.join(voice_dir, f"{emotion}.wav")

    if not os.path.isfile(emotion_file):
        raise HTTPException(status_code=404, detail=f"Emotion '{emotion}' not found for voice '{name}'.")

    os.remove(emotion_file)
    invalidate_voice_cache(name, emotion)

    # If directory is now empty, remove it
    if os.path.isdir(voice_dir) and not os.listdir(voice_dir):
        os.rmdir(voice_dir)

    return {"message": f"Emotion '{emotion}' deleted from voice '{name}'."}


# ─── Existing endpoints (updated) ───

@app.get("/base_tts/")
async def base_tts(text: str, speed: Optional[float] = 1.0):
    """
    Perform text-to-speech conversion using only the base speaker.
    """
    try:
        return await synthesize_speech(text=text, voice="default_en", speed=speed)
    except Exception as e:
        logging.error(f"Error in base_tts: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/change_voice/")
async def change_voice(reference_speaker: str = Form(...), file: UploadFile = File(...)):
    """
    Change the voice of an existing audio file.
    """
    try:
        logging.info(f'Changing voice to {reference_speaker}...')

        contents = await file.read()
        
        # Save the input audio temporarily
        input_path = f'{output_dir}/input_audio.wav'
        with open(input_path, 'wb') as f:
            f.write(contents)

        # Find reference audio
        reference_file = find_voice_reference(reference_speaker)
        if not reference_file:
            raise HTTPException(status_code=400, detail="No matching reference speaker found.")
        
        # Transcribe the input audio
        text = transcribe_audio(input_path)
        logging.info(f'Transcribed input audio: {text}')
        
        # Get or create cached voice data for the reference speaker
        cache_data = get_or_create_voice_cache(reference_speaker, reference_file)
        
        # Generate with the new voice using cached prompt
        audio_data, sr = generate_speech_with_prompt(text, cache_data["prompt"])
        
        # Save output
        save_path = f'{output_dir}/output_converted.wav'
        sf.write(save_path, audio_data, sr)

        return StreamingResponse(open(save_path, 'rb'), media_type="audio/wav")
    except Exception as e:
        logging.error(f"Error in change_voice: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/upload_audio/")
async def upload_audio(
    audio_file_label: str = Form(...),
    emotion: str = Form("neutral"),
    file: UploadFile = File(...),
):
    """
    Upload an audio file for later use as the reference audio.
    Saves to resources/{label}/{emotion}.wav
    """
    try:
        contents = await file.read()

        allowed_extensions = {'wav', 'mp3', 'flac', 'ogg'}
        max_file_size = 5 * 1024 * 1024  # 5MB

        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            return {"error": "Invalid file type. Allowed types are: wav, mp3, flac, ogg"}

        if len(contents) > max_file_size:
            return {"error": "File size is over limit. Max size is 5MB."}

        temp_file = io.BytesIO(contents)
        file_format = magic.from_buffer(temp_file.read(), mime=True)

        if 'audio' not in file_format:
            return {"error": "Invalid file content."}

        # New format: save to resources/{label}/{emotion}.wav
        voice_dir = os.path.join(resources_dir, audio_file_label)
        os.makedirs(voice_dir, exist_ok=True)

        # Save original temporarily
        temp_path = os.path.join(voice_dir, f"{emotion}_orig.{file_ext}")
        with open(temp_path, "wb") as f:
            f.write(contents)

        # Convert to WAV
        wav_path = os.path.join(voice_dir, f"{emotion}.wav")
        convert_to_wav(temp_path, wav_path)

        # Remove temp if different
        if temp_path != wav_path and os.path.exists(temp_path):
            os.remove(temp_path)

        # Clear cached voice data
        invalidate_voice_cache(audio_file_label, emotion)

        return {"message": f"File {file.filename} uploaded as {audio_file_label}/{emotion}."}
    except Exception as e:
        logging.error(f"Error in upload_audio: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/synthesize_speech/")
async def synthesize_speech(
        text: str,
        voice: str,
        speed: Optional[float] = 1.0,
        emotion: str = "neutral",
):
    """
    Synthesize speech from text using a specified voice, emotion, and speed.
    If the requested emotion is not available, falls back to 'neutral'.
    """
    start_time = time.time()
    try:
        logging.info(f'Generating speech for voice: {voice}, emotion: {emotion}')

        reference_file = find_voice_reference(voice, emotion)
        if not reference_file:
            raise HTTPException(status_code=400, detail="No matching voice found.")

        # Determine actual emotion used (for cache key)
        actual_emotion = emotion
        voice_dir = os.path.join(resources_dir, voice)
        if os.path.isdir(voice_dir):
            if not os.path.isfile(os.path.join(voice_dir, f"{emotion}.wav")):
                actual_emotion = "neutral"
        else:
            actual_emotion = "neutral"

        # Get or create cached voice data
        cache_data = get_or_create_voice_cache(voice, reference_file, actual_emotion)
        
        # Generate speech using cached voice prompt
        audio_data, sr = generate_speech_with_prompt(text, cache_data["prompt"], speed)
        
        # Save output
        save_path = f'{output_dir}/output_synthesized.wav'
        sf.write(save_path, audio_data, sr)

        result = StreamingResponse(open(save_path, 'rb'), media_type="audio/wav")

        end_time = time.time()
        elapsed_time = end_time - start_time

        result.headers["X-Elapsed-Time"] = str(elapsed_time)
        result.headers["X-Device-Used"] = device

        # Add CORS headers
        result.headers["Access-Control-Allow-Origin"] = "*"
        result.headers["Access-Control-Allow-Credentials"] = "true"
        result.headers["Access-Control-Allow-Headers"] = "Origin, Content-Type, X-Amz-Date, Authorization, X-Api-Key, X-Amz-Security-Token, locale"
        result.headers["Access-Control-Allow-Methods"] = "POST, OPTIONS"

        return result
    except Exception as e:
        logging.error(f"Error in synthesize_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
