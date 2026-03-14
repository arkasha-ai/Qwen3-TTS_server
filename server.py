import os
import re
import shutil
import time
import asyncio
import torch
from concurrent.futures import ThreadPoolExecutor
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
import uuid
from cachetools import LRUCache

from qdrant_client import QdrantClient
from qdrant_client.models import Distance, VectorParams, PointStruct, Filter, FieldCondition, MatchValue

logging.basicConfig(level=logging.INFO)

# ─── Concurrency helpers ───
# threading.Lock вместо asyncio.Lock — безопасно на уровне модуля (Python 3.10+)
import threading
_voice_cache_locks: dict = {}
_voice_cache_locks_mutex = threading.Lock()
executor = ThreadPoolExecutor(max_workers=2)


def _get_voice_lock_sync(cache_key: str) -> threading.Lock:
    """Per-voice threading lock — вызывать из sync контекста (executor thread)."""
    with _voice_cache_locks_mutex:
        if cache_key not in _voice_cache_locks:
            _voice_cache_locks[cache_key] = threading.Lock()
        return _voice_cache_locks[cache_key]

# ─── Qdrant configuration ───
QDRANT_HOST = os.getenv("QDRANT_HOST", "localhost")
QDRANT_PORT = int(os.getenv("QDRANT_PORT", 6333))
QDRANT_COLLECTION = os.getenv("QDRANT_COLLECTION", "voice_embeddings")

qdrant_client: Optional[QdrantClient] = None
_qdrant_collection_initialized = False

app = FastAPI()

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,  # wildcard origin + credentials=True is forbidden by spec
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

# Suppress noisy configuration logging
logging.getLogger("qwen_tts.core.models.configuration_qwen3_tts").setLevel(logging.WARNING)

# torch.compile disabled — net negative for autoregressive generation.
# Dynamic shapes per token cause constant recompilation; overhead > gains.
# Measured: baseline RTF 1.32x → with compile 1.64x (worse). Keep disabled.
# Re-enable only when switching to streaming decode with fixed windows (CUDA graphs).
_COMPILE_APPLIED = False

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
voice_cache = LRUCache(maxsize=50)


# ─── Qdrant helpers ───

def _connect_qdrant():
    """Lazily connect to Qdrant. Tolerates Qdrant being unavailable."""
    global qdrant_client
    if qdrant_client is not None:
        return qdrant_client
    try:
        qdrant_client = QdrantClient(host=QDRANT_HOST, port=QDRANT_PORT, timeout=5)
        logging.info(f"Connected to Qdrant at {QDRANT_HOST}:{QDRANT_PORT}")
        return qdrant_client
    except Exception as e:
        logging.warning(f"Could not connect to Qdrant: {e}. Voice embeddings will NOT be persisted.")
        return None


def init_qdrant_collection(vector_size: int):
    """Create the Qdrant collection if it doesn't exist."""
    global _qdrant_collection_initialized
    if _qdrant_collection_initialized:
        return
    client = _connect_qdrant()
    if client is None:
        return
    try:
        collections = [c.name for c in client.get_collections().collections]
        if QDRANT_COLLECTION not in collections:
            client.create_collection(
                collection_name=QDRANT_COLLECTION,
                vectors_config=VectorParams(size=vector_size, distance=Distance.COSINE),
            )
            logging.info(f"Created Qdrant collection '{QDRANT_COLLECTION}' with vector_size={vector_size}")
        else:
            logging.info(f"Qdrant collection '{QDRANT_COLLECTION}' already exists")
        _qdrant_collection_initialized = True
    except Exception as e:
        logging.warning(f"Qdrant init_collection failed: {e}")


def _extract_embedding(voice_prompt) -> Optional[np.ndarray]:
    """
    Extract the speaker embedding (x-vector) from a voice_clone_prompt object.
    Handles list of VoiceClonePromptItem (Qwen3-TTS returns a list).
    """
    # Known candidate field names (Qwen3-TTS and common TTS frameworks)
    candidates = [
        "ref_spk_embedding",  # Qwen3-TTS VoiceClonePromptItem — check first!
        "x_vector", "speaker_embedding", "spk_embedding", "xvector",
        "embedding", "spk_emb", "speaker_emb", "prompt_embedding",
    ]

    # Qwen3-TTS returns a list of VoiceClonePromptItem — unwrap first item
    if isinstance(voice_prompt, (list, tuple)) and len(voice_prompt) > 0:
        voice_prompt = voice_prompt[0]
        logging.info(f"Unwrapped list prompt → {type(voice_prompt)}")

    obj_attrs = vars(voice_prompt) if hasattr(voice_prompt, '__dict__') else {}

    logging.info(f"voice_clone_prompt type: {type(voice_prompt)}, attrs: {list(obj_attrs.keys())}")

    # --- dict-like prompt ---
    if isinstance(voice_prompt, dict):
        for name in candidates:
            if name in voice_prompt:
                val = voice_prompt[name]
                return _to_numpy(val)
        # Fallback: first tensor/ndarray value
        for k, v in voice_prompt.items():
            arr = _to_numpy(v)
            if arr is not None and arr.ndim == 1:
                logging.info(f"Using dict key '{k}' as embedding (shape {arr.shape})")
                return arr

    # --- object with attributes ---
    for name in candidates:
        val = getattr(voice_prompt, name, None)
        if val is not None:
            arr = _to_numpy(val)
            if arr is not None:
                logging.info(f"Using attribute '{name}' as embedding (shape {arr.shape})")
                return arr

    # Fallback: scan all attrs for a 1-D tensor/ndarray
    for name, val in obj_attrs.items():
        arr = _to_numpy(val)
        if arr is not None and arr.ndim == 1 and arr.shape[0] > 32:
            logging.info(f"Fallback: using attribute '{name}' as embedding (shape {arr.shape})")
            return arr

    logging.warning("Could not extract embedding from voice_clone_prompt")
    return None


def _to_numpy(val) -> Optional[np.ndarray]:
    """Convert a tensor or ndarray to a flat numpy float32 array."""
    if val is None:
        return None
    if isinstance(val, np.ndarray):
        return val.flatten().astype(np.float32)
    if isinstance(val, torch.Tensor):
        return val.detach().cpu().float().numpy().flatten()
    return None


def save_voice_to_qdrant(voice: str, emotion: str, x_vector: np.ndarray, ref_text: str):
    """Save or update a voice embedding in Qdrant."""
    client = _connect_qdrant()
    if client is None:
        return
    init_qdrant_collection(len(x_vector))
    try:
        # Delete existing point with same voice+emotion (upsert by payload match)
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(
                must=[
                    FieldCondition(key="voice", match=MatchValue(value=voice)),
                    FieldCondition(key="emotion", match=MatchValue(value=emotion)),
                ]
            ),
        )
        point_id = str(uuid.uuid4())
        client.upsert(
            collection_name=QDRANT_COLLECTION,
            points=[
                PointStruct(
                    id=point_id,
                    vector=x_vector.tolist(),
                    payload={"voice": voice, "emotion": emotion, "ref_text": ref_text},
                )
            ],
        )
        logging.info(f"Saved voice embedding to Qdrant: {voice}/{emotion} (dim={len(x_vector)})")
    except Exception as e:
        logging.warning(f"Failed to save embedding to Qdrant: {e}")


def load_voice_from_qdrant(voice: str, emotion: str) -> Optional[dict]:
    """
    Load a voice embedding from Qdrant.
    Returns {"x_vector": np.ndarray, "ref_text": str} or None.
    """
    client = _connect_qdrant()
    if client is None:
        return None
    try:
        results = client.scroll(
            collection_name=QDRANT_COLLECTION,
            scroll_filter=Filter(
                must=[
                    FieldCondition(key="voice", match=MatchValue(value=voice)),
                    FieldCondition(key="emotion", match=MatchValue(value=emotion)),
                ]
            ),
            limit=1,
            with_vectors=True,
        )
        points = results[0]
        if points:
            point = points[0]
            return {
                "x_vector": np.array(point.vector, dtype=np.float32),
                "ref_text": point.payload.get("ref_text", ""),
            }
    except Exception as e:
        logging.warning(f"Failed to load embedding from Qdrant: {e}")
    return None


def delete_voice_from_qdrant(voice: str, emotion: Optional[str] = None):
    """Delete voice embedding(s) from Qdrant."""
    client = _connect_qdrant()
    if client is None:
        return
    try:
        conditions = [FieldCondition(key="voice", match=MatchValue(value=voice))]
        if emotion:
            conditions.append(FieldCondition(key="emotion", match=MatchValue(value=emotion)))
        client.delete(
            collection_name=QDRANT_COLLECTION,
            points_selector=Filter(must=conditions),
        )
        logging.info(f"Deleted voice embedding from Qdrant: {voice}/{emotion or '*'}")
    except Exception as e:
        logging.warning(f"Failed to delete from Qdrant: {e}")


def _rebuild_prompt_from_embedding(x_vector: np.ndarray, ref_text: str):
    """
    Attempt to rebuild a voice_clone_prompt from a stored x_vector.
    This is model-specific — if the model exposes a way to build prompts
    from raw embeddings, we use it. Otherwise we wrap it in the expected structure.
    """
    # Try model-level reconstruction if available
    if hasattr(model, 'create_prompt_from_embedding'):
        tensor = torch.from_numpy(x_vector).to(device)
        return model.create_prompt_from_embedding(tensor, ref_text)

    # Generic fallback: wrap in a dict matching common prompt structures
    tensor = torch.from_numpy(x_vector).to(dtype=torch.bfloat16, device=device)

    # Try to match the structure of what create_voice_clone_prompt returns
    # by creating a dummy prompt and replacing its embedding
    logging.info("Rebuilding prompt by substituting stored embedding into fresh prompt structure")
    return {"x_vector": tensor, "ref_text": ref_text, "_restored_from_qdrant": True}


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
    
    Lookup order:
    1. In-memory cache (fastest)
    2. Qdrant persistent storage (restore embedding → rebuild prompt)
    3. Create from reference audio file (slowest — runs Whisper + prompt creation)
    """
    global voice_cache
    
    cache_key = get_cache_key(voice, emotion)
    
    # 1. In-memory cache
    if cache_key in voice_cache:
        logging.info(f"Using in-memory cached voice data for: {cache_key}")
        return voice_cache[cache_key]
    
    # 2. Try Qdrant
    qdrant_data = load_voice_from_qdrant(voice, emotion)
    if qdrant_data is not None:
        logging.info(f"Restoring voice from Qdrant for: {cache_key}")
        x_vector = qdrant_data["x_vector"]
        ref_text = qdrant_data["ref_text"]

        # We still need the reference audio to rebuild a proper prompt
        # because most TTS models need the full prompt object, not just x_vector.
        # Try to rebuild from audio + use the stored ref_text (skip Whisper).
        try:
            ref_audio_data, ref_sr = sf.read(reference_file)
            voice_prompt = model.create_voice_clone_prompt(
                ref_audio=(ref_audio_data, ref_sr),
                ref_text=ref_text,  # Use stored text — skips Whisper!
            )
            voice_cache[cache_key] = {
                "processed_audio": reference_file,
                "ref_text": ref_text,
                "prompt": voice_prompt,
                "audio_data": ref_audio_data,
                "sample_rate": ref_sr,
            }
            logging.info(f"Voice restored from Qdrant (skipped Whisper): {cache_key}")
            return voice_cache[cache_key]
        except Exception as e:
            logging.warning(f"Failed to restore prompt from Qdrant data: {e}, falling back to full creation")
    
    # 3. Full creation from reference file
    logging.info(f"Creating voice cache for: {cache_key}")
    
    # Process reference audio (clip to 15s, remove silence)
    processed_ref, ref_text = process_reference_audio(reference_file)
    
    # Create reusable voice clone prompt
    ref_audio_data, ref_sr = sf.read(processed_ref)
    voice_prompt = model.create_voice_clone_prompt(
        ref_audio=(ref_audio_data, ref_sr),
        ref_text=ref_text,
    )
    
    # Extract and persist embedding to Qdrant
    x_vector = _extract_embedding(voice_prompt)
    if x_vector is not None:
        save_voice_to_qdrant(voice, emotion, x_vector, ref_text)
    
    # Store in memory cache
    voice_cache[cache_key] = {
        "processed_audio": processed_ref,
        "ref_text": ref_text,
        "prompt": voice_prompt,
        "audio_data": ref_audio_data,
        "sample_rate": ref_sr,
    }
    
    logging.info(f"Voice cache created for: {cache_key} (transcription: '{ref_text[:50]}...')")
    return voice_cache[cache_key]


def validate_path_component(name: str, field: str = "name") -> str:
    """Reject path traversal / injection in voice/emotion params."""
    if not re.match(r'^[a-zA-Z0-9_\-]{1,64}$', name):
        raise HTTPException(
            status_code=400,
            detail=f"Invalid {field}: alphanumeric, _ or - only (max 64 chars)",
        )
    return name


def _sync_generate(voice: str, emotion: str, text: str, speed: float) -> tuple:
    """Synchronous wrapper for ML inference — runs in ThreadPoolExecutor."""
    reference_file = find_voice_reference(voice, emotion)
    if not reference_file:
        raise ValueError(f"No matching voice found: {voice}/{emotion}")
    cache_key = get_cache_key(voice, emotion)
    lock = _get_voice_lock_sync(cache_key)
    with lock:
        cache_data = get_or_create_voice_cache(voice, reference_file, emotion)
    return generate_speech_with_prompt(text, cache_data["prompt"], speed)


def _sync_generate_voice_change(reference_speaker: str, text: str) -> tuple:
    """Synchronous wrapper for voice-change inference — runs in ThreadPoolExecutor."""
    reference_file = find_voice_reference(reference_speaker)
    if not reference_file:
        raise ValueError("No matching reference speaker found.")
    cache_key = get_cache_key(reference_speaker, "neutral")
    lock = _get_voice_lock_sync(cache_key)
    with lock:
        cache_data = get_or_create_voice_cache(reference_speaker, reference_file)
    return generate_speech_with_prompt(text, cache_data["prompt"])


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
    
    # Also remove from Qdrant
    delete_voice_from_qdrant(voice, emotion)


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


# ─── GPU keepalive ───────────────────────────────────────────────────────────
# Prevents GPU clock downscaling during idle periods by doing a tiny dummy
# matmul + synchronize every 30 s. Ported from dingausmwald/Qwen3-TTS-Openai-Fastapi.
# Persistent tensors are created once and reused to avoid allocator churn.
_gpu_keepalive_tensors: tuple = ()

async def _gpu_keepalive_task():
    """Background task: keeps GPU clocks up with a small periodic matmul."""
    global _gpu_keepalive_tensors
    if not torch.cuda.is_available():
        return
    a = torch.randn(64, 64, device=device, dtype=torch.float16)
    b = torch.randn(64, 64, device=device, dtype=torch.float16)
    _gpu_keepalive_tensors = (a, b)
    logging.info("GPU keepalive task started (interval: 30 s)")
    while True:
        await asyncio.sleep(30)
        try:
            with torch.no_grad():
                _ = torch.matmul(a, b)
            torch.cuda.synchronize()
        except Exception:
            pass  # non-critical — never crash the server over this


@app.on_event("startup")
async def startup_event():
    """Warmup the model on startup and launch background tasks."""
    global default_ref_audio, default_voice_prompt

    # GPU keepalive — предотвращает даунклокинг GPU после idle
    if torch.cuda.is_available():
        asyncio.ensure_future(_gpu_keepalive_task())

    # Check if we have a default voice file
    ref = find_voice_reference("default_en")
    if ref:
        default_ref_audio = ref

    # torch.compile is disabled — no warmup needed.
    logging.info("Server ready (no warmup — torch.compile is disabled)")


# ─── Voice management endpoints ───

@app.get("/voices/")
async def get_voices():
    """List all available voices and their emotions."""
    return {"voices": list_voices()}


@app.post("/voices/{name}/emotions/{emotion}")
async def upload_voice_emotion(name: str, emotion: str, file: UploadFile = File(...)):
    """Upload a reference audio for a specific voice + emotion."""
    name = validate_path_component(name, "voice name")
    emotion = validate_path_component(emotion, "emotion")
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
    name = validate_path_component(name, "voice name")
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
    name = validate_path_component(name, "voice name")
    emotion = validate_path_component(emotion, "emotion")
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
    reference_speaker = validate_path_component(reference_speaker, "reference_speaker")
    try:
        logging.info(f'Changing voice to {reference_speaker}...')

        contents = await file.read()

        # Transcribe input audio (blocking — run in executor)
        loop = asyncio.get_running_loop()
        input_buf = io.BytesIO(contents)
        input_path = f'{output_dir}/input_audio_{id(input_buf)}.wav'
        with open(input_path, 'wb') as f:
            f.write(contents)
        try:
            text = await loop.run_in_executor(executor, lambda: transcribe_audio(input_path))
        finally:
            try:
                os.remove(input_path)
            except OSError:
                pass
        logging.info(f'Transcribed input audio: {text}')

        # Generate with the new voice (lock is handled inside _sync_generate_voice_change)
        audio_data, sr = await loop.run_in_executor(
            executor,
            lambda: _sync_generate_voice_change(reference_speaker, text),
        )

        buf = audio_to_wav_bytes(audio_data, sr)
        return StreamingResponse(buf, media_type="audio/wav")
    except HTTPException:
        raise
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
    audio_file_label = validate_path_component(audio_file_label, "audio_file_label")
    emotion = validate_path_component(emotion, "emotion")
    try:
        contents = await file.read()

        allowed_extensions = {'wav', 'mp3', 'flac', 'ogg'}
        max_file_size = 5 * 1024 * 1024  # 5MB

        file_ext = file.filename.split('.')[-1].lower()
        if file_ext not in allowed_extensions:
            raise HTTPException(status_code=400, detail="Invalid file type. Allowed: wav, mp3, flac, ogg")

        if len(contents) > max_file_size:
            raise HTTPException(status_code=400, detail="File size is over limit. Max size is 5MB.")

        temp_file = io.BytesIO(contents)
        file_format = magic.from_buffer(temp_file.read(), mime=True)

        if 'audio' not in file_format:
            raise HTTPException(status_code=400, detail="Invalid file content.")

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
    voice = validate_path_component(voice, "voice")
    emotion = validate_path_component(emotion, "emotion")
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

        # Lock is handled inside _sync_generate (threading.Lock, safe in executor)
        loop = asyncio.get_running_loop()
        audio_data, sr = await loop.run_in_executor(
            executor,
            lambda: _sync_generate(voice, actual_emotion, text, speed),
        )

        buf = audio_to_wav_bytes(audio_data, sr)
        result = StreamingResponse(buf, media_type="audio/wav")

        elapsed_time = time.time() - start_time
        result.headers["X-Elapsed-Time"] = str(elapsed_time)
        result.headers["X-Device-Used"] = device

        return result
    except HTTPException:
        raise
    except Exception as e:
        logging.error(f"Error in synthesize_speech: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
