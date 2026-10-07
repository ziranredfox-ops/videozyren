import os
import sys
import re
import json
import time
import math
import uuid
import queue
import tempfile
import threading
import subprocess
from pathlib import Path
from typing import Optional, List, Dict
from concurrent.futures import ThreadPoolExecutor, as_completed
import base64
import struct
import zlib

import requests
import yt_dlp
from fastapi import FastAPI, Query, HTTPException, BackgroundTasks
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
import uvicorn

# ==========================================
# API CREDENTIALS
# ==========================================
DEFAULT_ASSEMBLY_KEY = os.environ.get("ASSEMBLYAI_API_KEY", "")

# xKiro API
XKIRO_BASE_URL = "https://api.xkiro.com/v1/chat/completions"
XKIRO_KEY = os.environ.get("XKIRO_API_KEY", "")
XKIRO_MODEL = "qwen/qwen3.8-max:free"

# LiteRouter API
LITEROUTER_BASE_URL = "https://api.literouter.com/v1/chat/completions"
LITEROUTER_KEY = os.environ.get("LITEROUTER_API_KEY", "")
LITEROUTER_MODEL = "deepseek-v3.2:free"

# Aichixia API (Qwen 3.8 27B - Anti-Sensor & Sutradara Dewasa)
AICHIXIA_BASE_URL = "https://www.aichixia.xyz/api/v1/chat/completions"
AICHIXIA_KEY = os.environ.get("AICHIXIA_API_KEY", "")
AICHIXIA_MODEL = "alibaba/qwen3.8-27b"

# Cloudflare Workers AI Whisper (Word-Level Timing & Bisikan)
CF_TOKEN = os.environ.get("CLOUDFLARE_API_TOKEN", "")
CF_ACCT = os.environ.get("CLOUDFLARE_ACCOUNT_ID", "")

HANZI_RE = re.compile(r'[\u4e00-\u9fff]')

def clean_remaining_hanzi(text: str) -> str:
    desahan_map = {
        '嗯': 'hmm...', '啊': 'ahh...', '哦': 'ohh...', '呀': 'ya...',
        '哈': 'ha...', '哼': 'hng...', '妈': 'ibu', '爸': 'ayah',
        '姐': 'kakak', '滚': 'pergi', '好': 'baik', '来': 'ayo',
        '快': 'cepat', '慢': 'pelan', '慢点': 'pelan-pelan', '点': 'dikit',
        '大': 'besar', '爽': 'enak banget', '痛': 'sakit', '别': 'jangan',
        '要': 'mau', '死': 'mati'
    }
    for hz, ind in desahan_map.items():
        text = text.replace(hz, ind)
    text = HANZI_RE.sub('', text).strip()
    return text if text else "ahh..."

BACKUP_QUEUE_PATH = r"D:\translate\remote_web\queue_backup.json"

STATIC_DIR = r"D:\translate\remote_web\static"
os.makedirs(STATIC_DIR, exist_ok=True)

app = FastAPI(title="Remote Subtitle & Video Downloader")
app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

# Global Queues & Databases
JOB_QUEUE = queue.Queue()
TASKS_DB: Dict[str, dict] = {}
DOWNLOADS_DB: Dict[str, dict] = {}
WORKER_THREAD = None

class GlobalSettings(BaseModel):
    speech_model: str = "fireredasr2"
    language_code: str = "auto"
    word_boost: str = ""
    boost_param: str = "default"
    punctuate: bool = True
    format_text: bool = True
    disfluencies: bool = False
    pause_sec: float = 0.6
    max_char_len: int = 42
    translate_to_id: bool = True
    translation_mode: str = "aichixia"  # "aichixia", "both", "xkiro", "literouter"
    translation_style: str = "natural_no_sensor"
    kaggle_url: Optional[str] = "https://amenities-volvo-phones-looksmart.trycloudflare.com"

GLOBAL_CONFIG = GlobalSettings()

class EnqueueItemRequest(BaseModel):
    video_paths: List[str]
    settings: Optional[GlobalSettings] = None

class DownloadLinkRequest(BaseModel):
    url: str
    save_dir: Optional[str] = None
    custom_name: Optional[str] = None
    auto_queue: bool = True
    use_kaggle_480p: bool = False
    # "kaggle" = unduh & kompres 480p di server Google, "local" = unduh 480p ke laptop
    dl_mode: Optional[str] = None

class RegisterKaggleRequest(BaseModel):
    kaggle_url: str

def save_queue_state():
    try:
        data = {
            "counts": {
                "processing": sum(1 for t in TASKS_DB.values() if t.get("state") == "PROCESSING"),
                "queued": sum(1 for t in TASKS_DB.values() if t.get("state") == "QUEUED"),
                "completed": sum(1 for t in TASKS_DB.values() if t.get("state") == "COMPLETED"),
                "error": sum(1 for t in TASKS_DB.values() if t.get("state") == "ERROR")
            },
            "tasks": list(TASKS_DB.values())
        }
        with open(BACKUP_QUEUE_PATH, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2, default=str)
    except Exception:
        pass

def load_queue_state_on_startup():
    if not os.path.exists(BACKUP_QUEUE_PATH):
        return
    try:
        with open(BACKUP_QUEUE_PATH, "r", encoding="utf-8") as f:
            data = json.load(f)
        tasks = data.get("tasks", [])
        valid_count = 0
        for t in reversed(tasks):
            t_id = t.get("task_id")
            v_path = t.get("video_path")
            # Abaikan task yang file videonya sudah tidak ada di disk
            if not t_id or not v_path or not os.path.exists(v_path):
                continue
            
            st = t.get("state")
            cfg = GlobalSettings()
            t["config"] = cfg
            TASKS_DB[t_id] = t
            valid_count += 1
            
            if st in ["PROCESSING", "QUEUED"]:
                t["state"] = "QUEUED"
                t["status"] = "Melanjutkan antrian..."
                t["progress"] = 0
                JOB_QUEUE.put(t_id)
        print(f"[STARTUP] Berhasil memulihkan {valid_count} antrian valid dari disk.")
    except Exception as e:
        print(f"[STARTUP] Gagal memulihkan antrian: {e}")

def format_timestamp(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    hours = millis // 3600000
    millis %= 3600000
    minutes = millis // 60000
    millis %= 60000
    secs = millis // 1000
    millis %= 1000
    return f"{hours:02d}:{minutes:02d}:{secs:02d},{millis:03d}"

def convert_words_to_srt(words: list, pause_sec=0.6, max_char_len=42) -> list:
    cues = []
    if not words:
        return cues
    current_cue = {"words": [], "start": words[0]["start"] / 1000.0, "end": words[0]["end"] / 1000.0}
    for i, w in enumerate(words):
        w_start = w["start"] / 1000.0
        w_end = w["end"] / 1000.0
        w_text = w.get("text", "")
        if not current_cue["words"]:
            current_cue["words"].append(w_text)
            current_cue["start"] = w_start
            current_cue["end"] = w_end
            continue
        prev_end = current_cue["end"]
        pause = w_start - prev_end
        combined = " ".join(current_cue["words"] + [w_text])
        if pause >= pause_sec or len(combined) >= max_char_len:
            cues.append({
                "start": current_cue["start"],
                "end": current_cue["end"],
                "text": " ".join(current_cue["words"])
            })
            current_cue = {"words": [w_text], "start": w_start, "end": w_end}
        else:
            current_cue["words"].append(w_text)
            current_cue["end"] = w_end
    if current_cue["words"]:
        cues.append({
            "start": current_cue["start"],
            "end": current_cue["end"],
            "text": " ".join(current_cue["words"])
        })
    return cues

def extract_audio(video_path: str) -> str:
    temp_audio = tempfile.NamedTemporaryFile(suffix=".mp3", delete=False)
    temp_audio.close()
    cmd = [
        "ffmpeg", "-y",
        "-i", video_path,
        "-vn", "-ac", "1", "-ar", "16000", "-b:a", "64k",
        temp_audio.name
    ]
    res = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if res.returncode != 0:
        err = res.stderr.decode("utf-8", errors="ignore")
        raise RuntimeError(f"FFmpeg gagal mengekstrak audio: {err[-300:]}")
    return temp_audio.name

def transcribe_whisper(audio_path: str, cfg: GlobalSettings, task_id: str) -> dict:
    TASKS_DB[task_id]["progress"] = 25
    TASKS_DB[task_id]["status"] = "Mengunggah audio ke Whisper Engine (Word-Level Timing)..."
    save_queue_state()
    
    cf_url = f"https://api.cloudflare.com/client/v4/accounts/{CF_ACCT}/ai/run/@cf/openai/whisper"
    headers = {
        "Authorization": f"Bearer {CF_TOKEN}",
        "Content-Type": "application/octet-stream"
    }
    
    with open(audio_path, "rb") as f:
        audio_data = f.read()

    TASKS_DB[task_id]["progress"] = 35
    TASKS_DB[task_id]["status"] = "Whisper Engine memproses audio & timing..."
    save_queue_state()

    res = requests.post(cf_url, headers=headers, data=audio_data, timeout=180)
    if res.status_code != 200:
        raise RuntimeError(f"Gagal proses Whisper: {res.text}")

    cf_json = res.json()
    result = cf_json.get("result", {})
    segments = result.get("segments", [])

    words = []
    for seg in segments:
        seg_words = seg.get("words", [])
        if seg_words:
            for w in seg_words:
                words.append({
                    "text": w.get("word", ""),
                    "start": int(round(float(w.get("start", 0)) * 1000)),
                    "end": int(round(float(w.get("end", 0)) * 1000))
                })
        else:
            words.append({
                "text": seg.get("text", ""),
                "start": int(round(float(seg.get("start", 0)) * 1000)),
                "end": int(round(float(seg.get("end", 0)) * 1000))
            })

    return {
        "text": result.get("text", ""),
        "words": words
    }

def transcribe_fireredasr(audio_path: str, cfg: GlobalSettings, task_id: str) -> dict:
    TASKS_DB[task_id]["progress"] = 25
    TASKS_DB[task_id]["status"] = "Menghubungkan ke FireRedASR2S di Cloud Kaggle GPU..."
    save_queue_state()

    kaggle_url = get_active_kaggle_url(TASKS_DB[task_id])
    if not kaggle_url:
        raise RuntimeError("Worker Kaggle GPU tidak terdeteksi untuk menjalankan FireRedASR2S!")

    TASKS_DB[task_id]["progress"] = 30
    TASKS_DB[task_id]["status"] = "Mengunggah audio ke Kaggle GPU Tesla T4..."
    save_queue_state()

    with open(audio_path, "rb") as f_aud:
        files = {"file": (os.path.basename(audio_path), f_aud, "audio/mpeg")}
        res = requests.post(f"{kaggle_url}/api/transcribe", files=files, timeout=90)

    if res.status_code != 200:
        raise RuntimeError(f"Gagal submit ke FireRedASR2S di Kaggle: {res.text[:200]}")

    job_info = res.json()
    job_id = job_info.get("job_id")
    if not job_id:
        raise RuntimeError(f"Respon Kaggle tidak valid: {res.text[:200]}")

    start_poll = time.time()
    while True:
        time.sleep(3)
        try:
            chk_res = requests.get(f"{kaggle_url}/api/transcribe_job/{job_id}", timeout=10)
        except Exception:
            continue

        if chk_res.status_code != 200:
            continue

        chk = chk_res.json()
        st = chk.get("state")
        prog = chk.get("progress", 35)
        status_txt = chk.get("status", "Memproses audio...")

        TASKS_DB[task_id]["progress"] = int(30 + (prog / 100.0) * 30)
        TASKS_DB[task_id]["status"] = f"[Kaggle T4] {status_txt}"
        save_queue_state()

        if st == "COMPLETED":
            data = chk.get("result", {})
            cues = data.get("cues", [])
            words = []
            for c in cues:
                start_ms = int(float(c.get("start", 0)) * 1000)
                end_ms = int(float(c.get("end", 0)) * 1000)
                text = c.get("text", "")
                words.append({
                    "text": text,
                    "start": start_ms,
                    "end": end_ms,
                    "confidence": 0.99
                })

            TASKS_DB[task_id]["progress"] = 60
            TASKS_DB[task_id]["status"] = f"Transkripsi FireRedASR2S Selesai! ({len(cues)} kalimat terdeteksi)"
            save_queue_state()

            return {
                "text": data.get("text", ""),
                "words": words,
                "cues": cues
            }
        elif st == "ERROR":
            raise RuntimeError(f"FireRedASR2S Error di Kaggle: {chk.get('error', 'Unknown error')}")

        if time.time() - start_poll > 3600:
            raise RuntimeError("Transkripsi di Kaggle melebihi batas waktu maksimal 1 jam")

def transcribe(audio_path: str, cfg: GlobalSettings, task_id: str) -> dict:
    if cfg.speech_model == "fireredasr2":
        return transcribe_fireredasr(audio_path, cfg, task_id)

    if cfg.speech_model == "whisperx":
        return transcribe_whisper(audio_path, cfg, task_id)

    TASKS_DB[task_id]["progress"] = 20
    TASKS_DB[task_id]["status"] = "Mengunggah audio ke AssemblyAI..."
    save_queue_state()
    
    headers = {"authorization": DEFAULT_ASSEMBLY_KEY}
    with open(audio_path, "rb") as f:
        res = requests.post("https://api.assemblyai.com/v2/upload", headers=headers, data=f)
    if res.status_code != 200:
        raise RuntimeError(f"Gagal upload ke AssemblyAI: {res.text}")
    upload_url = res.json()["upload_url"]

    TASKS_DB[task_id]["progress"] = 35
    TASKS_DB[task_id]["status"] = f"Transkripsi AssemblyAI ({cfg.speech_model})..."
    save_queue_state()

    post_data = {
        "audio_url": upload_url,
        "punctuate": bool(cfg.punctuate),
        "format_text": bool(cfg.format_text),
        "filter_profanity": False,
        "speaker_labels": False,
        "disfluencies": bool(cfg.disfluencies)
    }
    
    valid_models = ["universal-3-5-pro", "universal-3-pro", "universal-2"]
    if cfg.speech_model in valid_models:
        post_data["speech_models"] = [cfg.speech_model]
    else:
        post_data["speech_models"] = ["universal-3-5-pro"]

    if cfg.language_code and cfg.language_code != "auto":
        post_data["language_code"] = cfg.language_code
    else:
        post_data["language_detection"] = True

    if cfg.word_boost.strip():
        keywords = [k.strip() for k in cfg.word_boost.replace("\n", ",").split(",") if k.strip()]
        if keywords:
            post_data["word_boost"] = keywords
            post_data["boost_param"] = cfg.boost_param

    tx_res = requests.post("https://api.assemblyai.com/v2/transcript", json=post_data, headers={"authorization": DEFAULT_ASSEMBLY_KEY, "content-type": "application/json"})
    if tx_res.status_code != 200:
        raise RuntimeError(f"Gagal submit AssemblyAI: {tx_res.text}")
    tx_id = tx_res.json()["id"]

    while True:
        time.sleep(3)
        chk = requests.get(f"https://api.assemblyai.com/v2/transcript/{tx_id}", headers=headers).json()
        st = chk.get("status")
        if st == "completed":
            return chk
        elif st == "error":
            raise RuntimeError(f"AssemblyAI Error: {chk.get('error')}")
        TASKS_DB[task_id]["status"] = f"AssemblyAI memproses audio ({st})..."

def translate_cues_with_provider(
    cues: list,
    provider: str,
    cfg: GlobalSettings,
    task_id: str,
    progress_start: int,
    progress_end: int
) -> list:
    if not cues:
        return []

    system_prompt = (
        "Kamu adalah penerjemah naskah subtitle film dewasa profesional. Dilarang menerjemahkan secara harfiah kata-per-kata.\n\n"
        "1. KONSISTENSI KATA GANTI & NADA BICARA INTIM:\n"
        "- WAJIB gunakan kata ganti 'aku - kamu' atau sebutan peran ('guru - murid', 'tuan/majikan - anjing/budak').\n"
        "- DILARANG KERAS menggunakan kata gaul seperti 'lo', 'lu', 'gue', 'gua'! Seluruh percakapan intim harus menggunakan 'aku - kamu' agar mengalir sensual, alami, dan tidak aneh.\n\n"
        "2. DEKONSTRUKSI METAFORA & EUFEMISME RANJANG:\n"
        "- Di dalam naskah film dewasa ini sama sekali TIDAK ADA percakapan tentang makanan, masakan, kuliner, unggas, hewan, atau perkakas.\n"
        "- Semua penyebutan makanan, unggas, telur, daging, buah, atau benda fisik saat suasana intim adalah 100% KIASAN ALAT KELAMIN dan AKTIVITAS SEKS.\n"
        "- Bongkar kiasan tersebut menjadi sebutan organ intim pria/wanita atau aksi persetubuhan yang vulgar, gamblang, dan nyata. DILARANG membiarkan kata makanan atau hewan lolos ke hasil terjemahan!\n\n"
        "3. LOGIKA FISIK & DIRTY TALK LISAN:\n"
        "- Tafsirkan ucapan berdasarkan apa yang sedang terjadi secara fisik pada organ tubuh karakter (oral, penetrasi, rangsangan, desahan, klimaks).\n"
        "- Gunakan bahasa obrolan ranjang (dirty talk) bahasa Indonesia yang sangat vulgar, sensual, dan mengalir alami tanpa canggung.\n\n"
        "Format output HARUS array JSON valid: [{\"id\": 0, \"text\": \"...\"}, {\"id\": 1, \"text\": \"...\"}] tanpa markdown code block."
    )

    if provider == "aichixia":
        api_url = AICHIXIA_BASE_URL
        api_key = AICHIXIA_KEY
        model_name = AICHIXIA_MODEL
        provider_name = "Aichixia (Qwen 3.8 27B)"
        delay_sec = 2.5
    elif provider == "xkiro":
        api_url = XKIRO_BASE_URL
        api_key = XKIRO_KEY
        model_name = XKIRO_MODEL
        provider_name = "xKiro (Qwen 3.8 Max)"
        delay_sec = 2.5
    else:
        api_url = LITEROUTER_BASE_URL
        api_key = LITEROUTER_KEY
        model_name = LITEROUTER_MODEL
        provider_name = "LiteRouter (DeepSeek V3.2)"
        delay_sec = 8.0

    batch_size = 50
    total = len(cues)
    batches = [cues[i:i+batch_size] for i in range(0, total, batch_size)]
    translated_cues = []

    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}

    for b_idx, batch in enumerate(batches):
        pct = progress_start + int((b_idx / len(batches)) * (progress_end - progress_start))
        TASKS_DB[task_id]["progress"] = pct
        TASKS_DB[task_id]["status"] = f"[{provider_name}] Terjemahkan batch {b_idx+1}/{len(batches)} ({pct}%)..."
        save_queue_state()

        lines_to_send = [{"id": i, "text": c["text"]} for i, c in enumerate(batch)]
        user_prompt = "Terjemahkan array subtitle berikut ke Bahasa Indonesia:\n" + json.dumps(lines_to_send, ensure_ascii=False)

        payload = {
            "model": model_name,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ],
            "temperature": 0.3
        }

        success = False
        last_error = ""

        for attempt in range(5):
            try:
                r = requests.post(api_url, json=payload, headers=headers, timeout=60)
                
                if r.status_code in [403, 429]:
                    wait_sec = 8
                    m = re.search(r'(\d+)\s*seconds', r.text)
                    if m:
                        wait_sec = int(m.group(1)) + 1
                    TASKS_DB[task_id]["status"] = f"[{provider_name}] Cooldown {wait_sec}s..."
                    time.sleep(wait_sec)
                    continue

                if r.status_code != 200:
                    last_error = f"HTTP {r.status_code}: {r.text[:150]}"
                    # Jika tertolak filter konten moral/safety (misal kata vulgar di video dewasa)
                    if "content safety filter" in r.text.lower() or "safety" in r.text.lower():
                        # Coba fallback ke provider lain (LiteRouter / DeepSeek) yang tidak ada sensor
                        if provider == "xkiro":
                            try:
                                alt_headers = {"Authorization": f"Bearer {LITEROUTER_KEY}", "Content-Type": "application/json"}
                                alt_payload = dict(payload)
                                alt_payload["model"] = LITEROUTER_MODEL
                                r_alt = requests.post(LITEROUTER_BASE_URL, json=alt_payload, headers=alt_headers, timeout=60)
                                if r_alt.status_code == 200:
                                    r = r_alt
                                    # Lanjut proses response r di bawah
                                else:
                                    last_error = f"xKiro filter safety & LiteRouter fallback HTTP {r_alt.status_code}"
                                    time.sleep(delay_sec)
                                    continue
                            except Exception as alt_err:
                                last_error = f"Safety fallback error: {alt_err}"
                                time.sleep(delay_sec)
                                continue
                        else:
                            time.sleep(delay_sec)
                            continue
                    else:
                        time.sleep(delay_sec)
                        continue

                raw = r.json()["choices"][0]["message"]["content"].strip()
                if raw.startswith("```json"): raw = raw[7:]
                if raw.startswith("```"): raw = raw[3:]
                if raw.endswith("```"): raw = raw[:-3]

                parsed = json.loads(raw.strip())
                lookup = {}
                if isinstance(parsed, list):
                    for item in parsed:
                        if isinstance(item, dict) and "id" in item and "text" in item:
                            try:
                                lookup[int(item["id"])] = str(item["text"])
                            except Exception:
                                pass
                elif isinstance(parsed, dict):
                    for k, v in parsed.items():
                        try:
                            lookup[int(k)] = v if isinstance(v, str) else str(v.get("text", v))
                        except Exception:
                            pass

                # Jika model menggunakan 1-based index (misal 1..25 bukan 0..24)
                if 0 not in lookup and len(batch) in lookup:
                    lookup = {k - 1: v for k, v in lookup.items()}

                batch_translated = []
                for i, c in enumerate(batch):
                    txt = lookup.get(i)
                    if not txt or not str(txt).strip():
                        if isinstance(parsed, list) and i < len(parsed):
                            elem = parsed[i]
                            if isinstance(elem, dict):
                                txt = elem.get("text", "")
                            elif isinstance(elem, str):
                                txt = elem

                    raw_txt = str(txt).strip() if (txt and str(txt).strip()) else c["text"]
                    if HANZI_RE.search(raw_txt):
                        final_txt = clean_remaining_hanzi(raw_txt)
                    else:
                        final_txt = raw_txt

                    batch_translated.append({
                        "start": c["start"],
                        "end": c["end"],
                        "text": final_txt
                    })

                translated_cues.extend(batch_translated)
                success = True
                time.sleep(delay_sec)
                break

            except Exception as e:
                last_error = str(e)
                time.sleep(delay_sec)

        if not success:
            print(f"[Warning] Batch {b_idx+1} gagal diterjemahkan ({last_error}), membersihkan karakter Mandarin.")
            for c in batch:
                translated_cues.append({
                    "start": c["start"],
                    "end": c["end"],
                    "text": clean_remaining_hanzi(c["text"])
                })

    return translated_cues

def build_srt_text(cues: list) -> str:
    lines = []
    for i, c in enumerate(cues, 1):
        s_ts = format_timestamp(c["start"])
        e_ts = format_timestamp(c["end"])
        lines.append(f"{i}\n{s_ts} --> {e_ts}\n{c['text']}\n")
    return "\n".join(lines)

def worker_loop():
    while True:
        task_id = JOB_QUEUE.get()
        if task_id is None:
            break
        task = TASKS_DB.get(task_id)
        if not task:
            JOB_QUEUE.task_done()
            continue

        try:
            task["state"] = "PROCESSING"
            task["progress"] = 5
            task["status"] = "Mengekstrak audio video..."
            save_queue_state()

            video_path = task["video_path"]
            cfg: GlobalSettings = task.get("config", GLOBAL_CONFIG)

            if not os.path.exists(video_path):
                raise FileNotFoundError(f"File video tidak ditemukan: {video_path}")

            audio_path = extract_audio(video_path)

            tx_data = transcribe(audio_path, cfg, task_id)

            try:
                if os.path.exists(audio_path):
                    os.remove(audio_path)
            except Exception:
                pass

            if tx_data.get("cues"):
                cues = tx_data["cues"]
            else:
                words = tx_data.get("words", [])
                cues = convert_words_to_srt(words, pause_sec=cfg.pause_sec, max_char_len=cfg.max_char_len)

            base, _ = os.path.splitext(video_path)
            srt_xkiro_path = f"{base}_xkiro.srt"
            srt_lite_path = f"{base}_literouter.srt"
            srt_main_path = f"{base}.srt"
            srt_mandarin_path = f"{base}_mandarin.srt"

            # Simpan transkripsi asli Mandarin langsung ke disk agar tidak pernah hilang!
            try:
                with open(srt_mandarin_path, "w", encoding="utf-8") as f_man:
                    f_man.write(build_srt_text(cues))
            except Exception as e_man:
                print(f"[Warning] Gagal simpan mandarin.srt: {e_man}")

            total_cues = len(cues)

            if cfg.translate_to_id and cues:
                mode = cfg.translation_mode
                
                if mode == "aichixia":
                    gemini_cues = translate_cues_with_provider(
                        cues, "aichixia", cfg, task_id, 45, 95
                    )
                    gemini_srt_txt = build_srt_text(gemini_cues)
                    srt_gemini_path = f"{base}_gemini.srt"
                    with open(srt_main_path, "w", encoding="utf-8") as f:
                        f.write(gemini_srt_txt)
                    with open(srt_gemini_path, "w", encoding="utf-8") as f:
                        f.write(gemini_srt_txt)
                    task["srt_path"] = srt_main_path
                    task["srt_gemini_path"] = srt_gemini_path
                    total_cues = len(gemini_cues)

                elif mode == "both":
                    task["progress"] = 45
                    task["status"] = "Menerjemahkan dengan xKiro & LiteRouter..."
                    save_queue_state()

                    with ThreadPoolExecutor(max_workers=2) as executor:
                        future_xkiro = executor.submit(
                            translate_cues_with_provider,
                            cues, "xkiro", cfg, task_id, 45, 95
                        )
                        future_lite = executor.submit(
                            translate_cues_with_provider,
                            cues, "literouter", cfg, task_id, 45, 95
                        )
                        xkiro_cues = None
                        lite_cues = None
                        errs = []
                        try:
                            xkiro_cues = future_xkiro.result()
                        except Exception as e:
                            errs.append(f"xKiro: {e}")
                        try:
                            lite_cues = future_lite.result()
                        except Exception as e:
                            errs.append(f"LiteRouter: {e}")

                        if not xkiro_cues and not lite_cues:
                            raise RuntimeError(f"Kedua AI penerjemah gagal: {' | '.join(errs)}")

                        if xkiro_cues:
                            xkiro_srt_txt = build_srt_text(xkiro_cues)
                            with open(srt_xkiro_path, "w", encoding="utf-8") as f:
                                f.write(xkiro_srt_txt)
                            task["srt_xkiro_path"] = srt_xkiro_path
                            task["srt_path"] = srt_xkiro_path
                            total_cues = len(xkiro_cues)

                        if lite_cues:
                            lite_srt_txt = build_srt_text(lite_cues)
                            with open(srt_lite_path, "w", encoding="utf-8") as f:
                                f.write(lite_srt_txt)
                            task["srt_lite_path"] = srt_lite_path
                            if not xkiro_cues:
                                task["srt_path"] = srt_lite_path
                                total_cues = len(lite_cues)

                        if xkiro_cues and not lite_cues:
                            task["status_note"] = f"Peringatan: LiteRouter gagal ({errs[-1]}), terselamatkan oleh xKiro."
                        elif lite_cues and not xkiro_cues:
                            task["status_note"] = f"Peringatan: xKiro gagal ({errs[0]}), terselamatkan oleh LiteRouter."

                elif mode == "literouter":
                    lite_cues = translate_cues_with_provider(
                        cues, "literouter", cfg, task_id, 45, 95
                    )
                    lite_srt_txt = build_srt_text(lite_cues)
                    with open(srt_main_path, "w", encoding="utf-8") as f:
                        f.write(lite_srt_txt)
                    with open(srt_lite_path, "w", encoding="utf-8") as f:
                        f.write(lite_srt_txt)
                    task["srt_path"] = srt_main_path
                    task["srt_lite_path"] = srt_lite_path
                    total_cues = len(lite_cues)

                else:
                    xkiro_cues = translate_cues_with_provider(
                        cues, "xkiro", cfg, task_id, 45, 95
                    )
                    xkiro_srt_txt = build_srt_text(xkiro_cues)
                    with open(srt_main_path, "w", encoding="utf-8") as f:
                        f.write(xkiro_srt_txt)
                    with open(srt_xkiro_path, "w", encoding="utf-8") as f:
                        f.write(xkiro_srt_txt)
                    task["srt_path"] = srt_main_path
                    task["srt_xkiro_path"] = srt_xkiro_path
                    total_cues = len(xkiro_cues)
            else:
                srt_txt = build_srt_text(cues)
                with open(srt_main_path, "w", encoding="utf-8") as f:
                    f.write(srt_txt)
                task["srt_path"] = srt_main_path

            task["progress"] = 100
            task["status"] = "SELESAI (2 SRT Tersimpan)" if cfg.translation_mode == "both" else "SELESAI!"
            task["state"] = "COMPLETED"
            task["total_cues"] = total_cues
            save_queue_state()

        except Exception as e:
            task["state"] = "ERROR"
            task["error"] = str(e)
            task["status"] = f"Error: {str(e)}"
            save_queue_state()
        finally:
            JOB_QUEUE.task_done()

# ==========================================
# DOWNLOAD WORKER (YT-DLP NO RENDER/CONVERT)
# ==========================================
def format_size(bytes_val):
    if not bytes_val or bytes_val <= 0:
        return "0 MB"
    mb = bytes_val / (1024 * 1024)
    if mb >= 1024:
        return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"

def unwrap_png(data: bytes) -> bytes:
    PNG_MAGIC = b'\x89PNG\r\n\x1a\n'
    if not data.startswith(PNG_MAGIC):
        return data
    i = len(PNG_MAGIC)
    while i + 8 <= len(data):
        length = struct.unpack('>I', data[i:i+4])[0]
        chunk_type = struct.unpack('>I', data[i+4:i+8])[0]
        d = i + 8
        if d + length > len(data):
            break
        if chunk_type == 1919898980: # 'rout' chunk
            flags = data[d]
            payload = data[d+1 : d+length]
            if flags & 1:
                try:
                    return zlib.decompress(payload, -zlib.MAX_WBITS)
                except Exception:
                    return zlib.decompress(payload)
            else:
                return payload
        i = d + length + 4
    # Fc2stream: segmen video disimpan mentah setelah chunk IEND
    iend_idx = data.find(b'IEND')
    if iend_idx != -1 and len(data) > iend_idx + 8:
        return data[iend_idx + 8:]
    return data

def _unpack_packed_js(text: str) -> str:
    m = re.search(r"eval\(function\(p,a,c,k,e,d\)\{.*?\}\('(.+)',(\d+),(\d+),'(.+?)'\.split\('\|'\)\)", text, re.S)
    if not m:
        return text
    packed, base_str, count_str, words_raw = m.group(1), m.group(2), m.group(3), m.group(4)
    base = int(base_str)
    words = words_raw.split("|")
    digits = "0123456789abcdefghijklmnopqrstuvwxyz"

    def to_base(n):
        if n == 0:
            return "0"
        s = ""
        while n:
            s = digits[n % base] + s
            n //= base
        return s

    p = packed
    for i in range(int(count_str) - 1, -1, -1):
        if i < len(words) and words[i]:
            p = re.sub(r"\b" + re.escape(to_base(i)) + r"\b", lambda _m, w=words[i]: w, p)
    return p

def resolve_supjav(page_url: str):
    from curl_cffi import requests as c_requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://supjav.com/'
    }
    resp = c_requests.get(page_url, headers=headers, impersonate='chrome124', timeout=25)
    resp.raise_for_status()
    html = resp.text

    title = ""
    t_match = re.search(r'<title>(.*?)</title>', html, re.I | re.S)
    if t_match:
        title = re.sub(r'[\\/*?:"<>|]', '', t_match.group(1)).strip()[:100]

    link_match = re.search(r'data-link="([0-9a-f]{40,})"', html)
    if not link_match:
        raise Exception("Server video Supjav tidak ditemukan di halaman")
    link = link_match.group(1)

    base_ep = "https://lk1.supremejav.com/supjav.php"
    c_requests.get(f"{base_ep}?l={link}&bg=", headers=headers, impersonate='chrome124', timeout=20)

    r2 = c_requests.get(
        f"{base_ep}?c={link[::-1]}",
        headers={**headers, 'Referer': f"{base_ep}?l={link}&bg="},
        impersonate='chrome124', timeout=20
    )
    r2.raise_for_status()

    unpacked = _unpack_packed_js(r2.text)
    for key in ("hls4", "hls3", "hls2"):
        mm = re.search(r'"' + key + r'":"([^"]+)"', unpacked)
        if mm:
            stream = mm.group(1).strip()
            if stream.startswith('/'):
                stream = "https://fc2stream.tv" + stream
            if stream.startswith('http'):
                return stream, title
    raise Exception("Link stream segar tidak ditemukan di halaman player Supjav")

def download_supjav_hls(dl_id: str, req: "DownloadLinkRequest", info_entry: dict, save_dir: str):
    info_entry["state"] = "DOWNLOADING"
    info_entry["status"] = "Mengambil playlist video (HLS)..."
    info_entry["progress"] = 2.0

    master_url = req.url.strip()
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://fc2stream.tv/'
    }
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    def fetch_text(url):
        r = session.get(url, headers=headers, timeout=25)
        r.raise_for_status()
        return unwrap_png(r.content).decode('utf-8', errors='ignore')

    text = fetch_text(master_url)
    media_url = master_url
    picked_height = 0

    if '#EXT-X-STREAM-INF' in text:
        lines = text.splitlines()
        variants = []
        for i, line in enumerate(lines):
            if not line.startswith('#EXT-X-STREAM-INF'):
                continue
            res = re.search(r'RESOLUTION=(\d+)x(\d+)', line)
            bw = re.search(r'BANDWIDTH=(\d+)', line)
            uri = ""
            for j in range(i + 1, len(lines)):
                cand = lines[j].strip()
                if cand and not cand.startswith('#'):
                    uri = cand
                    break
            if uri:
                variants.append((int(res.group(2)) if res else 0, int(bw.group(1)) if bw else 0, uri))
        if not variants:
            raise Exception("Playlist master tidak berisi varian kualitas")
        variants.sort(key=lambda v: (v[0], v[1]))
        heights = sorted({v[0] for v in variants if v[0] > 0})
        if req.use_kaggle_480p:
            target = heights[0]
        else:
            target = next((h for h in heights if h >= 720), heights[-1])
        pick = next((v for v in variants if v[0] == target), variants[-1])
        picked_height = pick[0]
        info_entry["resolution"] = f"{pick[0]}p"
        media_url = media_url.rsplit('/', 1)[0] + '/' + pick[2] if not pick[2].startswith('http') else pick[2]
        text = fetch_text(media_url)

    segments = []
    tot_duration = 0.0
    for line in text.splitlines():
        line = line.strip()
        if line.startswith('#EXTINF:'):
            try:
                tot_duration += float(line.split(':')[1].split(',')[0].strip())
            except Exception:
                pass
        elif line and not line.startswith('#'):
            segments.append(line if line.startswith('http') else media_url.rsplit('/', 1)[0] + '/' + line)

    if not segments:
        raise Exception("Tidak ada segmen video di playlist fc2stream")
    if tot_duration > 0:
        info_entry["duration_str"] = f"{int(tot_duration // 60)}m {int(tot_duration % 60)}s"

    if req.custom_name and req.custom_name.strip():
        title = re.sub(r'[\\/*?:"<>|]', "", req.custom_name.strip())
    else:
        m_id = re.search(r'/(\d{6,})/index', media_url)
        title = f"fc2_{m_id.group(1)}" if m_id else f"fc2_{dl_id}"
    info_entry["title"] = title

    total_segments = len(segments)
    info_entry["status"] = f"Mengunduh {total_segments} segmen video ({info_entry.get('resolution', 'HLS')})..."

    out_mp4 = os.path.join(save_dir, f"{title}.mp4")
    temp_ts = os.path.join(save_dir, f"temp_{dl_id}_{int(time.time())}.ts")
    downloaded_chunks = [None] * total_segments
    completed_count = 0
    total_bytes = 0
    start_time = time.time()

    def fetch_seg(idx, seg_url):
        r = session.get(seg_url, headers=headers, timeout=25)
        r.raise_for_status()
        return idx, unwrap_png(r.content)

    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_seg, i, s): i for i, s in enumerate(segments)}
        for future in as_completed(futures):
            idx, chunk_bytes = future.result()
            downloaded_chunks[idx] = chunk_bytes
            completed_count += 1
            total_bytes += len(chunk_bytes)

            elapsed = time.time() - start_time
            speed = total_bytes / (elapsed if elapsed > 0 else 1)
            pct = round((completed_count / total_segments) * 96.0, 1)
            est_total = (total_bytes / completed_count) * total_segments
            eta = (est_total - total_bytes) / speed if speed > 0 else 0
            speed_mb = speed / (1024 * 1024)

            info_entry["progress"] = pct
            info_entry["downloaded_str"] = format_size(total_bytes)
            info_entry["total_str"] = format_size(est_total)
            info_entry["speed_str"] = f"{speed_mb:.1f} MB/s"
            info_entry["eta_str"] = f"{int(eta)} detik lagi" if eta > 0 else "Hampir selesai..."
            info_entry["status"] = f"{pct}% • {info_entry['speed_str']} ({info_entry['downloaded_str']} / {info_entry['total_str']}) [{completed_count}/{total_segments}]"

    missing = sum(1 for c in downloaded_chunks if not c)
    if missing:
        raise Exception(f"{missing} dari {total_segments} segmen gagal diunduh")

    info_entry["status"] = "Menggabungkan potongan file (Direct Stream Copy)..."
    info_entry["progress"] = 97.0
    with open(temp_ts, "wb") as f_out:
        for chunk in downloaded_chunks:
            f_out.write(chunk)

    # Sumber sudah 480p -> tidak perlu encode ulang (hemat waktu, kualitas tetap asli)
    if req.use_kaggle_480p and picked_height > 480:
        info_entry["status"] = "Mengompres ke 480p (Intel QuickSync)..."
        cmd = ["ffmpeg", "-y", "-i", temp_ts, "-vf", "scale=-2:480",
               "-c:v", "h264_qsv", "-global_quality", "28",
               "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out_mp4]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            cmd = ["ffmpeg", "-y", "-i", temp_ts, "-vf", "scale=-2:480",
                   "-c:v", "libx264", "-preset", "veryfast", "-crf", "26",
                   "-c:a", "aac", "-b:a", "96k", "-movflags", "+faststart", out_mp4]
            subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        cmd = ["ffmpeg", "-y", "-i", temp_ts, "-c", "copy", "-movflags", "+faststart", out_mp4]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp_ts):
        try:
            os.remove(temp_ts)
        except Exception:
            pass

    info_entry["state"] = "COMPLETED"
    info_entry["progress"] = 100.0
    info_entry["status"] = "Selesai diunduh ke laptop!"
    info_entry["file_path"] = out_mp4
    info_entry["file_size_str"] = format_size(os.path.getsize(out_mp4))

    if req.auto_queue and os.path.exists(out_mp4):
        t_id = str(uuid.uuid4())[:8]
        TASKS_DB[t_id] = {
            "task_id": t_id,
            "video_path": out_mp4,
            "video_name": os.path.basename(out_mp4),
            "state": "QUEUED",
            "progress": 0,
            "status": "Dalam Antrian Subtitle...",
            "error": None,
            "srt_path": None,
            "srt_xkiro_path": None,
            "srt_lite_path": None,
            "total_cues": 0,
            "config": GLOBAL_CONFIG,
            "created_at": time.time()
        }
        JOB_QUEUE.put(t_id)
        save_queue_state()
        info_entry["status"] = "Selesai diunduh & Otomatis Masuk Antrian Subtitle!"

def resolve_kvs_video(page_url: str):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': page_url
    }
    resp = requests.get(page_url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text

    t_match = re.search(r'<title>(.*?)</title>', html, re.I)
    title = t_match.group(1).split('-')[0].strip() if t_match else "video"
    title = re.sub(r'[\\/*?:"<>|]', '', title).strip()

    m_alt = re.search(r'video_alt_url\s*:\s*[\'"]([^\'"]+)[\'"]', html)
    if m_alt:
        return m_alt.group(1), title

    m_v = re.search(r'video_url\s*:\s*[\'"]([^\'"]+)[\'"]', html)
    if m_v:
        return m_v.group(1), title

    m_c = re.search(r'"contentUrl":\s*"([^"]+)"', html)
    if m_c:
        return m_c.group(1), title

    return None, title

def download_rou_video(dl_id: str, req: DownloadLinkRequest, info_entry: dict, save_dir: str):
    info_entry["state"] = "DOWNLOADING"
    info_entry["status"] = "Mengekstrak playlist Rou.video..."
    info_entry["progress"] = 2.0

    page_url = req.url.strip()
    m = re.search(r'/v/([a-zA-Z0-9]+)', page_url)
    video_id = m.group(1) if m else "video"

    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://rou.video/',
        'Origin': 'https://rou.video'
    }

    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=25, pool_maxsize=25, max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    # 1. Fetch HTML
    resp = session.get(page_url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text

    # Title
    title = f"rou_{video_id}"
    orig_title = ""
    t_match = re.search(r'<title>(.*?)</title>', html)
    if t_match:
        t_clean = t_match.group(1).split('-')[0].strip()
        orig_title = t_clean
        t_clean = re.sub(r'[\\/*?:"<>|]', "", t_clean).strip()
        if t_clean:
            title = t_clean[:80]

    # Duration from JSON-LD
    dur_match = re.search(r'"duration":\s*"PT(\d+)S"', html)
    if dur_match:
        d_sec = int(dur_match.group(1))
        info_entry["duration_str"] = f"{d_sec//60}m {d_sec%60}s"
    
    # Thumbnail Cover URL
    thumb_match = re.search(r'"thumbnailUrl":\s*\["([^"]+)"\]', html)
    if not thumb_match:
        thumb_match = re.search(r'<meta\s+property="og:image"\s+content="([^"]+)"', html)
    if thumb_match:
        info_entry["thumbnail"] = thumb_match.group(1)

    info_entry["resolution"] = "720p HD"

    if req.custom_name and req.custom_name.strip():
        clean_name = re.sub(r'[\\/*?:"<>|]', "", req.custom_name.strip())
        if clean_name:
            title = clean_name
        info_entry["title"] = clean_name
    else:
        info_entry["title"] = orig_title or title

    # 2. Extract ev decryption
    hls_api_url = f"https://rou.video/api/hls/{video_id}"
    ev_match = re.search(r'ev:\$R\[\d+\]=\{d:"([^"]+)",k:(\d+)\}', html)
    if not ev_match:
        ev_match = re.search(r'ev:\{d:"([^"]+)",k:(\d+)\}', html)
    
    if ev_match:
        d_val = ev_match.group(1)
        k_val = int(ev_match.group(2))
        try:
            raw = base64.b64decode(d_val).decode('latin1')
            unmasked = ''.join([chr(ord(c) - k_val) for c in raw])
            ev_data = json.loads(unmasked)
            vurl = ev_data.get('videoUrl')
            if vurl:
                hls_api_url = "https://rou.video" + vurl if vurl.startswith('/') else vurl
        except Exception as e:
            print(f"ev decode error: {e}")

    # 3. Fetch M3U8 PNG playlist
    hls_headers = dict(headers)
    hls_headers['Referer'] = page_url
    resp_hls = session.get(hls_api_url, headers=hls_headers, timeout=15)
    resp_hls.raise_for_status()
    playlist_text = unwrap_png(resp_hls.content).decode('utf-8', errors='ignore')

    segments = []
    tot_duration = 0.0
    for line in playlist_text.splitlines():
        line = line.strip()
        if line.startswith('#EXTINF:'):
            try:
                tot_duration += float(line.split(':')[1].split(',')[0].strip())
            except Exception:
                pass
        elif line and not line.startswith('#'):
            segments.append(line)

    if not info_entry.get("duration_str") and tot_duration > 0:
        info_entry["duration_str"] = f"{int(tot_duration//60)}m {int(tot_duration%60)}s"

    total_segments = len(segments)
    if total_segments == 0:
        raise Exception("Tidak ada segmen video yang ditemukan di playlist!")

    dur_badge = f" ({info_entry.get('duration_str')})" if info_entry.get('duration_str') else ""
    info_entry["status"] = f"Mulai download {total_segments} segmen video{dur_badge}..."

    temp_ts = os.path.join(save_dir, f"temp_{video_id}_{int(time.time())}.ts")
    out_mp4 = os.path.join(save_dir, f"{title}.mp4")

    downloaded_chunks = [None] * total_segments
    completed_count = 0
    total_bytes = 0
    start_time = time.time()

    def fetch_seg(idx, seg_url):
        seg_headers = dict(headers)
        seg_headers['Referer'] = page_url
        r = session.get(seg_url, headers=seg_headers, timeout=20)
        r.raise_for_status()
        unwrapped = unwrap_png(r.content)
        return idx, unwrapped

    # 12 parallel threads
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = {executor.submit(fetch_seg, i, seg_url): i for i, seg_url in enumerate(segments)}
        for future in as_completed(futures):
            idx, chunk_bytes = future.result()
            downloaded_chunks[idx] = chunk_bytes
            completed_count += 1
            total_bytes += len(chunk_bytes)
            
            elapsed = time.time() - start_time
            speed = total_bytes / (elapsed if elapsed > 0 else 1)
            pct = round((completed_count / total_segments) * 100, 1)
            
            est_total = (total_bytes / completed_count) * total_segments if completed_count > 0 else 0
            rem_bytes = est_total - total_bytes
            eta = (rem_bytes / speed) if speed > 0 else 0
            
            info_entry["progress"] = pct
            info_entry["downloaded_str"] = format_size(total_bytes)
            info_entry["total_str"] = format_size(est_total)
            speed_mb = (speed / (1024 * 1024))
            info_entry["speed_str"] = f"{speed_mb:.1f} MB/s"
            info_entry["eta_str"] = f"{int(eta)} detik lagi" if eta > 0 else "Hampir selesai..."
            info_entry["status"] = f"{pct}% • {info_entry['speed_str']} ({info_entry['downloaded_str']} / {info_entry['total_str']}) [{completed_count}/{total_segments}]"

    info_entry["status"] = "Menggabungkan potongan file (Direct Stream Copy)..."
    info_entry["progress"] = 98.0

    with open(temp_ts, "wb") as f_out:
        for chunk in downloaded_chunks:
            if chunk:
                f_out.write(chunk)

    # Remuxing atau Kompresi 480p
    if req.use_kaggle_480p:
        info_entry["status"] = "Mengompres ke 480p (Intel QuickSync GPU Acceleration)..."
        info_entry["progress"] = 98.0
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_ts,
            "-vf", "scale=-2:480",
            "-c:v", "h264_qsv",
            "-global_quality", "28",
            "-c:a", "aac",
            "-b:a", "96k",
            "-movflags", "+faststart",
            out_mp4
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            # Fallback jika QSV busy
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", temp_ts,
                "-vf", "scale=-2:480",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "26",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                out_mp4
            ]
            subprocess.run(cmd_fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        # Direct stream copy tanpa re-encode
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_ts,
            "-c", "copy",
            "-movflags", "+faststart",
            out_mp4
        ]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    
    if os.path.exists(temp_ts):
        try:
            os.remove(temp_ts)
        except Exception:
            pass

    info_entry["state"] = "COMPLETED"
    info_entry["progress"] = 100.0
    status_suffix = " (Kompresi 480p Selesai)" if req.use_kaggle_480p else ""
    info_entry["status"] = f"Selesai diunduh ke laptop!{status_suffix}"
    info_entry["file_path"] = out_mp4
    info_entry["file_size_str"] = format_size(os.path.getsize(out_mp4))

    if req.auto_queue and os.path.exists(out_mp4):
        t_id = str(uuid.uuid4())[:8]
        TASKS_DB[t_id] = {
            "task_id": t_id,
            "video_path": out_mp4,
            "video_name": os.path.basename(out_mp4),
            "state": "QUEUED",
            "progress": 0,
            "status": "Dalam Antrian Subtitle...",
            "error": None,
            "srt_path": None,
            "srt_xkiro_path": None,
            "srt_lite_path": None,
            "total_cues": 0,
            "config": GLOBAL_CONFIG,
            "created_at": time.time()
        }
        JOB_QUEUE.put(t_id)

def resolve_avple_manifest(page_url: str):
    from curl_cffi import requests as c_requests
    headers = {
        'Referer': 'https://avple.tv/',
        'Origin': 'https://avple.tv',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    r = c_requests.get(page_url.strip(), impersonate='chrome124', headers=headers, timeout=20)
    if r.status_code != 200:
        return None, None
    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text)
    if not m:
        return None, None
    data = json.loads(m.group(1))
    instance = data.get("props", {}).get("pageProps", {}).get("instance", {})
    if not instance:
        return None, None
    title = instance.get("title", "avple_video")
    play = instance.get("play", "")
    if not play:
        return None, title

    m3u8_text = None
    base_url = None

    # Jika play langsung berupa URL penuh .m3u8 (seperti krevonix.com atau server pihak ke-3)
    if play.startswith("http://") or play.startswith("https://"):
        try:
            r_direct = requests.get(play, headers=headers, verify=False, timeout=12)
            if r_direct.status_code == 200 and '#EXTM3U' in r_direct.text:
                m3u8_text = r_direct.text
                base_url = play.rsplit('/', 1)[0]
                # Jika master playlist dengan stream nested
                for line in m3u8_text.splitlines():
                    l = line.strip()
                    if l and not l.startswith('#') and ('.m3u8' in l or 'hls' in l):
                        import urllib.parse
                        nested_url = urllib.parse.urljoin(play, l)
                        r_nested = requests.get(nested_url, headers=headers, verify=False, timeout=12)
                        if r_nested.status_code == 200 and '#EXTM3U' in r_nested.text:
                            m3u8_text = r_nested.text
                            base_url = nested_url.rsplit('/', 1)[0]
                            play = nested_url
                            break
        except Exception as e:
            print(f"[AVPLE-DIRECT-M3U8-ERR] {e}")
    else:
        cdns = ["d862cp1.cdnedge.live", "q2cyl71.cdnedge.live", "u89ey1.cdnedge.live", "wo8801.cdnedge.live", "6m7d1.cdnedge.live", "fa6781.cdnedge.live"]
        for cdn in cdns:
            t_url = f"https://{cdn}/file/avple-asserts/{play}"
            try:
                res = c_requests.get(t_url, impersonate='chrome124', headers=headers, timeout=8)
                if res.status_code == 200 and '#EXTM3U' in res.text:
                    m3u8_text = res.text
                    base_url = f"https://{cdn}/file/avple-asserts/{play.rsplit('/', 1)[0]}"
                    break
            except Exception:
                pass

    if not m3u8_text or not base_url:
        return None, title

    import urllib.parse
    rewritten_lines = []
    for line in m3u8_text.splitlines():
        line = line.strip()
        if line.startswith('#EXT-X-KEY:'):
            m_uri = re.search(r'URI=["\'](.*?)["\']', line)
            if m_uri:
                k_val = m_uri.group(1)
                full_k = k_val if k_val.startswith('http') else urllib.parse.urljoin(base_url + '/', k_val)
                line = re.sub(r'URI=["\'].*?["\']', f'URI="{full_k}"', line)
            rewritten_lines.append(line)
        elif line and not line.startswith('#'):
            if line.startswith('http'):
                rewritten_lines.append(line)
            else:
                rewritten_lines.append(urllib.parse.urljoin(base_url + '/', line))
        else:
            rewritten_lines.append(line)

    abs_m3u8 = "\n".join(rewritten_lines)
    dp_resp = requests.post('https://dpaste.org/api/', data={'content': abs_m3u8, 'format': 'url', 'expiry_days': 1}, timeout=10)
    if dp_resp.status_code == 200:
        manifest_url = dp_resp.text.strip() + "/raw"
        return manifest_url, title

    return None, title

def download_avple_video(dl_id: str, req: DownloadLinkRequest, info_entry: dict, save_dir: str):
    info_entry["state"] = "DOWNLOADING"
    info_entry["status"] = "Menganalisis link Avple & Mem-bypass Cloudflare Turnstile..."
    info_entry["progress"] = 2.0

    from curl_cffi import requests as c_requests

    page_url = req.url.strip()
    headers = {
        'Referer': 'https://avple.tv/',
        'Origin': 'https://avple.tv',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    r = c_requests.get(page_url, impersonate='chrome124', headers=headers, timeout=25)
    if r.status_code != 200:
        raise Exception(f"Gagal membuka halaman Avple: HTTP {r.status_code}")

    m = re.search(r'<script id="__NEXT_DATA__"[^>]*>(.*?)</script>', r.text)
    if not m:
        raise Exception("Gagal mengekstrak data video dari Avple (__NEXT_DATA__ tidak ditemukan)")

    data = json.loads(m.group(1))
    instance = data.get("props", {}).get("pageProps", {}).get("instance", {})
    if not instance:
        raise Exception("Data video instance tidak ditemukan pada halaman Avple")

    raw_title = instance.get("title", "avple_video")
    clean_title = re.sub(r'[\\/*?:"<>|]', "", raw_title).strip()
    title = clean_title if clean_title else "avple_video"
    if req.custom_name and req.custom_name.strip():
        title = re.sub(r'[\\/*?:"<>|]', "", req.custom_name.strip())

    info_entry["title"] = title
    info_entry["thumbnail"] = instance.get("img_preview", "")
    info_entry["duration_str"] = instance.get("timeLengh", "")

    play_path = instance.get("play", "")
    if not play_path:
        raise Exception("Path streaming video tidak ditemukan")

    cdns = [
        "d862cp1.cdnedge.live", "q2cyl71.cdnedge.live", "u89ey1.cdnedge.live",
        "wo8801.cdnedge.live", "6m7d1.cdnedge.live", "fa6781.cdnedge.live", "pg2z71.cdnedge.live"
    ]
    hls_url = None
    playlist_text = None
    base_cdn_url = None

    if play_path.startswith("http://") or play_path.startswith("https://"):
        try:
            r_m3u8 = requests.get(play_path, headers=headers, verify=False, timeout=12)
            if r_m3u8.status_code == 200 and '#EXTM3U' in r_m3u8.text:
                hls_url = play_path
                playlist_text = r_m3u8.text
                base_cdn_url = play_path.rsplit('/', 1)[0]
                for line in playlist_text.splitlines():
                    l = line.strip()
                    if l and not l.startswith('#') and ('.m3u8' in l or 'hls' in l):
                        import urllib.parse
                        nested_url = urllib.parse.urljoin(play_path, l)
                        r_nested = requests.get(nested_url, headers=headers, verify=False, timeout=12)
                        if r_nested.status_code == 200 and '#EXTM3U' in r_nested.text:
                            playlist_text = r_nested.text
                            base_cdn_url = nested_url.rsplit('/', 1)[0]
                            break
        except Exception:
            pass
    else:
        for cdn in cdns:
            test_url = f"https://{cdn}/file/avple-asserts/{play_path}"
            try:
                r_m3u8 = c_requests.get(test_url, impersonate='chrome124', headers=headers, timeout=8)
                if r_m3u8.status_code == 200 and "#EXTM3U" in r_m3u8.text:
                    hls_url = test_url
                    playlist_text = r_m3u8.text
                    base_cdn_url = test_url.rsplit("/", 1)[0]
                    break
            except Exception:
                continue

    if not playlist_text or not base_cdn_url:
        raise Exception("Gagal mendapatkan playlist HLS dari CDN Avple")

    import urllib.parse
    segments_with_keys = []
    current_key_bytes = None
    key_cache = {}

    for line in playlist_text.splitlines():
        line = line.strip()
        if line.startswith('#EXT-X-KEY'):
            if 'METHOD=NONE' in line:
                current_key_bytes = None
            elif 'METHOD=AES-128' in line:
                m_uri = re.search(r'URI=["\'](.*?)["\']', line)
                if m_uri:
                    k_url = m_uri.group(1)
                    if k_url not in key_cache:
                        try:
                            k_res = requests.get(k_url, headers=headers, verify=False, timeout=10)
                            if k_res.status_code == 200:
                                key_cache[k_url] = k_res.content
                        except Exception:
                            pass
                    current_key_bytes = key_cache.get(k_url)
        elif line and not line.startswith('#'):
            seg_url = line if line.startswith("http") else urllib.parse.urljoin(base_cdn_url + '/', line)
            segments_with_keys.append((seg_url, current_key_bytes))

    total_segments = len(segments_with_keys)
    if total_segments == 0:
        raise Exception("Playlist kosong atau tidak ada segmen video")

    info_entry["status"] = f"Mempersiapkan download {total_segments} segmen video..."
    downloaded_chunks = [None] * total_segments
    completed_count = 0
    total_bytes = 0
    start_time = time.time()

    temp_ts = os.path.join(save_dir, f"temp_{dl_id}.ts")
    out_mp4 = os.path.join(save_dir, f"{title}.mp4")

    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    def fetch_seg(idx, seg_item):
        seg_url, key_bytes = seg_item
        for _ in range(3):
            try:
                res = c_requests.get(seg_url, headers=headers, impersonate='chrome124', timeout=20)
                if res.status_code == 200 and len(res.content) > 0:
                    data = res.content
                    if key_bytes and len(key_bytes) == 16:
                        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(b'\x00' * 16))
                        decryptor = cipher.decryptor()
                        data = decryptor.update(data) + decryptor.finalize()
                    return idx, data
            except Exception:
                time.sleep(1)
        return idx, b""

    with ThreadPoolExecutor(max_workers=10) as executor:
        futures = {executor.submit(fetch_seg, i, item): i for i, item in enumerate(segments_with_keys)}
        for future in as_completed(futures):
            idx, chunk_bytes = future.result()
            downloaded_chunks[idx] = chunk_bytes
            completed_count += 1
            total_bytes += len(chunk_bytes)
            
            elapsed = time.time() - start_time
            speed = total_bytes / (elapsed if elapsed > 0 else 1)
            pct = round((completed_count / total_segments) * 100, 1)
            
            est_total = (total_bytes / completed_count) * total_segments if completed_count > 0 else 0
            rem_bytes = est_total - total_bytes
            eta = (rem_bytes / speed) if speed > 0 else 0
            
            info_entry["progress"] = pct
            info_entry["downloaded_str"] = format_size(total_bytes)
            info_entry["total_str"] = format_size(est_total)
            speed_mb = (speed / (1024 * 1024))
            info_entry["speed_str"] = f"{speed_mb:.1f} MB/s"
            info_entry["eta_str"] = f"{int(eta)} detik lagi" if eta > 0 else "Hampir selesai..."
            info_entry["status"] = f"{pct}% • {info_entry['speed_str']} ({info_entry['downloaded_str']} / {info_entry['total_str']}) [{completed_count}/{total_segments}]"

    info_entry["status"] = "Menggabungkan potongan file (Direct Stream Copy)..."
    info_entry["progress"] = 98.0

    with open(temp_ts, "wb") as f_out:
        for chunk in downloaded_chunks:
            if chunk:
                f_out.write(chunk)

    if req.use_kaggle_480p:
        info_entry["status"] = "Mengompres ke 480p (Intel QuickSync GPU Acceleration)..."
        cmd = [
            "ffmpeg", "-y",
            "-i", temp_ts,
            "-vf", "scale=-2:480",
            "-c:v", "h264_qsv",
            "-global_quality", "28",
            "-c:a", "aac",
            "-b:a", "96k",
            "-movflags", "+faststart",
            out_mp4
        ]
        res = subprocess.run(cmd, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
        if res.returncode != 0:
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", temp_ts,
                "-vf", "scale=-2:480",
                "-c:v", "libx264",
                "-preset", "veryfast",
                "-crf", "26",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                out_mp4
            ]
            subprocess.run(cmd_fallback, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    else:
        cmd = ["ffmpeg", "-y", "-i", temp_ts, "-c", "copy", "-movflags", "+faststart", out_mp4]
        subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    if os.path.exists(temp_ts):
        try:
            os.remove(temp_ts)
        except Exception:
            pass

    info_entry["state"] = "COMPLETED"
    info_entry["progress"] = 100.0
    status_suffix = " (Kompresi 480p Selesai)" if req.use_kaggle_480p else ""
    info_entry["status"] = f"Selesai diunduh ke laptop!{status_suffix}"
    info_entry["file_path"] = out_mp4
    info_entry["file_size_str"] = format_size(os.path.getsize(out_mp4))

    if req.auto_queue and os.path.exists(out_mp4):
        t_id = str(uuid.uuid4())[:8]
        TASKS_DB[t_id] = {
            "task_id": t_id,
            "video_path": out_mp4,
            "video_name": os.path.basename(out_mp4),
            "state": "QUEUED",
            "progress": 0,
            "status": "Dalam Antrian Subtitle...",
            "error": None,
            "srt_path": None,
            "srt_xkiro_path": None,
            "srt_lite_path": None,
            "total_cues": 0,
            "config": GLOBAL_CONFIG,
            "created_at": time.time()
        }
        JOB_QUEUE.put(t_id)
        save_queue_state()
        info_entry["status"] = "Selesai diunduh & Otomatis Masuk Antrian Subtitle!"

def get_active_kaggle_url(info_entry=None) -> str:
    # 1. Cek config aktif dulu (apakah masih benar-benar merespon?)
    if GLOBAL_CONFIG.kaggle_url and "trycloudflare.com" in GLOBAL_CONFIG.kaggle_url:
        try:
            r = requests.get(f"{GLOBAL_CONFIG.kaggle_url}/docs", timeout=3)
            if r.status_code == 200:
                return GLOBAL_CONFIG.kaggle_url
        except Exception:
            pass

    # 2. Ambil URL terbaru dari cloud relay ntfy.sh dan uji satu per satu dari yang paling baru
    try:
        r = requests.get("https://ntfy.sh/ziranaijav_cloud_worker_2026/json?poll=1", timeout=5)
        if r.ok:
            for line in reversed(r.text.strip().splitlines()):
                try:
                    data = json.loads(line)
                    msg = data.get("message", "")
                    m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", msg)
                    if m:
                        t_url = m.group(0)
                        chk = requests.get(f"{t_url}/docs", timeout=3)
                        if chk.status_code == 200:
                            GLOBAL_CONFIG.kaggle_url = t_url
                            with open(r"D:\translate\remote_web\kaggle_worker_url.txt", "w") as f_u:
                                f_u.write(t_url)
                            return t_url
                except Exception:
                    pass
    except Exception:
        pass

    # 3. Cek file txt tersimpan (jika respon 200)
    txt_path = r"D:\translate\remote_web\kaggle_worker_url.txt"
    if os.path.exists(txt_path):
        try:
            with open(txt_path) as f:
                u = f.read().strip()
                if u and requests.get(f"{u}/docs", timeout=3).status_code == 200:
                    GLOBAL_CONFIG.kaggle_url = u
                    return u
        except Exception:
            pass

    # 4. Jika semua offline, nyalakan otomatis lewat Kaggle CLI!
    try:
        if info_entry:
            info_entry["status"] = "Kaggle offline. Mem-boot GPU Cloud Kaggle secara otomatis..."
        subprocess.run(["kaggle", "kernels", "push", "-p", r"D:\translate\remote_web\kaggle_push"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL, timeout=25)
        # Tunggu ntfy relay melapor
        for _ in range(25):
            time.sleep(5)
            try:
                r = requests.get("https://ntfy.sh/ziranaijav_cloud_worker_2026/json?poll=1", timeout=4)
                if r.ok:
                    for line in reversed(r.text.strip().splitlines()):
                        data = json.loads(line)
                        m = re.search(r"https://[a-zA-Z0-9-]+\.trycloudflare\.com", data.get("message", ""))
                        if m:
                            t_url = m.group(0)
                            chk = requests.get(f"{t_url}/docs", timeout=4)
                            if chk.status_code == 200:
                                GLOBAL_CONFIG.kaggle_url = t_url
                                with open(r"D:\translate\remote_web\kaggle_worker_url.txt", "w") as f_u:
                                    f_u.write(t_url)
                                return t_url
            except Exception:
                pass
    except Exception as e:
        print(f"[KAGGLE-AUTOSTART-ERR] {e}")

    raise RuntimeError("Worker Cloud Kaggle sedang offline dan belum berhasil dihubungkan.")

def download_via_kaggle(dl_id: str, req: DownloadLinkRequest, info_entry: dict, save_dir: str):
    kaggle_url = get_active_kaggle_url(info_entry)

    info_entry["state"] = "DOWNLOADING"
    info_entry["status"] = "Menghubungkan ke Cloud Kaggle..."
    info_entry["progress"] = 5.0

    # 1. Kirim tugas kompresi ke Kaggle (dengan auto-retry jika URL baru berganti)
    payload = {
        "url": req.url.strip(),
        "custom_name": req.custom_name or ""
    }
    resp = None
    for attempt in range(3):
        try:
            resp = requests.post(f"{kaggle_url}/api/compress", json=payload, timeout=30)
            resp.raise_for_status()
            break
        except Exception as e:
            if attempt == 2:
                raise
            time.sleep(2)
            kaggle_url = get_active_kaggle_url(info_entry)
    job_id = resp.json()["job_id"]

    # 2. Polling progress Kaggle
    job_info = None
    poll_fail = 0
    job_missing = 0
    while True:
        time.sleep(3)
        payload = None
        st_code = 0
        try:
            poll = requests.get(f"{kaggle_url}/api/job/{job_id}", timeout=15)
            st_code = poll.status_code
            if poll.ok:
                payload = poll.json()
        except Exception:
            payload = None

        if payload is None:
            poll_fail += 1
            if poll_fail % 20 == 0:
                try:
                    kaggle_url = get_active_kaggle_url(info_entry)
                except Exception:
                    pass
            if poll_fail >= 100:
                raise Exception("Worker Kaggle tidak merespon selama ~5 menit (tunnel putus)")
            continue
        poll_fail = 0

        if st_code == 404 or "state" not in payload:
            job_missing += 1
            if job_missing >= 10:
                raise Exception("Job sudah hilang di Kaggle (kernel kemungkinan restart), silakan ulangi unduh")
            continue
        job_missing = 0
        job_info = payload

        st = job_info.get("status", "")
        pr = min(float(job_info.get("progress", 10)), 88.0)
        info_entry["status"] = f"[Cloud Kaggle] {st}"
        info_entry["progress"] = pr

        if job_info.get("state") == "COMPLETED":
            break
        elif job_info.get("state") == "ERROR":
            raise Exception(job_info.get("error", "Terjadi kesalahan di Cloud Kaggle"))

    # 3. Download hasil file 480p dari Kaggle ke Laptop
    info_entry["status"] = "Mengunduh file 480p yang sudah ringan dari Kaggle ke laptop..."
    info_entry["progress"] = 90.0

    dl_endpoint = f"{kaggle_url}/api/download/{job_id}"
    file_name = job_info.get("file_name", "video_480p.mp4")
    out_file = os.path.join(save_dir, file_name)

    with requests.get(dl_endpoint, stream=True, timeout=180) as r:
        r.raise_for_status()
        tot = int(r.headers.get('content-length', 0))
        down = 0
        with open(out_file, 'wb') as f:
            for chunk in r.iter_content(chunk_size=1024*1024):
                if chunk:
                    f.write(chunk)
                    down += len(chunk)
                    if tot > 0:
                        pct = 90.0 + ((down / tot) * 9.0)
                        info_entry["progress"] = round(pct, 1)

    info_entry["state"] = "COMPLETED"
    info_entry["progress"] = 100.0
    info_entry["status"] = f"Selesai! File 480p ({job_info.get('file_size')}) tersimpan di laptop."
    info_entry["file_path"] = out_file
    info_entry["title"] = file_name
    info_entry["file_size_str"] = format_size(os.path.getsize(out_file))

    if req.auto_queue and os.path.exists(out_file):
        t_id = str(uuid.uuid4())[:8]
        TASKS_DB[t_id] = {
            "task_id": t_id,
            "video_path": out_file,
            "video_name": os.path.basename(out_file),
            "state": "QUEUED",
            "progress": 0,
            "status": "Dalam Antrian Subtitle...",
            "error": None,
            "srt_path": None,
            "srt_xkiro_path": None,
            "srt_lite_path": None,
            "total_cues": 0,
            "config": GLOBAL_CONFIG,
            "created_at": time.time()
        }
        JOB_QUEUE.put(t_id)
        save_queue_state()
def download_pimpbunny_direct(dl_id: str, req: DownloadLinkRequest, info_entry: dict, save_dir: str):
    info_entry["state"] = "DOWNLOADING"
    info_entry["status"] = "Mengekstrak stream direct Pimpbunny (Asli 480p ~160MB)..."
    direct_v, p_title = resolve_kvs_video(req.url)
    if not direct_v:
        raise Exception("Gagal mengekstrak direct URL video dari Pimpbunny.")

    clean_name = re.sub(r'[\\/*?:"<>|]', '', (req.custom_name or p_title or "pimpbunny_video").strip())
    out_file = os.path.join(save_dir, f"{clean_name}.mp4")
    info_entry["status"] = "Mengunduh stream MP4 Pimpbunny langsung ke laptop..."
    
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': req.url
    }
    with requests.get(direct_v, headers=headers, stream=True, allow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        tot = int(r.headers.get('content-length', 0))
        down = 0
        start_t = time.time()
        with open(out_file, 'wb') as f_out:
            for chunk in r.iter_content(chunk_size=2*1024*1024):
                if chunk:
                    f_out.write(chunk)
                    down += len(chunk)
                    elapsed = time.time() - start_t
                    spd = down / elapsed if elapsed > 0 else 0
                    info_entry["speed_str"] = f"{spd/(1024*1024):.1f} MB/s"
                    info_entry["downloaded_str"] = format_size(down)
                    info_entry["total_str"] = format_size(tot)
                    if tot > 0:
                        pct = round((down / tot) * 100, 1)
                        rem = (tot - down) / spd if spd > 0 else 0
                        info_entry["progress"] = pct
                        info_entry["eta_str"] = f"{int(rem)} detik lagi"
                        info_entry["status"] = f"{pct}% • {info_entry['speed_str']} ({info_entry['downloaded_str']}/{info_entry['total_str']})"

    info_entry["state"] = "COMPLETED"
    info_entry["progress"] = 100.0
    info_entry["status"] = "Selesai diunduh ke laptop!"
    info_entry["file_path"] = out_file
    info_entry["title"] = clean_name
    info_entry["file_size_str"] = format_size(os.path.getsize(out_file))

    if req.auto_queue and os.path.exists(out_file):
        t_id = str(uuid.uuid4())[:8]
        TASKS_DB[t_id] = {
            "task_id": t_id,
            "video_path": out_file,
            "video_name": os.path.basename(out_file),
            "state": "QUEUED",
            "progress": 0,
            "status": "Dalam Antrian Subtitle...",
            "error": None,
            "srt_path": None,
            "srt_xkiro_path": None,
            "srt_lite_path": None,
            "total_cues": 0,
            "config": GLOBAL_CONFIG,
            "created_at": time.time()
        }
        JOB_QUEUE.put(t_id)
        save_queue_state()
        info_entry["status"] = "Selesai diunduh & Otomatis Masuk Antrian Subtitle!"

def run_download_job(dl_id: str, req: DownloadLinkRequest):
    info_entry = DOWNLOADS_DB[dl_id]
    info_entry["state"] = "STARTING"
    info_entry["status"] = "Menganalisis link video..."

    save_dir = req.save_dir if req.save_dir and os.path.exists(req.save_dir) else r"D:\translate\gg"

    # Mode unduh: "kaggle" (server Google) atau "local" (laptop). Keduanya menargetkan 480p.
    mode = (req.dl_mode if req.dl_mode in ("kaggle", "local")
            else ("kaggle" if req.use_kaggle_480p else "local"))
    if req.dl_mode in ("kaggle", "local"):
        req.use_kaggle_480p = True

    # Supjav/fc2stream memblokir IP datacenter (Kaggle/Google) -> selalu unduh lokal.
    # (token stream terikat IP pengambil; resolve pun diblokir dari IP cloud)
    force_local_hls = ("supjav.com" in req.url) or ("fc2stream" in req.url)
    if mode == "kaggle" and force_local_hls:
        mode = "local"

    if "supjav.com" in req.url:
        try:
            info_entry["status"] = "Membuka halaman Supjav & mengambil link stream segar..."
            info_entry["progress"] = 1.0
            stream_url, sv_title = resolve_supjav(req.url)
            req.url = stream_url
            if not req.custom_name and sv_title:
                req.custom_name = sv_title
        except Exception as e:
            err_msg = str(e)
            info_entry["state"] = "ERROR"
            info_entry["error"] = err_msg
            info_entry["status"] = f"Gagal Supjav: {err_msg[:120]}"
            return

    # Link HLS mentah (fc2stream dsb) -> downloader lokal khusus segmen PNG
    if ".m3u8" in req.url and (force_local_hls or mode == "local"):
        try:
            download_supjav_hls(dl_id, req, info_entry, save_dir)
        except Exception as e:
            err_msg = str(e)
            hint = ""
            if "403" in err_msg:
                hint = " — link stream kedaluwarsa, buka lagi halaman Supjav untuk link baru"
            info_entry["state"] = "ERROR"
            info_entry["error"] = err_msg
            info_entry["status"] = f"Gagal Download HLS: {err_msg[:120]}{hint}"
            return

    if mode == "kaggle":
        try:
            if "avple" in req.url:
                info_entry["status"] = "Mengekstrak stream Avple untuk diproses di Cloud Kaggle GPU..."
                info_entry["progress"] = 3.0
                manifest_url, av_title = resolve_avple_manifest(req.url)
                if manifest_url:
                    req.url = manifest_url
                    if not req.custom_name and av_title:
                        req.custom_name = av_title
            download_via_kaggle(dl_id, req, info_entry, save_dir)
            return
        except Exception as e:
            err_msg = str(e)
            print(f"[KAGGLE-ERROR] Cloud Kaggle gagal ({err_msg}).")
            info_entry["state"] = "ERROR"
            info_entry["error"] = err_msg
            info_entry["status"] = f"Gagal Cloud Kaggle: {err_msg[:120]}"
            return

    # Penanganan Khusus untuk Avple.tv jika user TIDAK menggunakan Kaggle (fallback lokal)
    if "avple" in req.url:
        try:
            download_avple_video(dl_id, req, info_entry, save_dir)
            return
        except Exception as e:
            err_msg = str(e)
            info_entry["state"] = "ERROR"
            info_entry["error"] = err_msg
            info_entry["status"] = f"Gagal Download Avple: {err_msg[:120]}"
            return

    # Penanganan Khusus untuk Pimpbunny (Direct High-Speed MP4 Stream di Laptop jika user tidak pakai Kaggle)
    if "pimpbunny" in req.url:
        try:
            download_pimpbunny_direct(dl_id, req, info_entry, save_dir)
            return
        except Exception as e:
            err_msg = str(e)
            info_entry["state"] = "ERROR"
            info_entry["error"] = err_msg
            info_entry["status"] = f"Gagal Download Pimpbunny: {err_msg[:120]}"
            return
            # Lanjut ke proses di bawah (fallback)

    # Penanganan Khusus untuk Rou.video (Steganography PNG HLS)
    if "rou.video" in req.url:
        try:
            download_rou_video(dl_id, req, info_entry, save_dir)
            return
        except Exception as e:
            err_msg = str(e)
            info_entry["state"] = "ERROR"
            info_entry["error"] = err_msg
            info_entry["status"] = f"Gagal Download Rou.video: {err_msg[:120]}"
            return
    
    # Progress hook callback
    def progress_hook(d):
        if d['status'] == 'downloading':
            info_entry["state"] = "DOWNLOADING"
            down_b = d.get('downloaded_bytes', 0)
            tot_b = d.get('total_bytes') or d.get('total_bytes_estimate', 0)
            speed = d.get('speed', 0)
            eta = d.get('eta', 0)

            pct = 0.0
            if tot_b and tot_b > 0:
                pct = round((down_b / tot_b) * 100, 1)

            info_entry["progress"] = pct
            info_entry["downloaded_str"] = format_size(down_b)
            info_entry["total_str"] = format_size(tot_b) if tot_b else "Ukuran tidak diketahui"
            
            speed_mb = (speed / (1024 * 1024)) if speed else 0
            info_entry["speed_str"] = f"{speed_mb:.1f} MB/s" if speed_mb > 0 else "Menghubungkan..."
            
            if eta:
                info_entry["eta_str"] = f"{int(eta)} detik lagi"
            else:
                info_entry["eta_str"] = "Menghitung..."

            info_entry["status"] = f"{pct}% • {info_entry['speed_str']} ({info_entry['downloaded_str']}/{info_entry['total_str']})"

        elif d['status'] == 'finished':
            info_entry["progress"] = 100.0
            info_entry["status"] = "Menggabungkan potongan file..."

    # Template nama file
    if req.custom_name and req.custom_name.strip():
        clean_name = re.sub(r'[\\/*?:"<>|]', "", req.custom_name.strip())
        out_template = os.path.join(save_dir, f"{clean_name}.%(ext)s")
    else:
        out_template = os.path.join(save_dir, "%(title)s.%(ext)s")

    ydl_opts = {
        'outtmpl': out_template,
        'progress_hooks': [progress_hook],
        'nocheckcertificate': True,
        'quiet': True,
        'no_warnings': True,
        # Jika diminta 480p, ambil stream 480p langsung dari web (jauh lebih hemat kuota & cepat)
        'format': 'bestvideo[height<=480]+bestaudio/best[height<=480]/best' if req.use_kaggle_480p else 'best',
        'http_headers': {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
            'Referer': 'https://avple.tv/'
        },
        'socket_timeout': 30,
    }


    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            # Ambil info dulu
            meta = ydl.extract_info(req.url.strip(), download=True)
            saved_file = ydl.prepare_filename(meta)

            info_entry["state"] = "COMPLETED"
            info_entry["progress"] = 100.0
            info_entry["status"] = "Selesai diunduh ke laptop!"
            info_entry["file_path"] = saved_file
            info_entry["title"] = meta.get("title", os.path.basename(saved_file))
            if os.path.exists(saved_file):
                info_entry["file_size_str"] = format_size(os.path.getsize(saved_file))

            # Auto queue jika diminta
            if req.auto_queue and os.path.exists(saved_file):
                t_id = str(uuid.uuid4())[:8]
                TASKS_DB[t_id] = {
                    "task_id": t_id,
                    "video_path": saved_file,
                    "video_name": os.path.basename(saved_file),
                    "state": "QUEUED",
                    "progress": 0,
                    "status": "Dalam Antrian Subtitle...",
                    "error": None,
                    "srt_path": None,
                    "srt_xkiro_path": None,
                    "srt_lite_path": None,
                    "total_cues": 0,
                    "config": GLOBAL_CONFIG,
                    "created_at": time.time()
                }
                JOB_QUEUE.put(t_id)
                save_queue_state()
                info_entry["status"] = "Selesai diunduh & Otomatis Masuk Antrian Subtitle!"

    except Exception as e:
        err_msg = str(e)
        info_entry["state"] = "ERROR"
        info_entry["error"] = err_msg
        info_entry["status"] = f"Gagal Download: {err_msg[:120]}"

def start_worker():
    global WORKER_THREAD
    if WORKER_THREAD is None or not WORKER_THREAD.is_alive():
        WORKER_THREAD = threading.Thread(target=worker_loop, daemon=True)
        WORKER_THREAD.start()

load_queue_state_on_startup()
start_worker()

def auto_sync_kaggle_worker():
    while True:
        try:
            # Cek apakah URL yang sedang dipakai masih aktif
            cur = (GLOBAL_CONFIG.kaggle_url or "").strip().rstrip('/')
            cur_alive = False
            if cur:
                try:
                    p = requests.get(f"{cur}/api/job/ping", timeout=3)
                    if p.status_code in [200, 404]:
                        cur_alive = True
                except:
                    cur_alive = False

            # Jika worker yang sekarang aktif masih hidup, pertahankan jangan ditimpa!
            if not cur_alive:
                r = requests.get("https://ntfy.sh/ziranaijav_cloud_worker_2026/json?poll=1", timeout=8)
                if r.status_code == 200:
                    lines = [l for l in r.text.strip().splitlines() if l.strip()]
                    # Cari dari belakang URL yang valid & merespon
                    for line in reversed(lines[-5:]):
                        try:
                            evt = json.loads(line)
                            msg = evt.get("message", "")
                            m_u = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', msg)
                            if m_u:
                                cand = m_u.group(0)
                                # Tes ping kandidat
                                ping = requests.get(f"{cand}/api/job/ping", timeout=3)
                                if ping.status_code in [200, 404]:
                                    GLOBAL_CONFIG.kaggle_url = cand
                                    print(f"\n[AUTO-DETECT] Kaggle Worker URL aktif terverifikasi: {cand}\n")
                                    with open(r"D:\translate\remote_web\kaggle_worker_url.txt", "w", encoding="utf-8") as f_kw:
                                        f_kw.write(cand)
                                    break
                        except: pass
        except: pass
        time.sleep(15)

# threading.Thread(target=auto_sync_kaggle_worker, daemon=True).start()

PUBLIC_TUNNEL_URL = ""
try:
    from pycloudflared import try_cloudflare
    _tun = try_cloudflare(port=8080)
    PUBLIC_TUNNEL_URL = str(_tun)
    with open(r"D:\translate\remote_web\tunnel_url.txt", "w", encoding="utf-8") as f_tun:
        f_tun.write(PUBLIC_TUNNEL_URL)
    print(f"\n============================================================")
    print(f"🌐 PUBLIC WEB REMOTE TUNNEL (HP): {PUBLIC_TUNNEL_URL}")
    print(f"============================================================\n")
except Exception as e:
    print(f"[TUNNEL NOTICE]: {e}")

# ==========================================
# REST API ENDPOINTS
# ==========================================
@app.get("/api/config")
def get_config():
    return GLOBAL_CONFIG.dict()

@app.post("/api/config")
def update_config(cfg: GlobalSettings):
    global GLOBAL_CONFIG
    GLOBAL_CONFIG = cfg
    return {"message": "Pengaturan berhasil disimpan", "config": GLOBAL_CONFIG.dict()}

@app.post("/api/register_kaggle")
def register_kaggle(req: RegisterKaggleRequest):
    global GLOBAL_CONFIG
    clean_url = req.kaggle_url.strip().rstrip('/')
    GLOBAL_CONFIG.kaggle_url = clean_url
    try:
        with open(r"D:\translate\remote_web\kaggle_worker_url.txt", "w", encoding="utf-8") as f_kw:
            f_kw.write(clean_url)
    except: pass
    print(f"\n[AUTO-CONNECT] Kaggle Worker otomatis terhubung: {clean_url}\n")
    return {"message": "Kaggle worker berhasil terdaftar otomatis!", "kaggle_url": clean_url}

@app.get("/api/browse")
def api_browse(path: Optional[str] = None):
    if not path or not os.path.exists(path):
        path = r"D:\translate\gg"
        if not os.path.exists(path):
            path = r"D:\translate"
            if not os.path.exists(path):
                path = r"C:\\"
    
    path = os.path.abspath(path)
    parent = str(Path(path).parent) if Path(path).parent != Path(path) else None

    VIDEO_EXTS = {".mp4", ".mkv", ".avi", ".mov", ".flv", ".webm", ".ts", ".m4v"}
    folders = []
    videos = []
    
    try:
        entries = sorted(os.scandir(path), key=lambda e: (not e.is_dir(), e.name.lower()))
        for entry in entries:
            try:
                if entry.is_dir():
                    folders.append({
                        "name": entry.name,
                        "path": entry.path
                    })
                elif entry.is_file():
                    ext = os.path.splitext(entry.name)[1].lower()
                    if ext in VIDEO_EXTS:
                        base = os.path.splitext(entry.path)[0]
                        srt_main = f"{base}.srt"
                        srt_xkiro = f"{base}_xkiro.srt"
                        srt_lite = f"{base}_literouter.srt"

                        has_main = os.path.exists(srt_main)
                        has_xkiro = os.path.exists(srt_xkiro)
                        has_lite = os.path.exists(srt_lite)

                        srt_mandarin = f"{base}_mandarin.srt"
                        srt_gemini = f"{base}_gemini.srt"
                        has_mandarin = os.path.exists(srt_mandarin)
                        has_gemini = os.path.exists(srt_gemini)

                        size_mb = round(entry.stat().st_size / (1024 * 1024), 1)
                        videos.append({
                            "name": entry.name,
                            "path": entry.path,
                            "size_mb": size_mb,
                            "has_srt": has_main or has_xkiro or has_lite or has_gemini,
                            "srt_path": srt_gemini if has_gemini else (srt_main if has_main else (srt_xkiro if has_xkiro else srt_lite)),
                            "srt_xkiro_path": srt_xkiro if has_xkiro else None,
                            "srt_lite_path": srt_lite if has_lite else None,
                            "srt_gemini_path": srt_gemini if has_gemini else None,
                            "has_mandarin": has_mandarin,
                            "srt_mandarin_path": srt_mandarin if has_mandarin else None
                        })
            except Exception:
                continue
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {
        "current_path": path,
        "parent_path": parent,
        "folders": folders,
        "videos": videos
    }

class TranslateSrtRequest(BaseModel):
    srt_path: str
    provider: Optional[str] = "aichixia"

def parse_srt_file(srt_path: str) -> list:
    with open(srt_path, 'r', encoding='utf-8', errors='ignore') as f:
        content = f.read()

    blocks = re.split(r'\n\s*\n', content.strip())
    cues = []
    for b in blocks:
        lines = b.strip().splitlines()
        if len(lines) >= 2:
            num = lines[0].strip()
            ts_idx = -1
            for idx, l in enumerate(lines):
                if '-->' in l:
                    ts_idx = idx
                    break
            if ts_idx != -1:
                ts = lines[ts_idx].strip()
                parts = ts.split('-->')
                start_sec, end_sec = 0.0, 0.0
                if len(parts) == 2:
                    try:
                        def parse_t(s):
                            s = s.strip().replace(',', '.')
                            p = s.split(':')
                            return float(p[0])*3600 + float(p[1])*60 + float(p[2])
                        start_sec = parse_t(parts[0])
                        end_sec = parse_t(parts[1])
                    except: pass
                text = " ".join([l.strip() for l in lines[ts_idx+1:] if l.strip()])
                cues.append({
                    'index': len(cues) + 1,
                    'start': start_sec,
                    'end': end_sec,
                    'text': text
                })
    return cues

@app.post("/api/translate_srt")
def api_translate_srt(req: TranslateSrtRequest):
    srt_path = req.srt_path.strip()
    if not os.path.exists(srt_path):
        raise HTTPException(status_code=404, detail=f"File SRT tidak ditemukan: {srt_path}")
    
    cues = parse_srt_file(srt_path)
    if not cues:
        raise HTTPException(status_code=400, detail="File SRT kosong atau format tidak valid")

    task_id = f"retrans_{uuid.uuid4().hex[:8]}"
    folder = os.path.dirname(srt_path)
    base_name = os.path.basename(srt_path)
    clean_base = re.sub(r'(_mandarin|_xkiro|_literouter|_gemini)?\.srt$', '', base_name)
    provider = req.provider or "aichixia"

    TASKS_DB[task_id] = {
        "task_id": task_id,
        "video_path": srt_path,
        "video_name": f"[⚡ Translate SRT] {clean_base}",
        "state": "PROCESSING",
        "progress": 10,
        "status": f"Menerjemahkan {len(cues)} baris dialog dengan {provider} (tanpa STT)...",
        "created_at": time.time(),
        "srt_path": None,
        "is_srt_only": True
    }
    save_queue_state()

    def run_retranslate():
        try:
            cfg = GlobalSettings()
            cfg.translation_mode = provider
            translated_cues = translate_cues_with_provider(
                cues, provider, cfg, task_id, 15, 95
            )
            final_txt = build_srt_text(translated_cues)
            
            gemini_out = os.path.join(folder, f"{clean_base}_gemini.srt")
            main_out = os.path.join(folder, f"{clean_base}.srt")
            
            with open(gemini_out, "w", encoding="utf-8") as f:
                f.write(final_txt)
            with open(main_out, "w", encoding="utf-8") as f:
                f.write(final_txt)

            TASKS_DB[task_id]["state"] = "COMPLETED"
            TASKS_DB[task_id]["progress"] = 100
            TASKS_DB[task_id]["status"] = f"Translasi selesai! Disimpan ke {clean_base}.srt & _gemini.srt"
            TASKS_DB[task_id]["srt_path"] = main_out
            save_queue_state()
        except Exception as ex:
            TASKS_DB[task_id]["state"] = "ERROR"
            TASKS_DB[task_id]["status"] = f"Error translasi SRT: {ex}"
            save_queue_state()

    threading.Thread(target=run_retranslate, daemon=True).start()
    return {"status": "ok", "task_id": task_id, "message": f"Translasi {len(cues)} baris SRT berhasil dimulai di antrean!"}

@app.post("/api/shutdown_kaggle")
def api_shutdown_kaggle():
    target_url = (GLOBAL_CONFIG.kaggle_url or "").strip().rstrip('/')
    if not target_url:
        raise HTTPException(status_code=400, detail="URL Kaggle Worker belum terdaftar")
    try:
        r = requests.post(f"{target_url}/api/shutdown", timeout=5)
        return {"status": "ok", "message": "Perintah matikan Kaggle Worker berhasil dikirim! GPU Kaggle segera mati."}
    except Exception as e:
        return {"status": "error", "message": f"Gagal menghubungi worker (kemungkinan sudah mati): {e}"}

@app.post("/api/queue/add")
def api_queue_add(req: EnqueueItemRequest):
    start_worker()
    created_ids = []
    config_to_use = req.settings if req.settings else GLOBAL_CONFIG
    
    for v_path in req.video_paths:
        task_id = str(uuid.uuid4())[:8]
        v_name = os.path.basename(v_path)
        TASKS_DB[task_id] = {
            "task_id": task_id,
            "video_path": v_path,
            "video_name": v_name,
            "state": "QUEUED",
            "progress": 0,
            "status": "Dalam Antrian...",
            "error": None,
            "srt_path": None,
            "srt_xkiro_path": None,
            "srt_lite_path": None,
            "total_cues": 0,
            "config": config_to_use,
            "created_at": time.time()
        }
        JOB_QUEUE.put(task_id)
        created_ids.append(task_id)

    save_queue_state()
    return {"message": f"{len(created_ids)} video berhasil ditambahkan ke antrian", "task_ids": created_ids}

@app.get("/api/queue/status")
def api_queue_status():
    all_tasks = sorted(TASKS_DB.values(), key=lambda t: t.get("created_at", 0), reverse=True)
    
    counts = {
        "processing": sum(1 for t in all_tasks if t.get("state") == "PROCESSING"),
        "queued": sum(1 for t in all_tasks if t.get("state") == "QUEUED"),
        "completed": sum(1 for t in all_tasks if t.get("state") == "COMPLETED"),
        "error": sum(1 for t in all_tasks if t.get("state") == "ERROR")
    }

    items = []
    for t in all_tasks:
        items.append({
            "task_id": t.get("task_id"),
            "video_name": t.get("video_name"),
            "state": t.get("state"),
            "progress": t.get("progress", 0),
            "status": t.get("status"),
            "error": t.get("error"),
            "srt_path": t.get("srt_path"),
            "srt_xkiro_path": t.get("srt_xkiro_path"),
            "srt_lite_path": t.get("srt_lite_path"),
            "total_cues": t.get("total_cues", 0),
            "model": t.get("config", GLOBAL_CONFIG).speech_model if hasattr(t.get("config", GLOBAL_CONFIG), "speech_model") else "universal-3-5-pro",
            "mode": t.get("config", GLOBAL_CONFIG).translation_mode if hasattr(t.get("config", GLOBAL_CONFIG), "translation_mode") else "both",
            "lang": t.get("config", GLOBAL_CONFIG).language_code if hasattr(t.get("config", GLOBAL_CONFIG), "language_code") else "auto"
        })

    return {
        "counts": counts,
        "tasks": items
    }

@app.post("/api/queue/clear")
def api_clear_queue():
    to_del = [k for k, v in TASKS_DB.items() if v.get("state") in ["COMPLETED", "ERROR"]]
    for k in to_del:
        del TASKS_DB[k]
    save_queue_state()
    return {"message": f"{len(to_del)} antrian selesai/error berhasil dibersihkan"}

@app.delete("/api/queue/{task_id}")
def api_delete_queue_task(task_id: str):
    if task_id in TASKS_DB:
        del TASKS_DB[task_id]
        save_queue_state()
        return {"message": "Tugas antrian berhasil dihapus"}
    raise HTTPException(status_code=404, detail="Tugas tidak ditemukan")

@app.post("/api/queue/{task_id}/cancel")
def api_cancel_task(task_id: str):
    task = TASKS_DB.get(task_id)
    if not task:
        raise HTTPException(status_code=404, detail="Tugas tidak ditemukan")
    if task.get("state") in ["COMPLETED", "ERROR", "CANCELLED"]:
        return {"message": f"Tugas sudah {task.get('state')}, tidak perlu dibatalkan"}
    task["state"] = "CANCELLED"
    task["status"] = "Dibatalkan oleh pengguna"
    save_queue_state()
    return {"message": f"Tugas {task_id} berhasil dibatalkan"}

# Endpoint Download dari Link
@app.post("/api/download_link")
def api_download_link(req: DownloadLinkRequest, background_tasks: BackgroundTasks):
    if not req.url or not req.url.strip().startswith("http"):
        raise HTTPException(status_code=400, detail="URL tidak valid")
    
    dl_id = f"dl_{str(uuid.uuid4())[:8]}"
    DOWNLOADS_DB[dl_id] = {
        "dl_id": dl_id,
        "url": req.url.strip(),
        "state": "QUEUED",
        "progress": 0.0,
        "speed_str": "0 MB/s",
        "downloaded_str": "0 MB",
        "total_str": "Menghitung...",
        "eta_str": "-",
        "status": "Menyiapkan download...",
        "error": None,
        "file_path": None,
        "title": "Mengambil info video...",
        "created_at": time.time()
    }
    background_tasks.add_task(run_download_job, dl_id, req)
    return {"dl_id": dl_id, "message": "Proses download dimulai"}

@app.get("/api/download_status/{dl_id}")
def api_download_status(dl_id: str):
    if dl_id not in DOWNLOADS_DB:
        raise HTTPException(status_code=404, detail="Task download tidak ditemukan")
    return DOWNLOADS_DB[dl_id]

@app.get("/api/downloads")
def api_downloads_list():
    all_dls = sorted(DOWNLOADS_DB.values(), key=lambda d: d.get("created_at", 0), reverse=True)
    return all_dls

@app.delete("/api/downloads/{dl_id}")
def api_delete_download(dl_id: str):
    if dl_id in DOWNLOADS_DB:
        del DOWNLOADS_DB[dl_id]
    return {"message": "Berhasil dihapus"}

@app.post("/api/downloads/clear")
def api_clear_downloads():
    to_del = [k for k, v in DOWNLOADS_DB.items() if v.get("state") in ["COMPLETED", "ERROR"]]
    for k in to_del:
        del DOWNLOADS_DB[k]
    return {"message": f"{len(to_del)} download dibersihkan"}

@app.get("/api/download_srt")
def api_download(path: str = Query(...)):
    if not os.path.exists(path):
        raise HTTPException(status_code=404, detail="File SRT belum ada")
    filename = os.path.basename(path)
    return FileResponse(path, filename=filename, media_type="application/x-subrip")

@app.get("/", response_class=HTMLResponse)
def index_page():
    html_content = """<!DOCTYPE html>
<html lang="id">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0, maximum-scale=1.0, user-scalable=no">
    <title>Laptop Video & Subtitle Controller 🎬</title>
    <script src="/static/tailwind.js"></script>
    <link rel="stylesheet" href="/static/fontawesome.min.css">
    <style>
        body { -webkit-tap-highlight-color: transparent; }
        .touch-card:active { transform: scale(0.985); }
        .no-scrollbar::-webkit-scrollbar { display: none; }
        .no-scrollbar { -ms-overflow-style: none; scrollbar-width: none; }
    </style>
</head>
<body class="bg-slate-950 text-slate-100 min-h-screen pb-32 font-sans select-none">

    <!-- Top Sticky Header -->
    <header class="sticky top-0 z-30 bg-slate-900/95 backdrop-blur border-b border-slate-800 px-4 py-3 flex items-center justify-between shadow-xl">
        <div class="flex items-center gap-2.5">
            <div class="w-9 h-9 rounded-xl bg-gradient-to-tr from-blue-500 via-indigo-500 to-purple-500 flex items-center justify-center shadow-lg shadow-indigo-500/20">
                <i class="fa-solid fa-cloud-arrow-down text-white text-sm"></i>
            </div>
            <div>
                <h1 class="text-sm font-bold leading-tight">Laptop Remote Hub</h1>
                <p class="text-[11px] text-slate-400">Download Link & Subtitle Queue</p>
            </div>
        </div>

        <div class="flex items-center gap-1.5">
            <button onclick="openSettingsModal()" class="w-9 h-9 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 active:bg-slate-700">
                <i class="fa-solid fa-sliders text-xs"></i>
            </button>
            <button onclick="loadFolder(currentPath)" class="w-9 h-9 rounded-xl bg-slate-800 border border-slate-700 flex items-center justify-center text-slate-300 active:bg-slate-700">
                <i class="fa-solid fa-rotate text-xs"></i>
            </button>
        </div>
    </header>

    <!-- Main Navigation Tabs -->
    <div class="max-w-lg mx-auto px-3 pt-3">
        <div class="grid grid-cols-2 gap-2 bg-slate-900 p-1 rounded-2xl border border-slate-800">
            <button id="tabBtnBrowse" onclick="switchTab('browse')" class="py-2.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 bg-indigo-600 text-white shadow">
                <i class="fa-solid fa-film"></i>
                <span>Video di Laptop</span>
            </button>
            <button id="tabBtnDownload" onclick="switchTab('download')" class="py-2.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 text-slate-400 hover:text-slate-200">
                <i class="fa-solid fa-link"></i>
                <span>Sedot Link Video</span>
            </button>
        </div>
    </div>

    <main class="p-3 max-w-lg mx-auto space-y-3.5">

        <!-- TAB 1: BROWSE & SUBTITLE QUEUE -->
        <div id="viewBrowse" class="space-y-3.5">

            <!-- Active Queue Progress Summary Bar -->
            <div onclick="openQueueDrawer()" class="bg-gradient-to-r from-slate-900 to-indigo-950/70 border border-indigo-500/40 rounded-2xl p-3.5 shadow-xl shadow-indigo-500/10 cursor-pointer active:scale-[0.99] transition">
                <div class="flex items-center justify-between mb-2">
                    <div class="flex items-center gap-2">
                        <span class="relative flex h-2.5 w-2.5">
                            <span id="headerPingRing" class="animate-ping absolute inline-flex h-full w-full rounded-full bg-emerald-400 opacity-75 hidden"></span>
                            <span id="headerPingDot" class="relative inline-flex rounded-full h-2.5 w-2.5 bg-slate-500"></span>
                        </span>
                        <span id="headerQueueTitle" class="text-xs font-bold text-slate-200">Antrian Kosong</span>
                    </div>
                    <div class="flex items-center gap-1.5 text-[11px] font-semibold">
                        <span id="badgeActive" class="px-2 py-0.5 rounded-full bg-indigo-500/20 text-indigo-300 border border-indigo-500/30">0 Jalan</span>
                        <span id="badgeQueued" class="px-2 py-0.5 rounded-full bg-amber-500/20 text-amber-300 border border-amber-500/30">0 Antri</span>
                        <span class="text-slate-400"><i class="fa-solid fa-chevron-right text-[10px]"></i></span>
                    </div>
                </div>
                
                <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                    <div id="headerProgressBar" class="bg-gradient-to-r from-indigo-500 to-emerald-400 h-1.5 rounded-full transition-all duration-300 w-0"></div>
                </div>
                <p id="headerStatusText" class="text-[10px] text-slate-400 mt-1.5 truncate">Siap menerima antrian video...</p>
            </div>

            <!-- Path & Parent Navigation -->
            <div class="bg-slate-900 border border-slate-800 rounded-xl p-2.5 flex items-center gap-2 text-xs">
                <i class="fa-regular fa-folder-open text-indigo-400 text-sm"></i>
                <span id="pathDisplay" class="font-mono text-slate-300 truncate flex-1 text-[11px]">Memuat...</span>
                <button id="btnParentFolder" onclick="goParent()" class="hidden bg-slate-800 border border-slate-700 px-2.5 py-1 rounded-lg text-slate-200 text-xs active:bg-slate-700 flex items-center gap-1">
                    <i class="fa-solid fa-arrow-up text-[10px]"></i> Naik
                </button>
            </div>

            <!-- Quick Folder Shortcuts -->
            <div class="flex gap-2 overflow-x-auto pb-1 text-xs no-scrollbar">
                <button onclick="loadFolder('D:\\\\translate\\\\gg')" class="whitespace-nowrap px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 active:bg-indigo-600">
                    📁 gg (Root)
                </button>
                <button onclick="loadFolder('D:\\\\translate\\\\gg\\\\AiXiAshley\\\\FFBatch')" class="whitespace-nowrap px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 active:bg-indigo-600">
                    📁 Ai Xi Ashley
                </button>
                <button onclick="loadFolder('D:\\\\translate\\\\gg\\\\iris')" class="whitespace-nowrap px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 active:bg-indigo-600">
                    📁 Iris
                </button>
                <button onclick="loadFolder('D:\\\\translate\\\\gg\\\\Brazzers\\\\FFBatch')" class="whitespace-nowrap px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 active:bg-indigo-600">
                    📁 Brazzers
                </button>
                <button onclick="loadFolder('D:\\\\translate\\\\gg\\\\nana_taipei')" class="whitespace-nowrap px-3 py-1.5 rounded-full bg-slate-800 border border-slate-700 text-slate-300 active:bg-indigo-600">
                    📁 Nana Taipei
                </button>
            </div>

            <!-- Subfolders -->
            <div id="subfoldersSection" class="hidden">
                <p class="text-[11px] font-semibold text-slate-400 mb-1.5 uppercase tracking-wider">Subfolder</p>
                <div id="subfoldersList" class="grid grid-cols-2 gap-2 text-xs"></div>
            </div>

            <!-- Video Selection Header -->
            <div class="flex items-center justify-between pt-1">
                <div class="flex items-center gap-2">
                    <span class="text-xs font-bold uppercase tracking-wider text-slate-300">Daftar Video</span>
                    <span id="videoCountBadge" class="bg-indigo-900/60 text-indigo-300 text-[10px] px-2 py-0.5 rounded-full border border-indigo-700/50">0 video</span>
                </div>
                
                <div class="flex items-center gap-1.5">
                    <button onclick="toggleSelectAll()" class="text-[11px] bg-slate-800 text-slate-300 px-2.5 py-1 rounded-lg border border-slate-700 active:bg-slate-700">
                        <span id="btnSelectAllText">Pilih Semua</span>
                    </button>
                </div>
            </div>

            <!-- Video List Container -->
            <div id="videosContainer" class="space-y-2">
                <div class="text-center py-12 text-slate-500 text-sm">
                    <i class="fa-solid fa-spinner fa-spin text-2xl mb-2 text-indigo-400"></i>
                    <p>Membaca video di laptop...</p>
                </div>
            </div>

        </div>

        <!-- TAB 2: DOWNLOAD VIDEO DARI LINK -->
        <div id="viewDownload" class="hidden space-y-4">
            
            <div class="bg-slate-900 border border-slate-800 rounded-3xl p-4 shadow-xl space-y-3.5">
                <div class="flex items-center gap-2">
                    <div class="w-8 h-8 rounded-xl bg-purple-500/20 text-purple-400 flex items-center justify-center">
                        <i class="fa-solid fa-cloud-arrow-down text-sm"></i>
                    </div>
                    <div>
                        <h3 class="text-xs font-bold text-slate-100">Sedot Video dari Link Web</h3>
                        <p class="text-[10px] text-slate-400">Direct Stream Copy (Tanpa Render / CPU Dingin)</p>
                    </div>
                </div>

                <!-- Input URL -->
                <div>
                    <label class="block text-[11px] font-semibold text-slate-300 mb-1">Link Video / M3U8 Streaming:</label>
                    <textarea id="dlUrlInput" rows="2" placeholder="Tempel link https://... playlist.m3u8 atau link situs video di sini" class="w-full bg-slate-950 border border-slate-800 rounded-xl p-2.5 text-xs text-slate-200 focus:outline-none focus:border-indigo-500 font-mono"></textarea>
                </div>

                <!-- Custom Filename -->
                <div>
                    <label class="block text-[11px] font-semibold text-slate-300 mb-1">Nama File (Opsional):</label>
                    <input id="dlCustomName" type="text" placeholder="Kosongkan jika ingin otomatis sesuai judul video" class="w-full bg-slate-950 border border-slate-800 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
                </div>

                <!-- Save Folder Display -->
                <div class="bg-slate-950 p-2.5 rounded-xl border border-slate-800 text-[11px] flex items-center justify-between">
                    <span class="text-slate-400">Tersimpan ke folder:</span>
                    <span id="dlFolderDisplay" class="font-mono text-indigo-300 truncate max-w-[200px]">D:\\translate\\gg</span>
                </div>

                <!-- Opsi Sumber: Cloud Kaggle vs Lokal -->
                <div class="space-y-2">
                    <div class="grid grid-cols-2 gap-2">
                        <label id="dlModeKaggleWrap" class="dl-mode-card bg-purple-950/40 p-3 rounded-xl border border-purple-800/60 cursor-pointer space-y-1" onclick="setDlMode('kaggle')">
                            <input id="dlModeKaggle" name="dlMode" type="radio" checked class="hidden">
                            <span class="text-xs font-bold text-purple-300 flex items-center gap-1.5">
                                <i class="fa-solid fa-cloud text-[11px]"></i> Cloud Kaggle
                            </span>
                            <span class="text-[10px] text-purple-300/80 leading-relaxed block">
                                Download &amp; kompres 480p di server Google (hemat kuota). File ringan ditarik ke laptop.
                            </span>
                        </label>
                        <label id="dlModeLocalWrap" class="dl-mode-card bg-slate-950 p-3 rounded-xl border border-slate-800 cursor-pointer space-y-1" onclick="setDlMode('local')">
                            <input id="dlModeLocal" name="dlMode" type="radio" class="hidden">
                            <span class="text-xs font-bold text-slate-300 flex items-center gap-1.5">
                                <i class="fa-solid fa-laptop text-[11px]"></i> Lokal (Laptop)
                            </span>
                            <span class="text-[10px] text-slate-400 leading-relaxed block">
                                Video diunduh langsung ke laptop, lalu <b class="text-indigo-300">audionya dikirim ke Kaggle</b> untuk transkripsi.
                            </span>
                        </label>
                    </div>

                    <p id="dlModeNote" class="text-[10px] text-purple-300/80 leading-relaxed bg-purple-950/30 border border-purple-900/50 p-2 rounded-lg"></p>

                    <!-- Quick Kaggle Worker URL Input (hanya untuk mode Kaggle) -->
                    <div id="kaggleWorkerBox" class="bg-slate-950 p-3 rounded-xl border border-purple-900/60 space-y-1.5">
                        <div class="flex items-center justify-between text-[11px]">
                            <span id="dlWorkerBoxLabel" class="text-purple-200 font-medium">Cloudflare Tunnel Worker URL:</span>
                                <a href="https://www.kaggle.com/code/ziranaijav/cloud-video-compressor" target="_blank" class="text-indigo-400 hover:text-indigo-300 underline font-semibold flex items-center gap-1">
                                    <i class="fa-solid fa-arrow-up-right-from-square text-[10px]"></i> Buka Notebook Kaggle
                                </a>
                            </div>
                            <div class="flex flex-wrap gap-2">
                                <input id="dlKaggleUrlDirect" type="url" placeholder="https://xxx.trycloudflare.com" class="flex-1 min-w-[140px] bg-slate-950 border border-purple-700/60 rounded-lg px-2.5 py-1.5 text-xs text-purple-200 font-mono focus:outline-none focus:border-purple-400">
                                <button type="button" onclick="saveDirectKaggleUrl()" class="bg-purple-700 hover:bg-purple-600 text-white text-[11px] font-semibold px-3 py-1.5 rounded-lg whitespace-nowrap">
                                    Simpan URL
                                </button>
                                <button type="button" onclick="stopKaggleWorkerNow()" class="bg-rose-700 hover:bg-rose-600 text-white text-[11px] font-semibold px-2.5 py-1.5 rounded-lg whitespace-nowrap flex items-center gap-1 shadow" title="Matikan GPU Kaggle sekarang agar kuota tidak terbuang">
                                    <i class="fa-solid fa-power-off text-[10px]"></i> Matikan Kaggle
                                </button>
                            </div>
                            <p class="text-[9px] text-purple-400/90 flex items-center gap-1">
                                <i class="fa-solid fa-shield-halved text-[9px] text-emerald-400"></i>
                                <span>Auto-Shutdown Aktif: Kaggle otomatis mati jika menganggur 5 menit tanpa request.</span>
                            </p>
                        </div>

                    <label class="flex items-center gap-2 text-xs text-slate-300 bg-slate-950 p-2.5 rounded-xl border border-slate-800 cursor-pointer">
                        <input id="dlAutoQueue" type="checkbox" checked class="w-4 h-4 accent-indigo-500 rounded">
                        <span>Otomatis Masuk Antrian Subtitle setelah download</span>
                    </label>
                </div>

                <!-- Start Button -->
                <button onclick="startDownloadLink()" class="w-full bg-gradient-to-r from-purple-600 to-indigo-600 active:from-purple-700 active:to-indigo-700 text-white font-bold py-3 px-4 rounded-xl text-xs flex items-center justify-center gap-2 shadow-lg shadow-purple-600/30">
                    <i class="fa-solid fa-download text-xs"></i>
                    <span>Mulai Download ke Laptop</span>
                </button>
            </div>

            <!-- Live Downloads Monitor Container -->
            <div>
                <div class="flex items-center justify-between mb-2">
                    <h4 class="text-xs font-bold uppercase tracking-wider text-slate-400">Aktivitas Download Realtime</h4>
                    <button onclick="clearDownloads()" class="text-[10px] text-slate-400 hover:text-slate-200 bg-slate-800 px-2 py-1 rounded-lg flex items-center gap-1 active:bg-slate-700">
                        <i class="fa-solid fa-broom text-[9px]"></i> Bersihkan Selesai
                    </button>
                </div>
                <div id="downloadsMonitorList" class="space-y-2.5">
                    <p class="text-center text-xs text-slate-600 py-6">Belum ada proses download.</p>
                </div>
            </div>

        </div>

    </main>

    <!-- Floating Batch Action Bar -->
    <div id="batchActionBar" class="fixed bottom-4 left-3 right-3 max-w-lg mx-auto z-40 bg-indigo-600 text-white p-3 rounded-2xl shadow-2xl flex items-center justify-between hidden transition-all transform duration-200">
        <div class="flex items-center gap-2">
            <span class="w-6 h-6 rounded-full bg-white/20 flex items-center justify-center text-xs font-bold" id="selectedCountText">0</span>
            <span class="text-xs font-semibold">Video Terpilih</span>
        </div>
        <button onclick="queueSelectedVideos()" class="bg-white text-indigo-900 font-bold px-4 py-2 rounded-xl text-xs flex items-center gap-1.5 active:bg-slate-100 shadow">
            <i class="fa-solid fa-plus text-xs"></i>
            <span>+ Masukkan Antrian</span>
        </button>
    </div>

    <!-- Modal Pengaturan Detail (Settings) -->
    <div id="settingsModal" class="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm hidden flex items-end sm:items-center justify-center p-0 sm:p-4">
        <div class="bg-slate-900 border-t sm:border border-slate-700 w-full max-w-md max-h-[85vh] overflow-y-auto rounded-t-3xl sm:rounded-2xl p-5 space-y-4 shadow-2xl no-scrollbar">
            
            <div class="flex items-center justify-between pb-2 border-b border-slate-800">
                <div class="flex items-center gap-2">
                    <div class="w-8 h-8 rounded-lg bg-indigo-600/30 text-indigo-400 flex items-center justify-center">
                        <i class="fa-solid fa-sliders text-sm"></i>
                    </div>
                    <div>
                        <h3 class="text-sm font-bold text-slate-100">Pengaturan Terjemahan</h3>
                        <p class="text-[10px] text-slate-400">Pilih xKiro atau LiteRouter</p>
                    </div>
                </div>
                <button onclick="closeSettingsModal()" class="w-7 h-7 rounded-full bg-slate-800 text-slate-400 flex items-center justify-center active:bg-slate-700">
                    <i class="fa-solid fa-xmark text-xs"></i>
                </button>
            </div>

            <!-- Engine Selection -->
            <div class="bg-indigo-950/40 p-3.5 rounded-2xl border border-indigo-700/50 space-y-2.5">
                <div class="flex items-center justify-between">
                    <div>
                        <span class="text-xs font-bold text-indigo-200 block">Pilihan Engine Terjemahan</span>
                        <span class="text-[10px] text-slate-400">Mau salah satu atau dua-duanya:</span>
                    </div>
                    <input id="cfgTranslateToId" type="checkbox" checked class="w-4 h-4 accent-indigo-500 rounded">
                </div>

                <select id="cfgTranslationMode" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs font-semibold text-slate-100 focus:outline-none focus:border-indigo-500">
                    <option value="aichixia" selected>✨ Aichixia (Qwen 3.8 27B - Anti-Sensor & Sutradara Ranah Dewasa)</option>
                    <option value="both">⚡ Dua-duanya Sekaligus (Output: 2 SRT Indo: xKiro + LiteRouter)</option>
                    <option value="xkiro">🤖 Hanya xKiro (Qwen 3.8 Max)</option>
                    <option value="literouter">🌐 Hanya LiteRouter (DeepSeek V3.2)</option>
                </select>
            </div>

            <!-- Speech Model -->
            <div>
                <label class="block text-xs font-semibold text-slate-300 mb-1">Model Transkripsi (Speech-to-Text):</label>
                <select id="cfgSpeechModel" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
                    <option value="fireredasr2">🔥 1. FireRedASR2S (Xiaomi SOTA Mandarin 2.89% CER - Kaggle GPU)</option>
                    <option value="universal-3-5-pro">⚡ 2. Universal-3.5 Pro (AssemblyAI Flagship - Kalimat & Grammar Rapi)</option>
                    <option value="whisperx">🎙️ 3. WhisperX Engine (Word-Level Timing Presisi & Peka Bisikan/Desahan)</option>
                    <option value="universal-3-pro">🌟 Universal-3 Pro (AssemblyAI Versi Stabil)</option>
                    <option value="universal-2">🏛️ Universal-2 (AssemblyAI Generasi Sebelumnya)</option>
                </select>
            </div>

            <!-- Language Audio -->
            <div>
                <label class="block text-xs font-semibold text-slate-300 mb-1">Bahasa Audio Asli:</label>
                <select id="cfgLangCode" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
                    <option value="auto">🌐 Deteksi Otomatis (Auto-Detect)</option>
                    <option value="zh">🇨🇳 Mandarin / Chinese</option>
                    <option value="en">🇺🇸 English</option>
                    <option value="ja">🇯🇵 Japanese</option>
                    <option value="ko">🇰🇷 Korean</option>
                    <option value="es">🇪🇸 Spanish</option>
                </select>
            </div>

            <!-- Word Boost -->
            <div>
                <label class="block text-xs font-semibold text-slate-300 mb-1">Word Boost (Kata Kunci Khusus / Nama Artis):</label>
                <input id="cfgWordBoost" type="text" placeholder="Contoh: Nana, Taipei, Brazzers, Iris" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
            </div>

            <!-- Timing Segmentation -->
            <div class="grid grid-cols-2 gap-2">
                <div>
                    <span class="text-[10px] text-slate-400 block mb-0.5">Jeda Baris (Detik):</span>
                    <input id="cfgPauseSec" type="number" step="0.1" value="0.6" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
                </div>
                <div>
                    <span class="text-[10px] text-slate-400 block mb-0.5">Maks Karakter:</span>
                    <input id="cfgMaxChar" type="number" step="1" value="42" class="w-full bg-slate-800 border border-slate-700 rounded-xl px-3 py-2 text-xs text-slate-200 focus:outline-none focus:border-indigo-500">
                </div>
            </div>

            <!-- Kaggle Cloud Worker URL (Opsional) -->
            <div class="border-t border-slate-800 pt-3">
                <div class="flex items-center justify-between mb-1">
                    <label class="block text-xs font-semibold text-purple-300">
                        <i class="fa-solid fa-cloud text-purple-400"></i> Kaggle Cloud Worker URL:
                    </label>
                    <span class="text-[9px] text-slate-400 bg-slate-800 px-1.5 py-0.5 rounded">Opsional / Boleh Kosong</span>
                </div>
                <input id="cfgKaggleUrl" type="text" placeholder="Kosongkan saja (Otomatis pakai GPU Laptop)" class="w-full bg-slate-800 border border-purple-800/80 rounded-xl px-3 py-2 text-xs text-purple-200 focus:outline-none focus:border-purple-400 font-mono">
                <p class="text-[10px] text-emerald-400/90 mt-1"><i class="fa-solid fa-circle-check text-[9px]"></i> <strong>Biarkan kosong:</strong> Sistem otomatis memakai Chip Intel QuickSync GPU laptop (beban CPU 0%, super adem, dan otomatis pangkas ukuran video ke 480p).</p>
            </div>

            <!-- Save Settings Button -->
            <div class="pt-2">
                <button onclick="saveSettings()" class="w-full bg-indigo-600 active:bg-indigo-700 text-white font-bold py-3 px-4 rounded-xl text-xs flex items-center justify-center gap-2 shadow-lg shadow-indigo-600/30">
                    <i class="fa-solid fa-check text-xs"></i>
                    <span>Simpan Pengaturan</span>
                </button>
            </div>

        </div>
    </div>

    <!-- Drawer Daftar Antrian Tugas (Queue Drawer) -->
    <div id="queueDrawer" class="fixed inset-0 z-50 bg-black/75 backdrop-blur-sm hidden flex items-end sm:items-center justify-center p-0 sm:p-4">
        <div class="bg-slate-900 border-t sm:border border-slate-700 w-full max-w-md max-h-[85vh] flex flex-col rounded-t-3xl sm:rounded-2xl shadow-2xl">
            
            <div class="p-4 border-b border-slate-800 flex items-center justify-between">
                <div class="flex items-center gap-2">
                    <i class="fa-solid fa-layer-group text-indigo-400"></i>
                    <h3 class="text-sm font-bold text-slate-100">Daftar Antrian Subtitle</h3>
                </div>
                <div class="flex items-center gap-2">
                    <button onclick="clearCompletedQueue()" class="text-[10px] text-slate-400 hover:text-slate-200 bg-slate-800 px-2.5 py-1 rounded-lg flex items-center gap-1 active:bg-slate-700">
                        <i class="fa-solid fa-broom text-[9px]"></i> Bersihkan Selesai
                    </button>
                    <button onclick="closeQueueDrawer()" class="w-7 h-7 rounded-full bg-slate-800 text-slate-400 flex items-center justify-center active:bg-slate-700">
                        <i class="fa-solid fa-xmark text-xs"></i>
                    </button>
                </div>
            </div>

            <div id="queueListContainer" class="p-4 overflow-y-auto flex-1 space-y-2.5 no-scrollbar">
                <p class="text-center text-xs text-slate-500 py-8">Belum ada antrian aktif.</p>
            </div>

        </div>
    </div>

    <script>
        let currentPath = "";
        let parentPath = null;
        let allCurrentVideos = [];
        let selectedVideoPaths = new Set();
        let queuePollTimer = null;
        let activeTab = "browse";

        function switchTab(tab) {
            activeTab = tab;
            const btnB = document.getElementById("tabBtnBrowse");
            const btnD = document.getElementById("tabBtnDownload");
            const viewB = document.getElementById("viewBrowse");
            const viewD = document.getElementById("viewDownload");

            if (tab === "browse") {
                btnB.className = "py-2.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 bg-indigo-600 text-white shadow";
                btnD.className = "py-2.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 text-slate-400 hover:text-slate-200";
                viewB.classList.remove("hidden");
                viewD.classList.add("hidden");
            } else {
                btnD.className = "py-2.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 bg-purple-600 text-white shadow";
                btnB.className = "py-2.5 rounded-xl text-xs font-bold transition flex items-center justify-center gap-2 text-slate-400 hover:text-slate-200";
                viewD.classList.remove("hidden");
                viewB.classList.add("hidden");
                document.getElementById("dlFolderDisplay").textContent = currentPath || "D:\\\\translate\\\\gg";
                pollDownloads();
            }
        }

        async function loadFolder(path) {
            const container = document.getElementById("videosContainer");
            container.innerHTML = `
                <div class="text-center py-12 text-slate-500 text-sm">
                    <i class="fa-solid fa-spinner fa-spin text-2xl mb-2 text-indigo-400"></i>
                    <p>Membaca video di laptop...</p>
                </div>
            `;
            selectedVideoPaths.clear();
            updateBatchBar();

            try {
                const url = path ? `/api/browse?path=${encodeURIComponent(path)}` : `/api/browse`;
                const res = await fetch(url);
                const data = await res.json();
                
                currentPath = data.current_path;
                parentPath = data.parent_path;
                allCurrentVideos = data.videos || [];

                document.getElementById("pathDisplay").textContent = currentPath;
                document.getElementById("dlFolderDisplay").textContent = currentPath;
                const btnParent = document.getElementById("btnParentFolder");
                if (parentPath) {
                    btnParent.classList.remove("hidden");
                } else {
                    btnParent.classList.add("hidden");
                }

                // Subfolders
                const subSec = document.getElementById("subfoldersSection");
                const subList = document.getElementById("subfoldersList");
                subList.innerHTML = "";
                if (data.folders && data.folders.length > 0) {
                    subSec.classList.remove("hidden");
                    data.folders.forEach(f => {
                        const btn = document.createElement("button");
                        btn.className = "flex items-center gap-2 bg-slate-900 border border-slate-800 p-2.5 rounded-xl text-left truncate active:bg-slate-800 touch-card";
                        btn.onclick = () => loadFolder(f.path);
                        btn.innerHTML = `<i class="fa-solid fa-folder text-amber-400 text-sm shrink-0"></i><span class="truncate text-slate-300 font-medium">${f.name}</span>`;
                        subList.appendChild(btn);
                    });
                } else {
                    subSec.classList.add("hidden");
                }

                renderVideos(allCurrentVideos);

            } catch (err) {
                container.innerHTML = `<div class="p-4 bg-red-950/50 border border-red-800 rounded-xl text-red-300 text-xs text-center">Gagal memuat folder: ${err.message}</div>`;
            }
        }

        function goParent() {
            if (parentPath) {
                loadFolder(parentPath);
            }
        }

        function renderVideos(videos) {
            const container = document.getElementById("videosContainer");
            const badge = document.getElementById("videoCountBadge");
            badge.textContent = `${videos.length} video`;

            if (videos.length === 0) {
                container.innerHTML = `
                    <div class="text-center py-12 text-slate-500 text-xs">
                        <i class="fa-regular fa-folder-open text-3xl mb-2 text-slate-600 block"></i>
                        Tidak ada file video di folder ini.
                    </div>
                `;
                return;
            }

            container.innerHTML = "";
            videos.forEach((v, idx) => {
                const card = document.createElement("div");
                card.className = "bg-slate-900 border border-slate-800 rounded-2xl p-3 flex flex-col gap-2.5 transition active:border-indigo-500/40 touch-card";
                
                let srtBadges = [];
                if (v.srt_gemini_path) {
                    srtBadges.push(`<span class="bg-emerald-950 text-emerald-300 border border-emerald-700/80 text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 font-medium"><i class="fa-solid fa-sparkles text-[9px]"></i> Qwen Aichixia</span>`);
                }
                if (v.srt_xkiro_path) {
                    srtBadges.push(`<span class="bg-blue-950 text-blue-300 border border-blue-800/80 text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 font-medium"><i class="fa-solid fa-check text-[9px]"></i> xKiro</span>`);
                }
                if (v.srt_lite_path) {
                    srtBadges.push(`<span class="bg-purple-950 text-purple-300 border border-purple-800/80 text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 font-medium"><i class="fa-solid fa-check text-[9px]"></i> LiteRouter</span>`);
                }
                if (v.has_mandarin) {
                    srtBadges.push(`<span class="bg-amber-950 text-amber-300 border border-amber-800/80 text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 font-medium"><i class="fa-solid fa-language text-[9px]"></i> Mandarin Asli</span>`);
                }
                if (v.has_srt && srtBadges.length === 0) {
                    srtBadges.push(`<span class="bg-emerald-950 text-emerald-300 border border-emerald-800/80 text-[10px] px-2 py-0.5 rounded-full flex items-center gap-1 font-medium"><i class="fa-solid fa-check text-[9px]"></i> Ada SRT</span>`);
                }
                if (!v.has_srt && !v.has_mandarin) {
                    srtBadges.push(`<span class="bg-slate-800 text-slate-400 border border-slate-700 text-[10px] px-2 py-0.5 rounded-full">Belum Ada SRT</span>`);
                }

                const isChecked = selectedVideoPaths.has(v.path);

                card.innerHTML = `
                    <div class="flex items-start gap-2.5">
                        <div class="pt-1">
                            <input type="checkbox" onchange="toggleSelectVideo('${encodeURIComponent(v.path)}')" ${isChecked ? 'checked' : ''} class="w-4 h-4 accent-indigo-500 rounded cursor-pointer">
                        </div>
                        <div class="min-w-0 flex-1">
                            <h4 class="text-xs font-semibold text-slate-200 break-words leading-snug line-clamp-2">${v.name}</h4>
                            <div class="flex flex-wrap items-center gap-1.5 mt-1 text-[11px] text-slate-500">
                                <span>${v.size_mb} MB</span>
                                <span>•</span>
                                ${srtBadges.join(' ')}
                            </div>
                        </div>
                    </div>

                    <div class="flex flex-wrap items-center gap-1.5 pt-1 border-t border-slate-800/80">
                        <button onclick="queueSingleVideo('${encodeURIComponent(v.path)}')" class="flex-1 min-w-[110px] bg-indigo-600/20 border border-indigo-500/40 text-indigo-300 hover:bg-indigo-600 hover:text-white active:bg-indigo-700 py-1.5 rounded-xl text-xs font-semibold flex items-center justify-center gap-1.5 transition">
                            <i class="fa-solid fa-plus text-[10px]"></i>
                            <span>${v.has_srt ? 'Antrikan Ulang' : '+ Masuk Antrian'}</span>
                        </button>
                        
                        ${v.has_mandarin ? `
                            <button onclick="translateSrtDirect('${encodeURIComponent(v.srt_mandarin_path)}')" class="px-2.5 py-1.5 bg-amber-500/20 border border-amber-500/40 text-amber-300 hover:bg-amber-600 hover:text-white active:bg-amber-700 rounded-xl text-[11px] font-semibold flex items-center justify-center gap-1 transition" title="Langsung terjemahkan SRT Mandarin dengan Qwen (Aichixia) tanpa proses STT ulang">
                                <i class="fa-solid fa-bolt text-[9px]"></i>
                                <span>⚡ Qwen (0 STT)</span>
                            </button>
                        ` : ''}

                        ${v.srt_gemini_path ? `
                            <a href="/api/download_srt?path=${encodeURIComponent(v.srt_gemini_path)}" download class="px-2.5 py-1.5 bg-emerald-950/70 border border-emerald-700/60 text-emerald-300 rounded-xl text-[11px] font-semibold flex items-center justify-center gap-1 active:bg-emerald-900 transition">
                                <i class="fa-solid fa-download text-[9px]"></i>
                                <span>Qwen</span>
                            </a>
                        ` : ''}

                        ${v.srt_xkiro_path ? `
                            <a href="/api/download_srt?path=${encodeURIComponent(v.srt_xkiro_path)}" download class="px-2.5 py-1.5 bg-blue-950/70 border border-blue-700/60 text-blue-300 rounded-xl text-[11px] font-semibold flex items-center justify-center gap-1 active:bg-blue-900 transition">
                                <i class="fa-solid fa-download text-[9px]"></i>
                                <span>xKiro</span>
                            </a>
                        ` : ''}

                        ${v.srt_lite_path ? `
                            <a href="/api/download_srt?path=${encodeURIComponent(v.srt_lite_path)}" download class="px-2.5 py-1.5 bg-purple-950/70 border border-purple-700/60 text-purple-300 rounded-xl text-[11px] font-semibold flex items-center justify-center gap-1 active:bg-purple-900 transition">
                                <i class="fa-solid fa-download text-[9px]"></i>
                                <span>LiteRouter</span>
                            </a>
                        ` : ''}

                        ${v.has_srt && !v.srt_gemini_path && !v.srt_xkiro_path && !v.srt_lite_path ? `
                            <a href="/api/download_srt?path=${encodeURIComponent(v.srt_path)}" download class="px-2.5 py-1.5 bg-emerald-950/70 border border-emerald-700/60 text-emerald-300 rounded-xl text-[11px] font-semibold flex items-center justify-center gap-1 active:bg-emerald-900 transition">
                                <i class="fa-solid fa-download text-[9px]"></i>
                                <span>SRT</span>
                            </a>
                        ` : ''}
                    </div>
                `;
                container.appendChild(card);
            });
        }

        function toggleSelectVideo(encodedPath) {
            const path = decodeURIComponent(encodedPath);
            if (selectedVideoPaths.has(path)) {
                selectedVideoPaths.delete(path);
            } else {
                selectedVideoPaths.add(path);
            }
            updateBatchBar();
        }

        function toggleSelectAll() {
            if (selectedVideoPaths.size === allCurrentVideos.length) {
                selectedVideoPaths.clear();
                document.getElementById("btnSelectAllText").textContent = "Pilih Semua";
            } else {
                allCurrentVideos.forEach(v => selectedVideoPaths.add(v.path));
                document.getElementById("btnSelectAllText").textContent = "Batal Pilih";
            }
            renderVideos(allCurrentVideos);
            updateBatchBar();
        }

        function updateBatchBar() {
            const bar = document.getElementById("batchActionBar");
            const countText = document.getElementById("selectedCountText");
            const count = selectedVideoPaths.size;
            countText.textContent = count;
            if (count > 0) {
                bar.classList.remove("hidden");
            } else {
                bar.classList.add("hidden");
            }
        }

        async function queueSingleVideo(encodedPath) {
            const path = decodeURIComponent(encodedPath);
            await sendToQueue([path]);
        }

        async function translateSrtDirect(encodedSrtPath) {
            const path = decodeURIComponent(encodedSrtPath);
            try {
                const res = await fetch("/api/translate_srt", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ srt_path: path, provider: "aichixia" })
                });
                const data = await res.json();
                if (res.ok) {
                    pollQueueStatus();
                    openQueueDrawer();
                } else {
                    alert("Gagal: " + (data.detail || data.message));
                }
            } catch (err) {
                alert("Error koneksi: " + err.message);
            }
        }

        async function stopKaggleWorkerNow() {
            if (!confirm("Matikan Worker Kaggle sekarang? GPU akan dilepas dan kuota jam berhenti dihitung.")) return;
            try {
                const res = await fetch("/api/shutdown_kaggle", { method: "POST" });
                const d = await res.json();
                alert(d.message);
            } catch (err) {
                alert("Error: " + err.message);
            }
        }

        async function queueSelectedVideos() {
            const paths = Array.from(selectedVideoPaths);
            if (paths.length === 0) return;
            await sendToQueue(paths);
            selectedVideoPaths.clear();
            updateBatchBar();
            renderVideos(allCurrentVideos);
        }

        async function sendToQueue(paths) {
            try {
                const res = await fetch("/api/queue/add", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ video_paths: paths })
                });
                const data = await res.json();
                pollQueueStatus();
                openQueueDrawer();
            } catch (err) {
                alert("Gagal menambahkan ke antrian: " + err.message);
            }
        }

        async function pollQueueStatus() {
            try {
                const res = await fetch("/api/queue/status");
                if (!res.ok) return;
                const data = await res.json();

                const counts = data.counts;
                document.getElementById("badgeActive").textContent = `${counts.processing} Jalan`;
                document.getElementById("badgeQueued").textContent = `${counts.queued} Antri`;

                const headerDot = document.getElementById("headerPingDot");
                const headerRing = document.getElementById("headerPingRing");
                const headerBar = document.getElementById("headerProgressBar");
                const headerTitle = document.getElementById("headerQueueTitle");
                const headerStatus = document.getElementById("headerStatusText");

                if (counts.processing > 0) {
                    headerRing.classList.remove("hidden");
                    headerDot.className = "relative inline-flex rounded-full h-2.5 w-2.5 bg-emerald-400";
                    
                    const activeTask = data.tasks.find(t => t.state === "PROCESSING");
                    if (activeTask) {
                        headerTitle.textContent = `Memproses: ${activeTask.video_name}`;
                        headerBar.style.width = `${activeTask.progress}%`;
                        headerStatus.textContent = `${activeTask.progress}% - ${activeTask.status}`;
                    }
                } else if (counts.queued > 0) {
                    headerRing.classList.add("hidden");
                    headerDot.className = "relative inline-flex rounded-full h-2.5 w-2.5 bg-amber-400";
                    headerTitle.textContent = `Menunggu giliran antrian (${counts.queued} video)`;
                    headerBar.style.width = "0%";
                    headerStatus.textContent = "Menunggu antrian berikutnya...";
                } else {
                    headerRing.classList.add("hidden");
                    headerDot.className = "relative inline-flex rounded-full h-2.5 w-2.5 bg-slate-500";
                    headerTitle.textContent = counts.completed > 0 ? `Semua antrian selesai (${counts.completed} video)!` : "Antrian Kosong";
                    headerBar.style.width = counts.completed > 0 ? "100%" : "0%";
                    headerStatus.textContent = "Klik video untuk memasukkan ke antrian...";
                }

                renderQueueList(data.tasks);

            } catch (e) {
                console.error("Queue poll error:", e);
            }
        }

        function renderQueueList(tasks) {
            const list = document.getElementById("queueListContainer");
            if (!tasks || tasks.length === 0) {
                list.innerHTML = `<p class="text-center text-xs text-slate-500 py-8">Belum ada antrian aktif.</p>`;
                return;
            }

            list.innerHTML = "";
            tasks.forEach(t => {
                const item = document.createElement("div");
                item.className = "bg-slate-950 border border-slate-800 rounded-2xl p-3 text-xs space-y-2";
                
                let stateBadge = "";
                if (t.state === "PROCESSING") {
                    stateBadge = `<span class="bg-indigo-950 text-indigo-300 border border-indigo-700/60 px-2 py-0.5 rounded-full text-[10px] font-bold flex items-center gap-1 animate-pulse"><i class="fa-solid fa-spinner fa-spin text-[9px]"></i> Sedang Diproses (${t.progress}%)</span>`;
                } else if (t.state === "QUEUED") {
                    stateBadge = `<span class="bg-amber-950 text-amber-300 border border-amber-700/60 px-2 py-0.5 rounded-full text-[10px] font-medium flex items-center gap-1"><i class="fa-solid fa-clock text-[9px]"></i> Menunggu Antrian</span>`;
                } else if (t.state === "COMPLETED") {
                    stateBadge = `<span class="bg-emerald-950 text-emerald-300 border border-emerald-700/60 px-2 py-0.5 rounded-full text-[10px] font-bold flex items-center gap-1"><i class="fa-solid fa-check text-[9px]"></i> Selesai (${t.total_cues} cue)</span>`;
                } else {
                    stateBadge = `<span class="bg-red-950 text-red-300 border border-red-700/60 px-2 py-0.5 rounded-full text-[10px] font-bold">Gagal</span>`;
                }

                item.innerHTML = `
                    <div class="flex items-start justify-between gap-2">
                        <div class="min-w-0 flex-1">
                            <h5 class="font-semibold text-slate-200 truncate">${t.video_name}</h5>
                            <p class="text-[10px] text-slate-500 mt-0.5">Mode: ${(t.mode || 'Aichixia').toUpperCase()} • ${(t.lang || 'ZH').toUpperCase()}</p>
                        </div>
                        <div class="flex items-center gap-1.5 flex-shrink-0">
                            ${stateBadge}
                            ${t.state !== "PROCESSING" ? `
                                <button onclick="deleteQueueTask('${t.task_id}')" class="text-slate-500 hover:text-red-400 p-1 transition" title="Hapus dari antrian">
                                    <i class="fa-solid fa-trash-can text-[10px]"></i>
                                </button>
                            ` : ''}
                        </div>
                    </div>

                    ${t.state === "PROCESSING" ? `
                        <div class="w-full bg-slate-800 rounded-full h-1.5 overflow-hidden">
                            <div class="bg-indigo-500 h-1.5 rounded-full transition-all duration-300" style="width: ${t.progress}%"></div>
                        </div>
                        <p class="text-[10px] text-indigo-300 italic">${t.status}</p>
                    ` : ''}

                    ${t.state === "ERROR" ? `
                        <div class="bg-red-950/60 border border-red-800/80 p-2 rounded-xl text-[10px] text-red-300">
                            <strong>Detail Error:</strong> ${t.error || t.status}
                        </div>
                    ` : ''}

                    ${t.state === "COMPLETED" ? `
                        <div class="pt-1 flex flex-wrap items-center justify-between gap-1 border-t border-slate-900">
                            <span class="text-[10px] text-slate-400">Tersimpan di laptop</span>
                            <div class="flex items-center gap-1.5">
                                ${t.srt_gemini_path ? `
                                    <a href="/api/download_srt?path=${encodeURIComponent(t.srt_gemini_path)}" download class="bg-emerald-600 text-white font-bold px-2.5 py-1 rounded-lg text-[10px] flex items-center gap-1 shadow">
                                        <i class="fa-solid fa-download"></i> Qwen
                                    </a>
                                ` : ''}
                                ${t.srt_xkiro_path ? `
                                    <a href="/api/download_srt?path=${encodeURIComponent(t.srt_xkiro_path)}" download class="bg-blue-600 text-white font-bold px-2.5 py-1 rounded-lg text-[10px] flex items-center gap-1 shadow">
                                        <i class="fa-solid fa-download"></i> xKiro
                                    </a>
                                ` : ''}
                                ${t.srt_lite_path ? `
                                    <a href="/api/download_srt?path=${encodeURIComponent(t.srt_lite_path)}" download class="bg-purple-600 text-white font-bold px-2.5 py-1 rounded-lg text-[10px] flex items-center gap-1 shadow">
                                        <i class="fa-solid fa-download"></i> LiteRouter
                                    </a>
                                ` : ''}
                                ${!t.srt_gemini_path && !t.srt_xkiro_path && !t.srt_lite_path && t.srt_path ? `
                                    <a href="/api/download_srt?path=${encodeURIComponent(t.srt_path)}" download class="bg-emerald-500 text-slate-950 font-bold px-3 py-1 rounded-lg text-[11px] flex items-center gap-1 shadow">
                                        <i class="fa-solid fa-download"></i> Unduh SRT
                                    </a>
                                ` : ''}
                            </div>
                        </div>
                    ` : ''}
                `;
                list.appendChild(item);
            });
        }

        async function clearCompletedQueue() {
            try {
                await fetch("/api/queue/clear", { method: "POST" });
                pollQueueStatus();
            } catch(e) {
                console.error("Gagal membersihkan antrian:", e);
            }
        }

        async function deleteQueueTask(taskId) {
            try {
                await fetch(`/api/queue/${taskId}`, { method: "DELETE" });
                pollQueueStatus();
            } catch(e) {
                console.error("Gagal menghapus tugas:", e);
            }
        }

        // ==========================================
        // DOWNLOAD LOGIC (TAB 2)
        // ==========================================
        async function saveDirectKaggleUrl(silent = false) {
            const urlInput = document.getElementById("dlKaggleUrlDirect");
            const val = urlInput ? urlInput.value.trim() : "";
            try {
                const res = await fetch("/api/config");
                const cfg = await res.json();
                cfg.kaggle_url = val;
                await fetch("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(cfg)
                });
                if (!silent) {
                    alert(val ? "✅ Kaggle Worker URL berhasil disimpan!" : "URL Kaggle dikosongkan.");
                }
            } catch (e) {
                if (!silent) alert("Gagal menyimpan URL Kaggle: " + e.message);
            }
        }

        function currentDlMode() {
            return document.getElementById("dlModeLocal").checked ? "local" : "kaggle";
        }

        function setDlMode(mode) {
            const isLocal = (mode === "local");
            document.getElementById("dlModeLocal").checked = isLocal;
            document.getElementById("dlModeKaggle").checked = !isLocal;

            const applyCard = (el, active) => {
                el.className = "dl-mode-card p-3 rounded-xl border cursor-pointer space-y-1 " +
                    (active ? "bg-purple-950/40 border-purple-800/60" : "bg-slate-950 border-slate-800");
                const spans = el.querySelectorAll(":scope > span");
                if (spans[0]) spans[0].className = "text-xs font-bold flex items-center gap-1.5 " + (active ? "text-purple-300" : "text-slate-300");
                if (spans[1]) spans[1].className = "text-[10px] leading-relaxed block " + (active ? "text-purple-300/80" : "text-slate-400");
            };
            applyCard(document.getElementById("dlModeKaggleWrap"), !isLocal);
            applyCard(document.getElementById("dlModeLocalWrap"), isLocal);

            document.getElementById("dlWorkerBoxLabel").textContent = isLocal
                ? "Cloudflare Tunnel Worker URL (untuk transkripsi audio):"
                : "Cloudflare Tunnel Worker URL:";

            const note = document.getElementById("dlModeNote");
            if (isLocal) {
                note.className = "text-[10px] text-indigo-300/90 leading-relaxed bg-indigo-950/30 border border-indigo-900/60 p-2 rounded-lg";
                note.innerHTML = '<i class="fa-solid fa-microphone-lines"></i> Video diunduh ke laptop (kualitas asli), lalu <b>audionya dikirim ke worker Kaggle</b> (FireRedASR2) untuk subtitle.';
            } else {
                note.className = "text-[10px] text-purple-300/80 leading-relaxed bg-purple-950/30 border border-purple-900/50 p-2 rounded-lg";
                note.innerHTML = '<i class="fa-solid fa-cloud"></i> Video diunduh &amp; dikompres 480p di server Google. Situs yang memblokir IP cloud (Supjav/fc2) otomatis pindah ke mode Lokal.';
            }
        }

        async function startDownloadLink() {
            const url = document.getElementById("dlUrlInput").value.trim();
            if (!url) {
                alert("Silakan masukkan URL link video terlebih dahulu!");
                return;
            }
            const customName = document.getElementById("dlCustomName").value.trim();
            const autoQueue = document.getElementById("dlAutoQueue").checked;
            const directKaggleUrl = document.getElementById("dlKaggleUrlDirect").value.trim();

            // Situs yang memblokir IP cloud -> paksa mode lokal (tanpa ganggu user)
            if (/supjav\.com|fc2stream|fc2\//i.test(url) && currentDlMode() === "kaggle") {
                setDlMode("local");
            }
            const useKaggle = currentDlMode() === "kaggle";

            if (useKaggle) {
                if (directKaggleUrl) {
                    await saveDirectKaggleUrl(true);
                } else {
                    alert(`⚠️ Mode Kompres Cloud Kaggle aktif, tetapi URL Kaggle Worker belum diisi!\n\n1. Buka Notebook Kaggle Anda (klik 'Buka Notebook Kaggle')\n2. Klik 'Run All'\n3. Salin URL Cloudflare Tunnel yang muncul, lalu tempel di kotak URL di atas.`);
                    document.getElementById("dlKaggleUrlDirect").focus();
                    return;
                }
            }

            try {
                const res = await fetch("/api/download_link", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({
                        url: url,
                        save_dir: currentPath || "D:\\\\translate\\\\gg",
                        custom_name: customName,
                        auto_queue: autoQueue,
                        dl_mode: useKaggle ? "kaggle" : "local",
                        use_kaggle_480p: useKaggle
                    })
                });
                if (!res.ok) {
                    const err = await res.json();
                    throw new Error(err.detail || "Gagal memulai download");
                }
                document.getElementById("dlUrlInput").value = "";
                document.getElementById("dlCustomName").value = "";
                pollDownloads();
            } catch (e) {
                alert("Error: " + e.message);
            }
        }

        async function pollDownloads() {
            try {
                const res = await fetch("/api/downloads");
                if (!res.ok) return;
                const dls = await res.json();
                renderDownloadsList(dls);
            } catch (e) {
                console.error("Poll downloads error:", e);
            }
        }

        async function clearDownloads() {
            try {
                await fetch("/api/downloads/clear", { method: "POST" });
                pollDownloads();
            } catch(e) {}
        }

        async function deleteDownload(dlId) {
            try {
                await fetch(`/api/downloads/${dlId}`, { method: "DELETE" });
                pollDownloads();
            } catch(e) {}
        }

        function renderDownloadsList(dls) {
            const container = document.getElementById("downloadsMonitorList");
            if (!dls || dls.length === 0) {
                container.innerHTML = `<p class="text-center text-xs text-slate-600 py-6">Belum ada proses download.</p>`;
                return;
            }

            container.innerHTML = "";
            dls.forEach(d => {
                const card = document.createElement("div");
                card.className = "bg-slate-900 border border-slate-800 rounded-2xl p-3.5 space-y-2.5 text-xs";

                let badge = "";
                if (d.state === "DOWNLOADING") {
                    badge = `<span class="bg-purple-950 text-purple-300 border border-purple-700/60 px-2 py-0.5 rounded-full text-[10px] font-bold flex items-center gap-1 animate-pulse"><i class="fa-solid fa-arrow-down animate-bounce text-[9px]"></i> Unduh (${d.speed_str})</span>`;
                } else if (d.state === "STARTING" || d.state === "QUEUED") {
                    badge = `<span class="bg-amber-950 text-amber-300 border border-amber-700/60 px-2 py-0.5 rounded-full text-[10px] font-medium flex items-center gap-1"><i class="fa-solid fa-spinner fa-spin text-[9px]"></i> Menghubungkan...</span>`;
                } else if (d.state === "COMPLETED") {
                    badge = `<span class="bg-emerald-950 text-emerald-300 border border-emerald-700/60 px-2 py-0.5 rounded-full text-[10px] font-bold flex items-center gap-1"><i class="fa-solid fa-check text-[9px]"></i> Selesai (${d.file_size_str || ''})</span>`;
                } else {
                    badge = `<span class="bg-red-950 text-red-300 border border-red-700/60 px-2 py-0.5 rounded-full text-[10px] font-bold">Gagal</span>`;
                }

                const thumbHtml = d.thumbnail ? `
                    <div class="relative flex-shrink-0">
                        <img src="${d.thumbnail}" class="w-16 h-12 object-cover rounded-lg border border-slate-700 bg-slate-950 shadow" onerror="this.parentElement.style.display='none'">
                        ${d.duration_str ? `<span class="absolute bottom-0.5 right-0.5 bg-black/85 text-[9px] font-mono text-white px-1 rounded font-bold">${d.duration_str}</span>` : ''}
                    </div>
                ` : '';

                const metaTags = (d.duration_str || d.resolution) ? `
                    <div class="flex items-center gap-1.5 flex-wrap mt-1">
                        ${d.duration_str ? `<span class="bg-indigo-950/80 text-indigo-300 border border-indigo-700/50 px-1.5 py-0.5 rounded text-[10px] font-medium flex items-center gap-1"><i class="fa-solid fa-clock text-[9px]"></i> ${d.duration_str}</span>` : ''}
                        ${d.resolution ? `<span class="bg-emerald-950/80 text-emerald-300 border border-emerald-700/50 px-1.5 py-0.5 rounded text-[10px] font-medium flex items-center gap-1"><i class="fa-solid fa-film text-[9px]"></i> ${d.resolution}</span>` : ''}
                    </div>
                ` : '';

                card.innerHTML = `
                    <div class="flex items-start gap-2.5">
                        ${thumbHtml}
                        <div class="flex-1 min-w-0">
                            <div class="flex items-start justify-between gap-1.5">
                                <h5 class="font-bold text-slate-200 text-xs leading-snug line-clamp-2">${d.title || 'Video Online'}</h5>
                                <button onclick="deleteDownload('${d.dl_id}')" class="text-slate-500 hover:text-red-400 p-1 transition flex-shrink-0" title="Hapus"><i class="fa-solid fa-xmark text-xs"></i></button>
                            </div>
                            ${metaTags}
                            <div class="mt-1 flex items-center justify-between gap-2">
                                <span class="text-[9px] font-mono text-slate-500 truncate max-w-[140px]">${d.url}</span>
                                <div>${badge}</div>
                            </div>
                        </div>
                    </div>

                    ${d.state === "DOWNLOADING" ? `
                        <div class="w-full bg-slate-800 rounded-full h-2 overflow-hidden">
                            <div class="bg-gradient-to-r from-purple-500 to-indigo-400 h-2 rounded-full transition-all duration-300" style="width: ${d.progress}%"></div>
                        </div>
                        <div class="flex items-center justify-between text-[10px] text-slate-400">
                            <span>${d.progress}% • ${d.downloaded_str} / ${d.total_str}</span>
                            <span class="text-purple-300 font-semibold">ETA: ${d.eta_str}</span>
                        </div>
                    ` : ''}

                    ${d.state === "ERROR" ? `
                        <div class="bg-red-950/60 border border-red-800/80 p-2 rounded-xl text-[10px] text-red-300">
                            <strong>Detail Error:</strong> ${d.error || d.status}
                        </div>
                    ` : ''}

                    ${d.state === "COMPLETED" ? `
                        <div class="pt-1.5 flex items-center justify-between border-t border-slate-800/80 text-[10px]">
                            <span class="text-emerald-400 font-medium truncate max-w-[220px]"><i class="fa-solid fa-folder-check"></i> ${d.file_path}</span>
                            <button onclick="loadFolder('${encodeURIComponent(currentPath)}')" class="bg-slate-800 px-2 py-1 rounded-lg text-slate-300 active:bg-slate-700">
                                Segarkan Folder
                            </button>
                        </div>
                    ` : ''}
                `;
                container.appendChild(card);
            });
        }

        function openQueueDrawer() {
            document.getElementById("queueDrawer").classList.remove("hidden");
            pollQueueStatus();
        }

        function closeQueueDrawer() {
            document.getElementById("queueDrawer").classList.add("hidden");
        }

        async function openSettingsModal() {
            try {
                const res = await fetch("/api/config");
                const cfg = await res.json();
                document.getElementById("cfgSpeechModel").value = cfg.speech_model || "universal-3-5-pro";
                document.getElementById("cfgLangCode").value = cfg.language_code || "auto";
                document.getElementById("cfgTranslateToId").checked = cfg.translate_to_id ?? true;
                document.getElementById("cfgTranslationMode").value = cfg.translation_mode || "both";
                document.getElementById("cfgWordBoost").value = cfg.word_boost || "";
                document.getElementById("cfgPauseSec").value = cfg.pause_sec ?? 0.6;
                document.getElementById("cfgMaxChar").value = cfg.max_char_len ?? 42;
                document.getElementById("cfgKaggleUrl").value = cfg.kaggle_url || "";
            } catch (e) {}
            document.getElementById("settingsModal").classList.remove("hidden");
        }

        function closeSettingsModal() {
            document.getElementById("settingsModal").classList.add("hidden");
        }

        async function saveSettings() {
            const mode = document.getElementById("cfgTranslationMode").value;
            const body = {
                speech_model: document.getElementById("cfgSpeechModel").value,
                language_code: document.getElementById("cfgLangCode").value,
                translate_to_id: document.getElementById("cfgTranslateToId").checked,
                translation_mode: mode,
                translation_style: "natural_no_sensor",
                word_boost: document.getElementById("cfgWordBoost").value,
                boost_param: "default",
                pause_sec: parseFloat(document.getElementById("cfgPauseSec").value) || 0.6,
                max_char_len: parseInt(document.getElementById("cfgMaxChar").value) || 42,
                kaggle_url: document.getElementById("cfgKaggleUrl").value.trim(),
                punctuate: true,
                format_text: true,
                disfluencies: false
            };

            try {
                await fetch("/api/config", {
                    method: "POST",
                    headers: { "Content-Type": "application/json" },
                    body: JSON.stringify(body)
                });
                closeSettingsModal();
                alert("Pengaturan mode berhasil disimpan!");
            } catch (e) {
                alert("Gagal menyimpan: " + e.message);
            }
        }

        function initApp() {
            if (typeof setDlMode === "function") setDlMode("kaggle");
            loadFolder();
            pollQueueStatus();
            pollDownloads();
            fetch("/api/config").then(r => r.json()).then(cfg => {
                if (cfg && cfg.kaggle_url) {
                    const el = document.getElementById("dlKaggleUrlDirect");
                    if (el) el.value = cfg.kaggle_url;
                }
            }).catch(()=>{});
            queuePollTimer = setInterval(() => {
                pollQueueStatus();
                if (activeTab === "download") pollDownloads();
            }, 2500);
        }

        if (document.readyState === "loading") {
            document.addEventListener("DOMContentLoaded", initApp);
        } else {
            initApp();
        }
    </script>
</body>
</html>
"""
    return HTMLResponse(content=html_content)

if __name__ == "__main__":
    try:
        from pycloudflared import try_cloudflare
        tunnel_url = try_cloudflare(port=8080)
        print("\n" + "="*60)
        print(f"🌐 PUBLIC WEB REMOTE TUNNEL (HP): {tunnel_url}")
        print("="*60 + "\n")
    except Exception as e:
        print(f"Cloudflare tunnel notice: {e}")
    uvicorn.run(app, host="0.0.0.0", port=8080)
