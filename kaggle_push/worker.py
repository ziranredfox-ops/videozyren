import os, sys, subprocess, time, json, uuid, re, shutil, base64, concurrent.futures
from pathlib import Path

print("Menyiapkan dependencies di Cloud Kaggle...")
reqs = [
    "yt-dlp", "fastapi", "uvicorn", "pycloudflared", "requests", "curl_cffi",
    "python-multipart", "soundfile", "huggingface_hub",
    "cn2an>=0.5.23", "kaldiio>=2.18.0", "kaldi_native_fbank>=1.15",
    "sentencepiece", "peft", "transformers>=4.40.0", "cryptography"
]
for attempt in range(5):
    try:
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q",
            "--default-timeout=180", "--retries=10"
        ] + reqs, check=True)
        break
    except subprocess.CalledProcessError as e:
        print(f"Pip install attempt {attempt+1} gagal ({e}), mengulang dalam 5 detik...")
        time.sleep(5)
else:
    raise RuntimeError("Gagal menginstall dependencies setelah 5 percobaan.")

import requests
from fastapi import FastAPI, BackgroundTasks, HTTPException, UploadFile, File
from fastapi.responses import FileResponse
from pydantic import BaseModel
import uvicorn
from pycloudflared import try_cloudflare

app = FastAPI(title="Kaggle Cloud Video Compressor & FireRedASR2S")
WORK_DIR = Path("/kaggle/working/compressed_videos")
WORK_DIR.mkdir(parents=True, exist_ok=True)

JOBS = {}

class CompressTask(BaseModel):
    url: str
    custom_name: str = ""

def format_size(bytes_val):
    if not bytes_val or bytes_val <= 0: return "0 MB"
    mb = bytes_val / (1024 * 1024)
    if mb >= 1024: return f"{mb / 1024:.2f} GB"
    return f"{mb:.1f} MB"

def unwrap_png(data: bytes) -> bytes:
    if len(data) >= 8 and data[:8] == b'\x89PNG\r\n\x1a\n':
        idx = data.find(b'rout')
        if idx != -1:
            chunk_len = int.from_bytes(data[idx-4:idx], byteorder='big')
            return data[idx+4 : idx+4+chunk_len]
        iend_idx = data.find(b'IEND')
        if iend_idx != -1 and len(data) > iend_idx + 8:
            return data[iend_idx + 8:]
    return data

def download_rou_in_cloud(url: str, out_raw_ts: str, job: dict):
    page_url = url.strip()
    m = re.search(r'/v/([a-zA-Z0-9]+)', page_url)
    video_id = m.group(1) if m else "video"
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://rou.video/',
        'Origin': 'https://rou.video'
    }
    session = requests.Session()
    adapter = requests.adapters.HTTPAdapter(pool_connections=30, pool_maxsize=30, max_retries=3)
    session.mount('https://', adapter)
    session.mount('http://', adapter)

    resp = session.get(page_url, headers=headers, timeout=15)
    resp.raise_for_status()
    html = resp.text

    hls_api_url = f"https://rou.video/api/hls/{video_id}"
    ev_match = re.search(r'ev:\$R\[\d+\]=\{d:"([^"]+)",k:(\d+)\}', html)
    if not ev_match:
        ev_match = re.search(r'ev:\{d:"([^"]+)",k:(\d+)\}', html)
    if ev_match:
        try:
            d_val = ev_match.group(1)
            k_val = int(ev_match.group(2))
            raw = base64.b64decode(d_val).decode('latin1')
            unmasked = ''.join([chr(ord(c) - k_val) for c in raw])
            ev_data = json.loads(unmasked)
            vurl = ev_data.get('videoUrl')
            if vurl:
                hls_api_url = "https://rou.video" + vurl if vurl.startswith('/') else vurl
        except Exception as e:
            print(f"ev error: {e}")

    hls_headers = dict(headers)
    hls_headers['Referer'] = page_url
    resp_hls = session.get(hls_api_url, headers=hls_headers, timeout=15)
    resp_hls.raise_for_status()
    playlist_text = unwrap_png(resp_hls.content).decode('utf-8', errors='ignore')

    segments = []
    for line in playlist_text.splitlines():
        line = line.strip()
        if line and not line.startswith('#'):
            segments.append(line)

    total_segments = len(segments)
    if total_segments == 0:
        raise Exception("Playlist kosong atau tidak ditemukan segmen video!")

    job["status"] = f"Kaggle mengunduh {total_segments} segmen di server Google..."
    downloaded = [None] * total_segments
    completed = 0

    def fetch(idx, seg_url):
        nonlocal completed
        try:
            r = session.get(seg_url, headers=headers, timeout=20)
            if r.status_code == 200:
                raw_seg = unwrap_png(r.content)
                downloaded[idx] = raw_seg
                completed += 1
                job["progress"] = int(10 + (completed / total_segments) * 40)
        except Exception as ex:
            print(f"Seg {idx} error: {ex}")

    with concurrent.futures.ThreadPoolExecutor(max_workers=25) as executor:
        futs = [executor.submit(fetch, i, u) for i, u in enumerate(segments)]
        concurrent.futures.wait(futs)

    with open(out_raw_ts, 'wb') as f_out:
        for chunk in downloaded:
            if chunk:
                f_out.write(chunk)

def download_avple_in_cloud(url: str, out_raw_ts: str, job: dict):
    from curl_cffi import requests as c_requests

    page_url = url.strip()
    headers = {
        'Referer': 'https://avple.tv/',
        'Origin': 'https://avple.tv',
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }

    if '.m3u8' in page_url or 'cdnedge.live' in page_url or 'dpaste.org' in page_url:
        hls_url = page_url
        r_m3u8 = c_requests.get(hls_url, impersonate='chrome124', headers=headers, timeout=15)
        if r_m3u8.status_code != 200:
            raise Exception(f"Gagal membuka playlist HLS di Kaggle: HTTP {r_m3u8.status_code}")
        playlist_text = r_m3u8.text
        base_cdn_url = hls_url.rsplit('/', 1)[0]
    else:
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

        play_path = instance.get("play", "")
        if not play_path:
            raise Exception("Path streaming video tidak ditemukan")

        cdns = [
            "d862cp1.cdnedge.live", "q2cyl71.cdnedge.live", "u89ey1.cdnedge.live",
            "wo8801.cdnedge.live", "6m7d1.cdnedge.live", "fa6781.cdnedge.live", "pg2z71.cdnedge.live"
        ]
        hls_url = None
        playlist_text = None

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

        if not hls_url or not playlist_text:
            raise Exception("Gagal mendapatkan playlist HLS dari CDN Avple")

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
            seg_url = line if line.startswith("http") else f"{base_cdn_url}/{line}"
            segments_with_keys.append((seg_url, current_key_bytes))

    total_segments = len(segments_with_keys)
    if total_segments == 0:
        raise Exception("Playlist kosong atau tidak ada segmen video")

    job["status"] = f"Kaggle mengunduh {total_segments} segmen Avple di Google Cloud..."
    downloaded = [None] * total_segments
    completed = 0

    from cryptography.hazmat.primitives.ciphers import Cipher, algorithms, modes

    def fetch(idx, seg_item):
        nonlocal completed
        seg_url, key_bytes = seg_item
        for _ in range(3):
            try:
                res = c_requests.get(seg_url, headers=headers, impersonate='chrome124', timeout=20)
                if res.status_code == 200 and len(res.content) > 0:
                    data = res.content
                    if key_bytes and len(key_bytes) == 16:
                        # HLS AES-128 CBC decrypt
                        cipher = Cipher(algorithms.AES(key_bytes), modes.CBC(b'\x00' * 16))
                        decryptor = cipher.decryptor()
                        data = decryptor.update(data) + decryptor.finalize()
                    downloaded[idx] = data
                    completed += 1
                    job["progress"] = int(10 + (completed / total_segments) * 45)
                    return
            except Exception:
                time.sleep(0.5)

    with concurrent.futures.ThreadPoolExecutor(max_workers=20) as executor:
        futs = [executor.submit(fetch, i, item) for i, item in enumerate(segments_with_keys)]
        concurrent.futures.wait(futs)

    with open(out_raw_ts, 'wb') as f_out:
        for chunk in downloaded:
            if chunk:
                f_out.write(chunk)

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

    m_alt = re.search(r"video_alt_url\s*:\s*['\"]([^'\"]+)['\"]", html)
    if m_alt:
        return m_alt.group(1), title

    m_v = re.search(r"video_url\s*:\s*['\"]([^'\"]+)['\"]", html)
    if m_v:
        return m_v.group(1), title

    m_c = re.search(r'"contentUrl":\s*"([^"]+)"', html)
    if m_c:
        return m_c.group(1), title

    return None, title

def download_http_stream(vurl, out_path, referer, job=None):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': referer
    }
    with requests.get(vurl, headers=headers, stream=True, allow_redirects=True, timeout=180) as r:
        r.raise_for_status()
        tot = int(r.headers.get('content-length', 0))
        down = 0
        with open(out_path, 'wb') as f:
            for chunk in r.iter_content(chunk_size=2*1024*1024):
                if chunk:
                    f.write(chunk)
                    down += len(chunk)
                    if job and tot > 0:
                        job["progress"] = int(10 + (down / tot) * 45)

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
    # Resolve harus dari IP yang sama dengan yang mengunduh (token fc2stream terikat IP)
    from curl_cffi import requests as c_requests
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://supjav.com/'
    }
    resp = c_requests.get(page_url, headers=headers, impersonate='chrome124', timeout=25)
    resp.raise_for_status()
    html = resp.text

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
                return stream
    raise Exception("Link stream segar tidak ditemukan di halaman player Supjav")

def download_hls_png_in_cloud(m3u8_url: str, out_raw_ts: str, job: dict):
    from urllib.parse import urljoin
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36',
        'Referer': 'https://fc2stream.tv/'
    }

    def fetch_text(u):
        r = requests.get(u, headers=headers, timeout=25)
        r.raise_for_status()
        return unwrap_png(r.content).decode('utf-8', errors='ignore')

    text = fetch_text(m3u8_url)
    media_url = m3u8_url

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
        pick = variants[0]  # 480p, karena hasil akhir memang 480p
        job["status"] = f"Kaggle memilih varian {pick[0]}p dari master playlist..."
        media_url = urljoin(m3u8_url, pick[2])
        text = fetch_text(media_url)

    segments = [urljoin(media_url, l.strip()) for l in text.splitlines() if l.strip() and not l.startswith('#')]
    total_segments = len(segments)
    if total_segments == 0:
        raise Exception("Playlist kosong atau tidak ada segmen video fc2stream")

    job["status"] = f"Kaggle mengunduh {total_segments} segmen fc2stream di server Google..."
    downloaded = [None] * total_segments
    completed = 0

    def fetch(idx, seg_url):
        nonlocal completed
        for attempt in range(3):
            try:
                r = requests.get(seg_url, headers=headers, timeout=25)
                if r.status_code == 200 and r.content:
                    downloaded[idx] = unwrap_png(r.content)
                    completed += 1
                    job["progress"] = int(10 + (completed / total_segments) * 45)
                    return
            except Exception as ex:
                print(f"Seg {idx} error (percobaan {attempt + 1}): {ex}")
                time.sleep(0.5)
        print(f"Seg {idx} gagal total")

    with concurrent.futures.ThreadPoolExecutor(max_workers=16) as executor:
        futs = [executor.submit(fetch, i, u) for i, u in enumerate(segments)]
        concurrent.futures.wait(futs)

    missing = sum(1 for c in downloaded if not c)
    if missing == total_segments:
        raise Exception("Semua segmen fc2stream gagal diunduh")
    if missing:
        job["status"] = f"Peringatan: {missing} segmen hilang, melanjutkan..."

    with open(out_raw_ts, 'wb') as f_out:
        for chunk in downloaded:
            if chunk:
                f_out.write(chunk)

def process_video_in_cloud(job_id: str, url: str, custom_name: str):
    job = JOBS[job_id]
    job["status"] = "Mendownload video asli di server cloud Google (Kaggle)..."
    job["progress"] = 10

    clean_title = re.sub(r'[\\/*?:"<>|]', '', custom_name.strip()) if custom_name else f"video_{job_id}"
    raw_path = WORK_DIR / f"raw_{job_id}.mp4"
    out_480p = WORK_DIR / f"{clean_title}_480p.mp4"

    try:
        if 'supjav.com' in url:
            job["status"] = "Kaggle mengambil link stream segar dari Supjav (token terikat IP)..."
            url = resolve_supjav(url)
            job["status"] = "Link stream segar didapat, mulai diunduh di server Google..."
        if 'rou.video' in url:
            raw_ts = WORK_DIR / f"raw_{job_id}.ts"
            download_rou_in_cloud(url, str(raw_ts), job)
            actual_raw = raw_ts
        elif 'avple' in url or 'cdnedge.live' in url or 'dpaste.org' in url:
            raw_ts = WORK_DIR / f"raw_{job_id}.ts"
            download_avple_in_cloud(url, str(raw_ts), job)
            actual_raw = raw_ts
        elif 'pimpbunny' in url:
            direct_vurl, page_title = resolve_kvs_video(url)
            if not direct_vurl:
                raise Exception("Gagal mengekstrak direct URL video dari pimpbunny.")
            if not custom_name and page_title:
                clean_title = page_title
                out_480p = WORK_DIR / f"{clean_title}_480p.mp4"
            job["status"] = "Kaggle mengunduh stream Pimpbunny langsung di Google Cloud..."
            download_http_stream(direct_vurl, str(raw_path), url, job)
            actual_raw = raw_path if raw_path.exists() else None
            if not actual_raw:
                found = list(WORK_DIR.glob(f"raw_{job_id}.*"))
                if found: actual_raw = found[0]
        elif '.m3u8' in url:
            raw_ts = WORK_DIR / f"raw_{job_id}.ts"
            download_hls_png_in_cloud(url, str(raw_ts), job)
            actual_raw = raw_ts
        else:
            cmd_dl = [
                "yt-dlp",
                "--no-check-certificates",
                "-f", "bestvideo+bestaudio/best",
                "--merge-output-format", "mp4",
                "-o", str(raw_path),
                url
            ]
            subprocess.run(cmd_dl, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            actual_raw = raw_path if raw_path.exists() else None
            if not actual_raw:
                found = list(WORK_DIR.glob(f"raw_{job_id}.*"))
                if found: actual_raw = found[0]

        if not actual_raw or not actual_raw.exists():
            raise Exception("Gagal mendownload video asli di server Google Kaggle.")

        raw_size = format_size(actual_raw.stat().st_size)

        has_gpu = False
        try:
            chk = subprocess.run(["nvidia-smi"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            if chk.returncode == 0:
                has_gpu = True
        except:
            has_gpu = False

        if has_gpu:
            job["status"] = f"Video asli ({raw_size}) siap! GPU T4 mengompres ke 480p via NVENC (super cepat)..."
            cmd_ffmpeg = [
                "ffmpeg", "-y",
                "-i", str(actual_raw),
                "-vf", "scale=-2:480",
                "-c:v", "h264_nvenc",
                "-preset", "p4",
                "-cq", "28",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                str(out_480p)
            ]
        else:
            job["status"] = f"Video asli ({raw_size}) siap! CPU mengompres ke 480p (ultrafast)..."
            cmd_ffmpeg = [
                "ffmpeg", "-y",
                "-i", str(actual_raw),
                "-vf", "scale=-2:480",
                "-c:v", "libx264",
                "-crf", "26",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                str(out_480p)
            ]

        job["progress"] = 65
        subprocess.run(cmd_ffmpeg, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, text=True)
        if not out_480p.exists():
            cmd_fallback = [
                "ffmpeg", "-y",
                "-i", str(actual_raw),
                "-vf", "scale=-2:480",
                "-c:v", "libx264",
                "-crf", "26",
                "-preset", "ultrafast",
                "-c:a", "aac",
                "-b:a", "96k",
                "-movflags", "+faststart",
                str(out_480p)
            ]
            subprocess.run(cmd_fallback, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

        if not out_480p.exists():
            raise Exception("Gagal mengompres video ke 480p.")

        if actual_raw.exists():
            try: actual_raw.unlink()
            except: pass

        final_size = format_size(out_480p.stat().st_size)
        job["status"] = f"Selesai dikompres ke 480p ({final_size}, hemat dari {raw_size})!"
        job["progress"] = 100
        job["state"] = "COMPLETED"
        job["file_path"] = str(out_480p)
        job["file_name"] = out_480p.name
        job["file_size"] = final_size

    except Exception as e:
        job["state"] = "ERROR"
        job["status"] = f"Error di Cloud Kaggle: {str(e)[:120]}"
        job["error"] = str(e)

@app.post("/api/compress")
def start_compress(task: CompressTask, background_tasks: BackgroundTasks):
    job_id = str(uuid.uuid4())[:8]
    JOBS[job_id] = {
        "job_id": job_id,
        "url": task.url,
        "state": "PROCESSING",
        "progress": 5,
        "status": "Tugas kompresi diterima oleh server Kaggle...",
        "created_at": time.time()
    }
    background_tasks.add_task(process_video_in_cloud, job_id, task.url, task.custom_name)
    return {"job_id": job_id, "message": "Tugas kompresi 480p dimulai di Cloud Kaggle"}

@app.get("/api/job/{job_id}")
def get_job(job_id: str):
    if job_id not in JOBS:
        raise HTTPException(status_code=404, detail="Job tidak ditemukan")
    return JOBS[job_id]

@app.get("/api/download/{job_id}")
def download_result(job_id: str):
    job = JOBS.get(job_id)
    if not job or job.get("state") != "COMPLETED":
        raise HTTPException(status_code=400, detail="Video belum selesai diproses")
    file_path = job.get("file_path")
    if not os.path.exists(file_path):
        raise HTTPException(status_code=404, detail="File video tidak ditemukan")
    return FileResponse(file_path, filename=job.get("file_name", "video_480p.mp4"), media_type="video/mp4")

# ==========================================
# FireRedASR2S - SOTA Mandarin ASR Engine (Official Pipeline)
# ==========================================

def fmt_srt_time(seconds: float) -> str:
    millis = int(round(seconds * 1000))
    h = millis // 3600000; millis %= 3600000
    m = millis // 60000;   millis %= 60000
    s = millis // 1000;    millis %= 1000
    return f"{h:02d}:{m:02d}:{s:02d},{millis:03d}"

FIRERED_SYSTEM = None
FIREREDASR_REPO = Path("/kaggle/working/FireRedASR2S")

def setup_fireredasr():
    global FIRERED_SYSTEM

    if FIRERED_SYSTEM is not None:
        return FIRERED_SYSTEM

    # 1. Clone repository jika belum ada
    if not FIREREDASR_REPO.exists():
        print("Cloning FireRedASR2S repository...")
        subprocess.run([
            "git", "clone", "--depth", "1",
            "https://github.com/FireRedTeam/FireRedASR2S.git",
            str(FIREREDASR_REPO)
        ], check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 2. Setup paths
    for p in [
        str(FIREREDASR_REPO),
        str(FIREREDASR_REPO / "fireredasr2s"),
        str(FIREREDASR_REPO / "fireredasr2s" / "fireredasr2"),
        str(FIREREDASR_REPO / "fireredasr2s" / "vad"),
        str(FIREREDASR_REPO / "fireredasr2s" / "punc"),
    ]:
        if p not in sys.path:
            sys.path.insert(0, p)
    os.environ["PYTHONPATH"] = str(FIREREDASR_REPO) + ":" + os.environ.get("PYTHONPATH", "")

    # 3. Install requirements
    req = FIREREDASR_REPO / "requirements.txt"
    if req.exists():
        print("Installing FireRedASR2S requirements...")
        subprocess.run([
            sys.executable, "-m", "pip", "install", "-q", "-r", str(req),
            "--extra-index-url", "https://download.pytorch.org/whl/cu118"
        ], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)

    # 4. Download models ke pretrained_models/
    from huggingface_hub import snapshot_download
    models_dir = FIREREDASR_REPO / "pretrained_models"
    models_dir.mkdir(parents=True, exist_ok=True)

    # ASR: FireRedASR2-AED (1.1B)
    asr_dir = models_dir / "FireRedASR2-AED"
    if not asr_dir.exists() or not any(asr_dir.iterdir()):
        print("Downloading FireRedASR2-AED weights...")
        snapshot_download(repo_id="FireRedTeam/FireRedASR2-AED", local_dir=str(asr_dir), ignore_patterns=["*.md"])

    # VAD: FireRedVAD (~3MB)
    vad_dir = models_dir / "FireRedVAD"
    if not vad_dir.exists() or not (vad_dir / "VAD").exists():
        print("Downloading FireRedVAD weights...")
        snapshot_download(repo_id="FireRedTeam/FireRedVAD", local_dir=str(vad_dir), ignore_patterns=["*.md"])

    # Punc: FireRedPunc (tanda baca otomatis)
    punc_dir = models_dir / "FireRedPunc"
    if not punc_dir.exists() or not (punc_dir / "chinese-lert-base").exists():
        print("Downloading FireRedPunc weights...")
        snapshot_download(repo_id="FireRedTeam/FireRedPunc", local_dir=str(punc_dir), ignore_patterns=["*.md"])

    # 5. Inisialisasi FireRedAsr2System resmi
    print("Memuat FireRedAsr2System (VAD + ASR + Punc) di GPU Tesla T4...")
    from fireredasr2s.fireredasr2system import FireRedAsr2System, FireRedAsr2SystemConfig
    cfg = FireRedAsr2SystemConfig(
        vad_model_dir=str(vad_dir / "VAD"),
        asr_type="aed",
        asr_model_dir=str(asr_dir),
        punc_model_dir=str(punc_dir),
        enable_vad=True,
        enable_lid=False,
        enable_punc=True
    )
    FIRERED_SYSTEM = FireRedAsr2System(cfg)
    print("FireRedAsr2System (VAD + ASR + Punc) SIAP di GPU Tesla T4!")
    return FIRERED_SYSTEM

TRANSCRIBE_JOBS = {}

def process_transcribe_task(job_id: str, audio_path: Path):
    job = TRANSCRIBE_JOBS[job_id]
    try:
        job["status"] = "Memuat model FireRedASR2S di GPU T4..."
        job["progress"] = 15
        system = setup_fireredasr()
        if not system:
            raise Exception("Gagal memuat FireRedAsr2System di GPU")

        job["status"] = "FireRedASR2S mentranskripsi audio (VAD + ASR + Punc)..."
        job["progress"] = 35

        res = system.process(str(audio_path))

        sentences = res.get("sentences", [])
        cues = []
        srt_lines = []
        full_text_parts = []

        if sentences:
            for idx, s in enumerate(sentences, 1):
                start_sec = float(s.get("start_ms", 0)) / 1000.0
                end_sec = float(s.get("end_ms", 0)) / 1000.0
                txt = str(s.get("text", "")).strip()
                if txt:
                    cues.append({
                        "index": idx,
                        "start": round(start_sec, 3),
                        "end": round(end_sec, 3),
                        "text": txt
                    })
                    srt_lines.append(f"{idx}\n{fmt_srt_time(start_sec)} --> {fmt_srt_time(end_sec)}\n{txt}\n")
                    full_text_parts.append(txt)

        if not cues and res.get("text"):
            full_txt = str(res.get("text", "")).strip()
            cues.append({"index": 1, "start": 0.0, "end": 5.0, "text": full_txt})
            srt_lines.append(f"1\n00:00:00,000 --> 00:00:05,000\n{full_txt}\n")
            full_text_parts.append(full_txt)

        job["state"] = "COMPLETED"
        job["progress"] = 100
        job["status"] = f"Transkripsi selesai! ({len(cues)} kalimat terdeteksi)"
        job["result"] = {
            "status": "success",
            "text": "".join(full_text_parts) if full_text_parts else res.get("text", ""),
            "cues": cues,
            "srt": "\n".join(srt_lines)
        }
    except Exception as e:
        import traceback
        traceback.print_exc()
        job["state"] = "ERROR"
        job["status"] = f"Error transkripsi di GPU: {str(e)[:120]}"
        job["error"] = str(e)
    finally:
        if audio_path.exists():
            try: audio_path.unlink()
            except: pass

@app.post("/api/transcribe")
def start_transcribe_job(background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    job_id = uuid.uuid4().hex[:8]
    tmp_in  = WORK_DIR / f"in_{job_id}_{file.filename}"
    tmp_wav = WORK_DIR / f"audio_{job_id}.wav"

    with open(tmp_in, "wb") as f_out:
        shutil.copyfileobj(file.file, f_out)

    subprocess.run([
        "ffmpeg", "-y", "-i", str(tmp_in),
        "-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1",
        str(tmp_wav)
    ], stdout=subprocess.DEVNULL, stderr=subprocess.PIPE, check=True)

    if tmp_in.exists():
        try: tmp_in.unlink()
        except: pass

    TRANSCRIBE_JOBS[job_id] = {
        "job_id": job_id,
        "state": "PROCESSING",
        "progress": 10,
        "status": "Audio diterima di Cloud GPU, memulai transkripsi...",
        "created_at": time.time()
    }
    background_tasks.add_task(process_transcribe_task, job_id, tmp_wav)
    return {"job_id": job_id, "status": "PROCESSING", "message": "Tugas transkripsi FireRedASR2S dimulai"}

@app.get("/api/transcribe_job/{job_id}")
def get_transcribe_job(job_id: str):
    if job_id not in TRANSCRIBE_JOBS:
        raise HTTPException(status_code=404, detail="Job transkripsi tidak ditemukan")
    return TRANSCRIBE_JOBS[job_id]

LAST_ACTIVITY = time.time()
IDLE_TIMEOUT_SECONDS = 300  # 5 Menit tanpa aktivitas -> otomatis matikan GPU!

@app.middleware("http")
async def update_activity_middleware(request, call_next):
    global LAST_ACTIVITY
    # Kecualikan polling ping agar polling status tidak menipu timer watchdog
    if not request.url.path.endswith("/ping") and not request.url.path.endswith("/docs"):
        LAST_ACTIVITY = time.time()
    response = await call_next(request)
    return response

@app.get("/api/job/ping")
def ping_status():
    global LAST_ACTIVITY
    idle_for = int(time.time() - LAST_ACTIVITY)
    return {
        "status": "online",
        "idle_seconds": idle_for,
        "auto_shutdown_in": max(0, IDLE_TIMEOUT_SECONDS - idle_for)
    }

@app.post("/api/shutdown")
def api_shutdown():
    def kill_worker():
        time.sleep(1)
        print("\n[STOP] Menerima sinyal remote shutdown! Mematikan Kaggle Worker sekarang...")
        os._exit(0)

    import threading
    threading.Thread(target=kill_worker, daemon=True).start()
    return {"status": "ok", "message": "Worker Kaggle dimatikan! Kuota GPU berhenti dihitung."}

def idle_watchdog_loop():
    global LAST_ACTIVITY
    print(f"\n[WATCHDOG] Aktif: Jika worker menganggur selama {IDLE_TIMEOUT_SECONDS//60} menit, GPU akan otomatis dimatikan sendiri!\n")
    while True:
        time.sleep(15)
        # Jangan matikan jika ada job kompresi atau transkripsi yang sedang jalan
        is_compressing = any(j.get("state") in ["PROCESSING", "DOWNLOADING"] for j in list(JOBS.values()))
        is_transcribing = any(t.get("state") in ["PROCESSING"] for t in list(TRANSCRIBE_JOBS.values()))
        
        if is_compressing or is_transcribing:
            LAST_ACTIVITY = time.time()
            continue

        idle_sec = time.time() - LAST_ACTIVITY
        if idle_sec >= IDLE_TIMEOUT_SECONDS:
            print(f"\n==================================================================")
            print(f"🛑 [AUTO-SHUTDOWN] Menganggur selama {int(idle_sec)} detik tanpa aktivitas!")
            print(f"   Mematikan proses Python sekarang agar kuota GPU Kaggle tidak habis!")
            print(f"==================================================================\n")
            os._exit(0)

if __name__ == "__main__":
    print("\n" + "="*60)
    print(" MENGHUBUNGKAN CLOUDFLARE TUNNEL KE INTERNET...")
    print("="*60)
    public_url = try_cloudflare(port=8000)

    raw_str = str(public_url)
    m_url = re.search(r'https://[a-zA-Z0-9-]+\.trycloudflare\.com', raw_str)
    tunnel_str = m_url.group(0) if m_url else raw_str

    print("\n" + " "*15)
    print(" KAGGLE CLOUD WORKER AKTIF & SIAP MENERIMA PERINTAH!")
    print(f" URL TUNNEL WORKER: {tunnel_str}")
    print(" "*15 + "\n")

    import threading
    def publish_to_cloud_relay(t_url):
        for _ in range(300):
            try:
                requests.post("https://ntfy.sh/ziranaijav_cloud_worker_2026", data=t_url, timeout=10)
            except:
                pass
            time.sleep(30)

    threading.Thread(target=publish_to_cloud_relay, args=(tunnel_str,), daemon=True).start()
    threading.Thread(target=idle_watchdog_loop, daemon=True).start()
    uvicorn.run(app, host="0.0.0.0", port=8000)
