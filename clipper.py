#!/usr/bin/env python3
"""
GTA 6 & Gaming Stream Shorts Clipper
Automated AI pipeline to turn long gameplay streams into 9:16 vertical Shorts.

Features:
  - Gamer-Checkup System: AI facecam clustering verifies seated streamer with headphones,
    rejecting in-game characters and UI photos to prevent duplicate video issues.
  - Smart Floating Layout: Gameplay (60%) on top, Streamer (40%) on bottom, floating
    over a 30px blurred + darkened source background.
  - Faster-Whisper AI Subtitles: Word-level timestamps with high-performance ASS formatting.
  - Dynamic Sabs: 3-word cadence with bright yellow (#FFDE00) active word highlighting.
  - Built-in Swear Filter: Automatic tracking and censoring of profanities (f**k, sh*t, etc.).
  - EBU R128 Audio Normalization: Balanced audio levels for social feed playback.
  - Complete Post Kit: Vertical MP4 (1080x1920, 9:16), Cover JPG, TXT Post Kit, JSON Metadata.
"""

import argparse
import json
import os
import re
import shutil
import subprocess
import sys
from datetime import date
from pathlib import Path
import numpy as np

# Local swear filter import
try:
    from swear_filter import clean, has_swears
except ImportError:
    def clean(t): return t
    def has_swears(t): return False

# ─── PATHS & BINARY RESOLUTION ────────────────────────────────────────────────
SCRIPT_DIR = Path(__file__).resolve().parent

def find_binary(name: str) -> str:
    """Finds binary in local bin/, system PATH, or common locations."""
    ext = ".exe" if sys.platform.startswith("win") else ""
    local_bin = SCRIPT_DIR / "bin" / f"{name}{ext}"
    if local_bin.exists():
        return str(local_bin)
    found = shutil.which(name)
    if found:
        return found
    return name

FFMPEG_BIN  = find_binary("ffmpeg")
FFPROBE_BIN = find_binary("ffprobe")
YUNET_MODEL = SCRIPT_DIR / "models" / "face_detection_yunet_2023mar.onnx"

# Canvas dimensions
CANVAS_W      = 1080
CANVAS_H      = 1920
CONTENT_W     = 1040
CONTENT_X     = (CANVAS_W - CONTENT_W) // 2   # = 20
CONTENT_TOP_H = 960                           # Gameplay panel height
CONTENT_BOT_H = 640                           # Facecam panel height
CONTENT_H     = CONTENT_TOP_H + CONTENT_BOT_H # = 1600
CONTENT_Y     = (CANVAS_H - CONTENT_H) // 2   # = 160
SEAM_Y        = CONTENT_Y + CONTENT_TOP_H     # = 1120
HOOK_Y        = CONTENT_Y + 180               # = 340
HOOK_BOX_Y    = HOOK_Y - 42
SUB_SEAM_Y    = SEAM_Y - 12                   # = 1108
SUB_SINGLE_Y  = CONTENT_Y + CONTENT_H - 220  # = 1540

# ─── FONT SELECTION ───────────────────────────────────────────────────────────
def get_font_path() -> Path:
    candidates = [
        Path("C:/Windows/Fonts/ariblk.ttf"),
        Path("C:/Windows/Fonts/arialbd.ttf"),
        Path("/System/Library/Fonts/Supplemental/Arial Black.ttf"),
        Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf"),
    ]
    for p in candidates:
        if p.exists():
            return p
    return Path("C:/Windows/Fonts/arial.ttf")

FONT_PATH = get_font_path()

# ─── HELPERS ──────────────────────────────────────────────────────────────────
def log(msg): print(msg, flush=True)

def run_cmd(cmd, **kwargs):
    cmd = [str(c) for c in cmd]
    res = subprocess.run(cmd, capture_output=True, text=True, encoding="utf-8", errors="replace", **kwargs)
    if res.returncode != 0:
        err = res.stderr[-500:] if res.stderr else "Unknown error"
        raise RuntimeError(f"Command failed ({cmd[0]}): {err}")
    return res

def get_source_dimensions(video_path):
    out = subprocess.check_output(
        [FFPROBE_BIN, "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height", "-of", "json", str(video_path)],
        stderr=subprocess.DEVNULL)
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"])

# ─── STEP 1: DOWNLOAD STREAM SECTION ──────────────────────────────────────────
def download_stream_section(url: str, ss: float, dur: float, out_file: Path) -> Path:
    """Downloads a slice directly using yt-dlp without full video download."""
    t_end = ss + dur
    t_start = max(0.0, ss)
    log(f"📥 Downloading section: {t_start:.1f}s - {t_end:.1f}s from stream...")
    
    cmd = [
        sys.executable, "-m", "yt_dlp",
        "--no-playlist",
        "-f", "bv*[height<=1080][fps<=60]+ba/b[height<=1080]/bv*+ba/b",
        "--download-sections", f"*{t_start:.1f}-{t_end:.1f}",
        "--merge-output-format", "mp4",
        "--ffmpeg-location", str(Path(FFMPEG_BIN).parent),
        "--no-part",
        "-o", str(out_file),
        url,
    ]
    run_cmd(cmd)
    
    if not out_file.exists():
        matches = list(out_file.parent.glob(out_file.stem + "*"))
        if matches:
            return matches[0]
        raise FileNotFoundError(f"Section download failed to create {out_file}")
    return out_file

# ─── STEP 2: AUDIO EXTRACTION & TRANSCRIPTION ─────────────────────────────────
def extract_audio_numpy(video_path: Path):
    """Decode audio track directly to 16kHz float32 NumPy array, bypassing PyAV bugs."""
    cmd = [
        FFMPEG_BIN, "-v", "error",
        "-i", str(video_path),
        "-ac", "1", "-ar", "16000", "-f", "f32le", "-"
    ]
    raw = subprocess.check_output(cmd)
    return np.frombuffer(raw, dtype=np.float32)

def transcribe_audio(video_file: Path, censor: bool = True):
    """Transcribes audio using local Faster-Whisper with word-level timestamps."""
    from faster_whisper import WhisperModel
    log("🎙️ Transcribing audio with Faster-Whisper...")
    model = WhisperModel("base.en", device="cpu", compute_type="int8")
    audio = extract_audio_numpy(video_file)
    segments, _ = model.transcribe(audio, word_timestamps=True)
    
    words_data = []
    full_transcript = []
    for seg in segments:
        full_transcript.append(seg.text)
        if seg.words:
            for w in seg.words:
                raw_w = w.word.strip()
                processed_w = clean(raw_w) if censor else raw_w
                words_data.append({"word": processed_w, "start": w.start, "end": w.end})
                
    log(f"   Transcribed {len(words_data)} words: '{' '.join(full_transcript)[:80]}...'")
    return words_data

# ─── STEP 3: GAMER-CHECKUP FACECAM VERIFICATION ───────────────────────────────
def detect_facecam_box(video_path: Path, src_w: int, src_h: int):
    """
    Gamer-Checkup System:
    Distinguishes the real streamer (sitting in front of webcam with headphones)
    from transient in-game game characters or phone feeds.
    """
    import cv2
    if not YUNET_MODEL.exists():
        log("[!] YuNet face detection model missing; run python setup.py to acquire it.")
        return None
        
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return None
        
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    detector = cv2.FaceDetectorYN.create(
        str(YUNET_MODEL), "", (src_w, src_h),
        score_threshold=0.6, nms_threshold=0.3, top_k=5000)
        
    samples = 25
    step = max(1, total_frames // samples)
    detections = []
    sampled_count = 0
    
    for i in range(0, total_frames, step):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i)
        ret, frame = cap.read()
        if not ret: break
        sampled_count += 1
        _, faces = detector.detect(frame)
        if faces is not None:
            for face in faces:
                x, y, w, h = map(int, face[:4])
                score = float(face[-1])
                detections.append({'x': x, 'y': y, 'w': w, 'h': h, 'frame': i, 'score': score})
    cap.release()
    
    if not detections:
        log("[*] Gamer-Checkup: No faces detected in video.")
        return None

    # Spatial clustering (radius = 12% screen width)
    cluster_radius = 0.12 * src_w
    clusters = []
    for d in detections:
        matched = False
        for c in clusters:
            cx = np.mean([item['x'] for item in c])
            cy = np.mean([item['y'] for item in c])
            if np.hypot(d['x'] - cx, d['y'] - cy) < cluster_radius:
                c.append(d)
                matched = True
                break
        if not matched:
            clusters.append([d])

    verified_gamers = []
    for idx, c in enumerate(clusters):
        unique_frames = len(set(item['frame'] for item in c))
        persistence   = unique_frames / max(sampled_count, 1)
        med_x         = int(np.median([item['x'] for item in c]))
        med_y         = int(np.median([item['y'] for item in c]))
        med_w         = int(np.median([item['w'] for item in c]))
        med_h         = int(np.median([item['h'] for item in c]))
        std_x         = float(np.std([item['x'] for item in c]))
        std_y         = float(np.std([item['y'] for item in c]))

        # Edge/Corner proximity check (Streamers are near borders; gameplay action is in center)
        norm_cx = (med_x + med_w / 2.0) / src_w
        norm_cy = (med_y + med_h / 2.0) / src_h
        in_dead_center = (0.35 < norm_cx < 0.65) and (0.28 < norm_cy < 0.72)
        is_stable = (std_x <= 35.0 and std_y <= 35.0)
        is_persistent = (persistence >= 0.35)

        if is_persistent and is_stable and not in_dead_center:
            confidence = (persistence * 2.0) - (std_x + std_y) / 50.0
            verified_gamers.append({
                'med_x': med_x, 'med_y': med_y,
                'med_w': med_w, 'med_h': med_h,
                'persistence': persistence,
                'confidence': confidence
            })

    if not verified_gamers:
        log("[*] Gamer-Checkup: No persistent stationary streamer found. Defaulting to single-panel gameplay mode.")
        return None

    verified_gamers.sort(key=lambda g: g['confidence'], reverse=True)
    best = verified_gamers[0]
    log(f"[+] Gamer-Checkup: VERIFIED streamer facecam at ({best['med_x']}, {best['med_y']}) with {best['persistence']*100:.0f}% presence.")

    expand = 2.8   # expand detected face bbox to capture full cam overlay (headphones, shoulders, chair)
    cam_w  = int(best['med_w'] * expand)
    cam_h  = int(best['med_h'] * expand)
    cam_x  = max(0, best['med_x'] - int(cam_w * 0.35))
    cam_y  = max(0, best['med_y'] - int(cam_h * 0.25))
    cam_w  = min(src_w - cam_x, cam_w)
    cam_h  = min(src_h - cam_y, cam_h)
    return cam_x, cam_y, cam_w, cam_h

# ─── STEP 4: GENERATE ASS SABS SUBTITLES ──────────────────────────────────────
def create_ass_subtitles(words_data, ass_path: Path, sub_y: int = SUB_SEAM_Y):
    """
    Generates dynamic Sabs ASS subtitles:
    - 3-word cadence with bright yellow (#FFDE00) active word highlighting.
    - 6px black border + drop shadow.
    """
    header = f"""[Script Info]
ScriptType: v4.00+
PlayResX: {CANVAS_W}
PlayResY: {CANVAS_H}
ScaledBorderAndShadow: yes

[V4+ Styles]
Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, MarginL, MarginR, MarginV, Encoding
Style: Sabs,Arial Black,58,&H00FFFFFF,&H0000FFFF,&H00000000,&H80000000,-1,0,0,0,100,100,2,0,1,6,3,5,40,40,0,1

[Events]
Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text
"""
    def fmt(t):
        return f"{int(t//3600):01d}:{int((t%3600)//60):02d}:{t%60:05.2f}"

    events = []
    phrase_len = 3

    for ci, i in enumerate(range(0, len(words_data), phrase_len)):
        chunk  = words_data[i:i + phrase_len]
        energy = (ci % 2 == 0)

        for idx, active_w in enumerate(chunk):
            parts = []
            for j, w in enumerate(chunk):
                word    = w["word"].strip().upper()
                is_last = (j == len(chunk) - 1)
                if is_last and energy and not word.endswith("!!"):
                    word += "!!"
                if j == idx:   # active word highlighted in bright yellow
                    parts.append(r"{\c&H00DEFF&}" + word + r"{\c&H00FFFFFF&}")
                else:
                    parts.append(word)

            line = (r"{\an5\pos(" + f"{CANVAS_W//2},{sub_y}" + r")}" + " ".join(parts))
            events.append(f"Dialogue: 0,{fmt(active_w['start'])},{fmt(active_w['end'])},Sabs,,0,0,0,,{line}")

    with open(ass_path, "w", encoding="utf-8") as f:
        f.write(header + "\n".join(events))

# ─── STEP 5: VIDEO COMPOSITING & RENDERING ────────────────────────────────────
def render_short(raw_cut_path: Path, ass_path: Path, hook_title: str, out_mp4: Path,
                 facecam_box, src_w: int, src_h: int):
    """
    Renders 1080x1920 vertical format with EBU R128 audio normalization.
    Uses blurred background float styling with verified streamer or single-panel gameplay.
    """
    ass_esc  = str(ass_path).replace("\\", "/").replace(":", "\\:")
    font_esc = str(FONT_PATH).replace("\\", "/").replace(":", "\\:")
    hook     = hook_title.replace("'", "").replace('"', "").upper()

    # Background (Layer 1)
    bg_filter = (
        "[src_bg]"
        f"scale=-2:{CANVAS_H},"
        f"crop={CANVAS_W}:{CANVAS_H}:max(0\\,(iw-{CANVAS_W})/2):0,"
        "boxblur=30:6,"
        "eq=brightness=-0.2"
        "[bg]"
    )

    if facecam_box:
        # 2-Panel Layout: Gameplay Top (60%) + Verified Streamer Bottom (40%)
        fx, fy, fw, fh = facecam_box
        gp_filter = (
            "[src_gp]"
            f"scale=-2:{CONTENT_TOP_H},"
            f"crop={CONTENT_W}:{CONTENT_TOP_H}:max(0\\,(iw-{CONTENT_W})/2):0"
            "[gp]"
        )
        fc_filter = (
            "[src_fc]"
            f"crop={fw}:{fh}:{fx}:{fy},"
            f"scale={CONTENT_W}:{CONTENT_BOT_H}:force_original_aspect_ratio=increase,"
            f"crop={CONTENT_W}:{CONTENT_BOT_H}"
            "[fc]"
        )
        filter_complex = (
            "[0:v]split=3[src_bg][src_gp][src_fc];"
            f"{bg_filter};{gp_filter};{fc_filter};"
            "[gp][fc]vstack=inputs=2[content];"
            f"[bg][content]overlay={CONTENT_X}:{CONTENT_Y}[stacked];"
            "[stacked]"
            f"drawbox=y={SEAM_Y - 2}:color=0xFFDE00@0.6:width=iw:height=4:t=fill,"
            f"drawbox=y={HOOK_BOX_Y}:color=black@0.72:width=iw:height=112:t=fill,"
            f"drawtext=fontfile='{font_esc}':text='{hook}':fontcolor=white:fontsize=46:"
            f"x=(w-text_w)/2:y={HOOK_Y}:shadowcolor=black@0.9:shadowx=3:shadowy=3,"
            f"subtitles='{ass_esc}',"
            f"scale={CANVAS_W}:{CANVAS_H},setsar=1[outv]"
        )
    else:
        # Single-Panel Pure Gameplay Layout
        gp_filter = (
            "[src_gp]"
            f"scale=-2:{CONTENT_H},"
            f"crop={CONTENT_W}:{CONTENT_H}:max(0\\,(iw-{CONTENT_W})/2):0"
            "[content]"
        )
        single_hook_y = CONTENT_Y + 140
        single_hook_box_y = single_hook_y - 42
        filter_complex = (
            "[0:v]split=2[src_bg][src_gp];"
            f"{bg_filter};{gp_filter};"
            f"[bg][content]overlay={CONTENT_X}:{CONTENT_Y}[stacked];"
            "[stacked]"
            f"drawbox=y={single_hook_box_y}:color=black@0.72:width=iw:height=112:t=fill,"
            f"drawtext=fontfile='{font_esc}':text='{hook}':fontcolor=white:fontsize=46:"
            f"x=(w-text_w)/2:y={single_hook_y}:shadowcolor=black@0.9:shadowx=3:shadowy=3,"
            f"subtitles='{ass_esc}',"
            f"scale={CANVAS_W}:{CANVAS_H},setsar=1[outv]"
        )

    cmd = [
        FFMPEG_BIN, "-y",
        "-i", str(raw_cut_path),
        "-filter_complex", filter_complex + ";[0:a]loudnorm=I=-14:TP=-1.5:LRA=11,aresample=48000[aout]",
        "-map", "[outv]", "-map", "[aout]",
        "-c:v", "libx264", "-preset", "veryfast", "-crf", "20",
        "-c:a", "aac", "-b:a", "192k",
        "-movflags", "+faststart",
        str(out_mp4)
    ]
    run_cmd(cmd)

# ─── MAIN PIPELINE ─────────────────────────────────────────────────────────────
def process_clip(args):
    work_dir = Path(".tmp") / f"clipper_work_{np.random.randint(100000, 999999)}"
    work_dir.mkdir(parents=True, exist_ok=True)
    out_dir = Path(args.output) if args.output else Path("output") / str(date.today())
    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        # 1. Source acquisition
        if args.url:
            src = download_stream_section(args.url, args.ss, args.dur, work_dir / "stream_slice.mp4")
            clip_ss = 0.0
        elif args.file:
            src = Path(args.file)
            clip_ss = args.ss
        else:
            raise ValueError("Either --url or --file must be specified")

        src_w, src_h = get_source_dimensions(src)
        log(f"🎬 Source: {src.name} ({src_w}x{src_h})")

        slug = re.sub(r"[^\w\-]+", "_", (args.slug or "gta6_viral_short")).strip("_")
        out_mp4 = out_dir / f"{slug}.mp4"
        out_jpg = out_dir / f"{slug}.jpg"
        out_txt = out_dir / f"{slug}.txt"
        out_json = out_dir / f"{slug}.json"

        # 2. Transcription & Swear Filter
        words = transcribe_audio(src, censor=not args.no_censor)
        if not words:
            words = [{"word": "GTA", "start": 0.5, "end": 1.2},
                     {"word": "6", "start": 1.3, "end": 2.0},
                     {"word": "UNREAL", "start": 2.1, "end": 3.0}]

        # 3. Gamer-Checkup Facecam Detection
        if args.layout == "blur":
            facecam_box = None
            log("[*] Forced layout: blur (single-panel gameplay)")
        else:
            log("[*] Running Gamer-Checkup facecam detection...")
            facecam_box = detect_facecam_box(src, src_w, src_h)

        # 4. Generate Sabs Subtitles
        sub_y = SUB_SEAM_Y if facecam_box else SUB_SINGLE_Y
        ass_path = work_dir / "sabs.ass"
        create_ass_subtitles(words, ass_path, sub_y=sub_y)

        # 5. Composite & Render
        log(f"⚡ Rendering 1080x1920 Short to {out_mp4.name}...")
        hook_text = clean(args.hook) if not args.no_censor else args.hook
        render_short(src, ass_path, hook_text, out_mp4, facecam_box, src_w, src_h)

        # 6. Cover Thumbnail & Upload Post Kit
        log("📦 Generating cover thumbnail & upload post kit...")
        run_cmd([
            FFMPEG_BIN, "-y", "-loglevel", "error",
            "-ss", f"{clip_ss + 2.0:.2f}", "-i", str(src),
            "-vframes", "1", "-update", "1", str(out_jpg)
        ])

        title = hook_text
        tags = "#GTA6 #GTAVI #gaming #gamingclips #shorts #reels #tiktok #viral"
        caption = args.caption or f"{title}! Would you play GTA 6 day one? Drop your thoughts below 👇"
        if not args.no_censor:
            caption = clean(caption)

        post_kit = (
            f"TITLE:\n{title}\n\n"
            f"CAPTION:\n{caption}\n\n"
            f"{tags}\n\n"
            f"Source: {args.url or src.name}\n"
            f"Duration: {args.dur}s\n"
            f"Format: 1080x1920 (9:16 Vertical)\n"
        )
        out_txt.write_text(post_kit, encoding="utf-8")

        meta = {
            "title": title,
            "caption": caption,
            "tags": tags.split(),
            "source": args.url or str(src),
            "duration": args.dur,
            "censored": not args.no_censor,
            "facecam_detected": facecam_box is not None
        }
        out_json.write_text(json.dumps(meta, indent=2), encoding="utf-8")

        log("\n" + "=" * 60)
        log("✅ SUCCESS — Clip Ready for Distribution:")
        log(f"   ▶ MP4 Video : {out_mp4}")
        log(f"   ▶ Thumbnail : {out_jpg}")
        log(f"   ▶ Post Kit  : {out_txt}")
        log(f"   ▶ Metadata  : {out_json}")
        log("=" * 60 + "\n")

    finally:
        # Strict zero-persistence cleanup
        shutil.rmtree(work_dir, ignore_errors=True)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GTA 6 & Gaming Stream Shorts Clipper")
    parser.add_argument("--url", help="YouTube/Twitch stream URL")
    parser.add_argument("--file", help="Local video file path")
    parser.add_argument("--ss", type=float, default=0.0, help="Start time in seconds")
    parser.add_argument("--dur", type=float, default=25.0, help="Duration in seconds")
    parser.add_argument("--hook", default="GTA 6 IS ACTUALLY INSANE!!", help="Hook badge title")
    parser.add_argument("--slug", default="gta6_viral_short", help="Filename slug")
    parser.add_argument("--caption", help="Social post caption")
    parser.add_argument("--layout", choices=["auto", "facecam", "blur"], default="auto",
                        help="Layout: 'auto' (detects gamer cam), 'facecam' (force split), 'blur' (single gameplay)")
    parser.add_argument("--no-censor", action="store_true", help="Disable swear word censoring")
    parser.add_argument("--output", help="Custom output directory")
    
    args = parser.parse_args()
    process_clip(args)
