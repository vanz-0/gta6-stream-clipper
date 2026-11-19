# GTA 6 & Gaming Stream Shorts Clipper

An automated AI-powered pipeline to turn long gameplay live streams and VODs into high-retention 9:16 vertical Shorts, TikToks, and Reels.

---

## What's New & Highlights

- **AI Gamer-Checkup System**: Uses multi-frame spatial clustering with OpenCV YuNet to lock onto the seated streamer with headphones and chair, rejecting in-game characters and cutscene faces to eliminate duplicated gameplay.
- **Auto-Switching Layouts**:
  - `auto` (Default): Automatically verifies if a streamer cam is present. If found, renders a 60/40 split (gameplay top, streamer bottom); if not found, falls back cleanly to a single-panel gameplay reel without fake boxes.
  - `facecam`: Enforces split screen mode.
  - `blur`: Enforces single-panel gameplay over a 30px blurred background.
- **Faster-Whisper AI Subtitles**: Ultra-fast local transcription with word-level timestamps.
- **Dynamic Sabs Subtitles**: High-impact 3-word cadence with vibrant yellow (`#FFDE00`) active word highlighting and black outlines.
- **Built-in Swear Filter**: Automatically censors profanities (`f**k`, `sh*t`, `b*tch`, etc.) to protect your posts from platform shadowbans or demonetization.
- **EBU R128 Broadcast Audio Normalization**: Applies `-14 LUFS` loudness normalization for crystal-clear mobile playback.
- **Complete Post Kits**: Generates the 1080×1920 MP4, cover thumbnail JPG, social caption text file, and JSON metadata with zero intermediate file bloat.

---

## Architecture Overview

```
                      Stream URL / Video File
                                 |
                    [yt-dlp Section Extractor]
                                 |
              +------------------+------------------+
              |                                     |
        [Audio Track]                         [Video Track]
              |                                     |
    [Faster-Whisper AI]                    [Gamer-Checkup]
              |                         (YuNet Spatial Cluster)
   [Word Timestamps & Tokens]                       |
              |                        Verified / Not Detected?
    [Swear Filter (*)]                         /        \
              |                      [60/40 Split]   [Single Gameplay]
   [ASS Sabs Subtitles]                        \        /
              |                           [1080x1920 9:16]
              +------------------+------------------+
                                 |
                     [FFmpeg Filter Complex]
                                 |
                     +-----------+-----------+
                     |                       |
                [Video MP4]             [Cover JPG]
                [Post TXT]              [Meta JSON]
```

---

## Quickstart

### 1. Requirements

- Python 3.10+
- FFmpeg installed and available on system PATH (or in `bin/`)
- Git

### 2. Installation

Clone this repository and run the setup script:

```bash
git clone https://github.com/vanz-0/gta6-stream-clipper.git
cd gta6-stream-clipper
python setup.py
```

The setup script verifies FFmpeg, installs Python dependencies, and downloads the YuNet face detection ONNX model.

---

## Usage Examples

### 1. Clip Directly from a Stream URL

```bash
python clipper.py --url "https://www.youtube.com/watch?v=EXAMPLE" --ss 120 --dur 25 --hook "UNREAL MOMENT!!"
```

### 2. Clip from a Local File

```bash
python clipper.py --file "stream_vod.mp4" --ss 450 --dur 25 --hook "THE GRAPHICS ARE UNREAL!!"
```

### 3. Force Specific Layout Mode

```bash
# Force single-panel gameplay over blurred background:
python clipper.py --file "gameplay.mp4" --ss 765 --dur 25 --layout blur --hook "FIRST CAR THEFT!!"

# Force split-screen facecam mode:
python clipper.py --url "https://www.youtube.com/watch?v=EXAMPLE" --ss 3680 --dur 25 --layout facecam --hook "FIRST PERSON MODE!!"
```

---

## Command Line Arguments

| Flag | Type | Default | Description |
|---|---|---|---|
| `--url` | string | `None` | YouTube or Twitch video URL |
| `--file` | string | `None` | Local video file path |
| `--ss` | float | `0.0` | Start offset time in seconds |
| `--dur` | float | `25.0` | Duration of the clip in seconds |
| `--hook` | string | `"GTA 6 IS ACTUALLY INSANE!!"` | Hook title badge text |
| `--slug` | string | `"gta6_viral_short"` | Output filename prefix |
| `--caption` | string | `None` | Custom social media caption |
| `--layout` | string | `auto` | Layout mode: `auto`, `facecam`, or `blur` |
| `--no-censor` | flag | `False` | Disable automatic swear word censoring |
| `--output` | string | `output/<date>/` | Destination folder for export files |

---

## License

This project is licensed under the MIT License.
