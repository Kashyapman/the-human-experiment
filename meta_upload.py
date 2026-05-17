"""
THE HUMAN EXPERIMENT — Social Broadcaster
==========================================
Handles Facebook Reels and Instagram Reels publishing.

Instagram upload architecture (3-stage):
  Stage 1 : Upload video to a temporary public host (file.io → Catbox fallback)
            Instagram's container API requires a publicly accessible URL —
            it cannot accept a direct file upload.
  Stage 2 : Create an IG media container using the public URL.
            Meta processes the video server-side (takes 30s–5min).
  Stage 3 : Poll container status every 10s until FINISHED, then publish.

Facebook upload architecture (direct):
  Single POST with the video file as multipart form data.
  No intermediate hosting required.

Design:
  Every function returns a bool (success/failure).
  No function raises exceptions — all errors are caught, logged, and returned as False.
  This ensures the pipeline continues even if social uploads fail.
"""

import os
import time
import requests

# ═══════════════════════════════════════════════════════════════════════════
#  CREDENTIALS  (loaded from GitHub Actions secrets)
# ═══════════════════════════════════════════════════════════════════════════
ACCESS_TOKEN = os.environ.get("META_ACCESS_TOKEN", "")
FB_PAGE_ID   = os.environ.get("FB_PAGE_ID",        "")
IG_USER_ID   = os.environ.get("IG_USER_ID",        "")

GRAPH_VERSION = "v19.0"
GRAPH_BASE    = f"https://graph.facebook.com/{GRAPH_VERSION}"

# User-Agent for hosting service requests
_UA = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36"


# ═══════════════════════════════════════════════════════════════════════════
#  FACEBOOK REELS  — direct binary upload
# ═══════════════════════════════════════════════════════════════════════════

def upload_to_facebook(video_path: str, caption: str) -> bool:
    """
    Uploads a video file directly to Facebook Reels via the Graph API.

    Parameters
    ----------
    video_path : Local path to the rendered .mp4 file.
    caption    : Full post caption including hashtags.

    Returns True on success, False on any failure.
    """
    print("📘 Uploading to Facebook Reels...")

    if not ACCESS_TOKEN or not FB_PAGE_ID:
        print("  ❌ Missing credentials: META_ACCESS_TOKEN or FB_PAGE_ID not set.")
        return False

    if not os.path.exists(video_path):
        print(f"  ❌ Video file not found: {video_path}")
        return False

    url     = f"{GRAPH_BASE}/{FB_PAGE_ID}/videos"
    payload = {
        "description":  caption,
        "access_token": ACCESS_TOKEN,
    }

    try:
        with open(video_path, "rb") as vf:
            response = requests.post(
                url,
                data    = payload,
                files   = {"source": vf},
                timeout = 180,
            )

        result = response.json()

        if "id" in result:
            print(f"  ✅ Facebook upload success. Video ID: {result['id']}")
            return True

        print(f"  ❌ Facebook upload rejected: {result}")
        return False

    except requests.exceptions.Timeout:
        print("  ❌ Facebook upload timed out (180s).")
        return False
    except Exception as e:
        print(f"  ❌ Facebook upload error: {e}")
        return False


# ═══════════════════════════════════════════════════════════════════════════
#  TEMPORARY PUBLIC URL  — required for Instagram container API
# ═══════════════════════════════════════════════════════════════════════════

def get_temp_public_url(file_path: str) -> str | None:
    """
    Uploads the video to a temporary public host and returns the URL.

    Method 1 — file.io:
        Single-use, auto-expiring, bot-friendly.
        Returns a direct download link immediately.
        Preferred: cleanest integration with Meta's container API.

    Method 2 — Catbox (fallback):
        Persistent file host with no expiry.
        Used if file.io fails or returns a non-success response.

    Returns the public URL string, or None if both methods fail.
    """
    print("  ☁️  Getting temporary public URL for Instagram...")

    if not os.path.exists(file_path):
        print(f"  ❌ File not found: {file_path}")
        return None

    file_size_mb = os.path.getsize(file_path) / (1024 * 1024)
    print(f"  📦 File size: {file_size_mb:.1f} MB")

    # ── Method 1: file.io ────────────────────────────────────────────────
    print("  ☁️  Trying file.io...")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                "https://file.io",
                files   = {"file": f},
                headers = {"User-Agent": _UA},
                timeout = 120,
            )
        if response.status_code == 200:
            data = response.json()
            if data.get("success") and data.get("link"):
                url = data["link"]
                print(f"  ✅ file.io success: {url}")
                return url
        print(f"  ⚠️  file.io returned {response.status_code}: {response.text[:150]}")
    except requests.exceptions.Timeout:
        print("  ⚠️  file.io timed out.")
    except Exception as e:
        print(f"  ⚠️  file.io error: {e}")

    # ── Method 2: Catbox fallback ────────────────────────────────────────
    print("  ☁️  Falling back to Catbox...")
    try:
        with open(file_path, "rb") as f:
            response = requests.post(
                "https://catbox.moe/user/api.php",
                data    = {"reqtype": "fileupload"},
                files   = {"fileToUpload": f},
                headers = {"User-Agent": _UA},
                timeout = 120,
            )
        if response.status_code == 200:
            text = response.text.strip()
            if text.startswith("http"):
                print(f"  ✅ Catbox success: {text}")
                return text
        print(f"  ❌ Catbox failed: {response.text[:150]}")
    except requests.exceptions.Timeout:
        print("  ❌ Catbox timed out.")
    except Exception as e:
        print(f"  ❌ Catbox error: {e}")

    print("  ❌ All hosting methods failed. Instagram upload skipped.")
    return None


# ═══════════════════════════════════════════════════════════════════════════
#  INSTAGRAM REELS  — 3-stage container flow
# ═══════════════════════════════════════════════════════════════════════════

def upload_to_instagram(video_url: str, caption: str) -> bool:
    """
    Publishes a video to Instagram Reels via the Graph API.

    The Instagram API does not accept direct file uploads.
    It requires a publicly accessible URL (from get_temp_public_url()).

    Stage 1 — Create container:
        POST to /{ig-user-id}/media with video_url + caption.
        Meta begins downloading and processing the video server-side.
        Returns a creation_id to track the job.

    Stage 2 — Poll until ready:
        GET /{creation-id}?fields=status_code every 10 seconds.
        Status values: IN_PROGRESS → FINISHED | ERROR | EXPIRED
        Max wait: 30 attempts × 10s = 5 minutes.

    Stage 3 — Publish:
        POST to /{ig-user-id}/media_publish with creation_id.
        Returns the final published post ID.

    Parameters
    ----------
    video_url : Publicly accessible video URL (from get_temp_public_url).
    caption   : Full post caption including hashtags.

    Returns True on successful publish, False on any failure.
    """
    print("📸 Uploading to Instagram Reels...")

    if not ACCESS_TOKEN or not IG_USER_ID:
        print("  ❌ Missing credentials: META_ACCESS_TOKEN or IG_USER_ID not set.")
        return False

    if not video_url or not video_url.startswith("http"):
        print(f"  ❌ Invalid video URL: {video_url}")
        return False

    # ── Stage 1: Create media container ─────────────────────────────────
    print("  ⚙️  Stage 1: Creating media container...")
    try:
        container_response = requests.post(
            f"{GRAPH_BASE}/{IG_USER_ID}/media",
            data = {
                "media_type":   "REELS",
                "video_url":    video_url,
                "caption":      caption,
                "access_token": ACCESS_TOKEN,
            },
            timeout = 30,
        )
        container_data = container_response.json()
    except Exception as e:
        print(f"  ❌ Container creation request failed: {e}")
        return False

    if "id" not in container_data:
        print(f"  ❌ Container creation failed: {container_data}")
        return False

    creation_id = container_data["id"]
    print(f"  ✅ Container created. ID: {creation_id}")

    # ── Stage 2: Poll until FINISHED ─────────────────────────────────────
    print("  ⏳ Stage 2: Waiting for Meta to process video...")
    max_polls    = 30       # 30 × 10s = 5 minutes max
    poll_interval = 10      # seconds between checks

    for poll_count in range(1, max_polls + 1):
        time.sleep(poll_interval)
        try:
            status_response = requests.get(
                f"{GRAPH_BASE}/{creation_id}",
                params  = {"fields": "status_code", "access_token": ACCESS_TOKEN},
                timeout = 15,
            )
            status_data = status_response.json()
        except Exception as e:
            print(f"  ⚠️  Poll {poll_count} request error: {e}")
            continue

        status = status_data.get("status_code", "UNKNOWN")
        print(f"  🔄 Poll {poll_count}/{max_polls}: {status}")

        if status == "FINISHED":
            print("  ✅ Video processing complete.")
            break
        elif status in ("ERROR", "EXPIRED"):
            print(f"  ❌ Instagram processing failed with status: {status}")
            print(f"     Full response: {status_data}")
            return False
        # IN_PROGRESS or UNKNOWN → continue polling
    else:
        print(f"  ❌ Timed out after {max_polls * poll_interval}s waiting for processing.")
        return False

    # ── Stage 3: Publish ─────────────────────────────────────────────────
    print("  🚀 Stage 3: Publishing...")
    try:
        publish_response = requests.post(
            f"{GRAPH_BASE}/{IG_USER_ID}/media_publish",
            data = {
                "creation_id":  creation_id,
                "access_token": ACCESS_TOKEN,
            },
            timeout = 30,
        )
        publish_data = publish_response.json()
    except Exception as e:
        print(f"  ❌ Publish request failed: {e}")
        return False

    if "id" in publish_data:
        print(f"  ✅ Instagram publish success. Post ID: {publish_data['id']}")
        return True

    print(f"  ❌ Instagram publish failed: {publish_data}")
    return False
