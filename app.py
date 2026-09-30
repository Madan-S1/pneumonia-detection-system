from __future__ import annotations

import base64
import json
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse

from src.config import ASSETS_DIR
from src.inference import predict_image_bytes


import os

HOST = "0.0.0.0"
PORT = int(os.environ.get("PORT", 7860))


INDEX_HTML = """<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Pneumonia Detection | REVA Project</title>
  <style>
    :root {
      --bg: #fff8f2;
      --surface: #ffffff;
      --surface-soft: #fff3e8;
      --ink: #24140b;
      --muted: #735c4e;
      --line: #f0d7c2;
      --orange: #f26a21;
      --orange-dark: #c74c0d;
      --orange-soft: #fff0e4;
      --danger: #b42318;
      --success: #157f45;
      --shadow: 0 18px 48px rgba(151, 72, 19, 0.14);
    }

    * { box-sizing: border-box; }

    body {
      margin: 0;
      min-height: 100vh;
      font-family: Inter, ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
      color: var(--ink);
      background:
        radial-gradient(circle at 15% 10%, rgba(242, 106, 33, 0.16), transparent 28%),
        linear-gradient(180deg, #fffaf6 0%, var(--bg) 46%, #ffffff 100%);
    }

    .page {
      width: min(1180px, calc(100% - 32px));
      margin: 0 auto;
      padding: 22px 0 30px;
    }

    .topbar {
      display: flex;
      align-items: center;
      justify-content: space-between;
      gap: 18px;
      padding: 12px 0 22px;
    }

    .brand {
      display: flex;
      align-items: center;
      gap: 12px;
      min-width: 0;
    }

    .brand-mark {
      width: 44px;
      height: 44px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: #ffffff;
      font-weight: 900;
      background: linear-gradient(135deg, var(--orange), var(--orange-dark));
      box-shadow: 0 10px 24px rgba(242, 106, 33, 0.28);
    }

    .brand-title {
      margin: 0;
      font-size: 16px;
      line-height: 1.15;
    }

    .brand-subtitle {
      margin: 3px 0 0;
      color: var(--muted);
      font-size: 13px;
    }

    .status-pill {
      border: 1px solid var(--line);
      border-radius: 999px;
      padding: 8px 12px;
      color: var(--orange-dark);
      background: rgba(255, 255, 255, 0.78);
      font-weight: 800;
      font-size: 13px;
      white-space: nowrap;
    }

    .hero {
      display: grid;
      grid-template-columns: minmax(0, 1.08fr) minmax(330px, 0.92fr);
      gap: 22px;
      align-items: stretch;
    }

    .intro {
      min-height: 610px;
      border-radius: 8px;
      padding: clamp(24px, 4vw, 42px);
      color: #ffffff;
      background:
        linear-gradient(135deg, rgba(124, 50, 4, 0.88), rgba(242, 106, 33, 0.82)),
        url("data:image/svg+xml,%3Csvg width='900' height='700' viewBox='0 0 900 700' xmlns='http://www.w3.org/2000/svg'%3E%3Cg fill='none' stroke='rgba(255,255,255,.18)' stroke-width='2'%3E%3Cpath d='M520 130c76 39 122 118 122 204 0 126-102 228-228 228S186 460 186 334c0-86 48-164 124-203'/%3E%3Cpath d='M416 122v458M305 214c51 24 82 68 82 121s-31 97-82 121M527 214c-51 24-82 68-82 121s31 97 82 121'/%3E%3Cpath d='M416 288c-44-49-82-54-117-27-39 31-42 117-11 168 35 57 86 39 128 7 42 32 93 50 128-7 31-51 28-137-11-168-35-27-73-22-117 27z'/%3E%3C/g%3E%3C/svg%3E");
      background-size: cover;
      box-shadow: var(--shadow);
      display: flex;
      flex-direction: column;
      justify-content: space-between;
      overflow: hidden;
      position: relative;
    }

    .intro::after {
      content: "";
      position: absolute;
      inset: auto -80px -130px auto;
      width: 330px;
      height: 330px;
      border-radius: 999px;
      background: rgba(255, 255, 255, 0.15);
    }

    .eyebrow {
      width: fit-content;
      border: 1px solid rgba(255, 255, 255, 0.42);
      border-radius: 999px;
      padding: 8px 12px;
      font-size: 12px;
      font-weight: 900;
      letter-spacing: 0.08em;
      text-transform: uppercase;
      background: rgba(255, 255, 255, 0.12);
    }

    h1 {
      max-width: 720px;
      margin: 22px 0 14px;
      font-size: clamp(34px, 5vw, 62px);
      line-height: 0.98;
      letter-spacing: 0;
    }

    .lead {
      max-width: 660px;
      margin: 0;
      color: rgba(255, 255, 255, 0.86);
      line-height: 1.65;
      font-size: 16px;
    }

    .hero-stats {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 12px;
      margin-top: 30px;
      position: relative;
      z-index: 1;
    }

    .stat {
      border: 1px solid rgba(255, 255, 255, 0.26);
      border-radius: 8px;
      padding: 14px;
      background: rgba(255, 255, 255, 0.12);
      backdrop-filter: blur(10px);
    }

    .stat span {
      display: block;
      color: rgba(255, 255, 255, 0.72);
      font-size: 12px;
      margin-bottom: 6px;
    }

    .stat strong {
      font-size: 19px;
    }

    .workspace {
      display: grid;
      gap: 16px;
    }

    .card {
      border: 1px solid var(--line);
      border-radius: 8px;
      padding: 18px;
      background: rgba(255, 255, 255, 0.92);
      box-shadow: 0 12px 30px rgba(151, 72, 19, 0.09);
    }

    .card-header {
      display: flex;
      align-items: flex-start;
      justify-content: space-between;
      gap: 14px;
      margin-bottom: 14px;
    }

    h2 {
      margin: 0;
      font-size: 18px;
    }

    .helper {
      margin: 4px 0 0;
      color: var(--muted);
      line-height: 1.45;
      font-size: 13px;
    }

    .step {
      width: 30px;
      height: 30px;
      border-radius: 999px;
      display: grid;
      place-items: center;
      flex: 0 0 auto;
      color: #ffffff;
      background: var(--orange);
      font-weight: 900;
      font-size: 13px;
    }

    .upload-zone {
      min-height: 260px;
      border: 2px dashed #f0b180;
      border-radius: 8px;
      display: grid;
      place-items: center;
      padding: 18px;
      text-align: center;
      background:
        linear-gradient(180deg, rgba(255, 243, 232, 0.82), rgba(255, 255, 255, 0.94));
      overflow: hidden;
    }

    .upload-empty strong {
      display: block;
      font-size: 18px;
    }

    .upload-empty p {
      max-width: 320px;
      margin: 8px auto 0;
      color: var(--muted);
      line-height: 1.45;
    }

    .scan-icon {
      width: 72px;
      height: 72px;
      margin: 0 auto 12px;
      border-radius: 8px;
      display: grid;
      place-items: center;
      color: var(--orange);
      background: #ffffff;
      border: 1px solid var(--line);
      box-shadow: 0 8px 20px rgba(242, 106, 33, 0.12);
      font-size: 34px;
      font-weight: 900;
    }

    .upload-zone img {
      display: none;
      width: 100%;
      max-height: 360px;
      object-fit: contain;
      border-radius: 8px;
      background: #111111;
    }

    .controls {
      display: grid;
      grid-template-columns: 1fr 1fr;
      gap: 10px;
      margin-top: 12px;
    }

    button,
    label.file-label {
      min-height: 44px;
      border: 0;
      border-radius: 8px;
      padding: 11px 14px;
      display: inline-flex;
      align-items: center;
      justify-content: center;
      gap: 8px;
      color: #ffffff;
      background: linear-gradient(135deg, var(--orange), var(--orange-dark));
      font-weight: 900;
      cursor: pointer;
      box-shadow: 0 10px 22px rgba(242, 106, 33, 0.2);
    }

    button.secondary {
      color: var(--orange-dark);
      background: var(--orange-soft);
      border: 1px solid #f3c19a;
      box-shadow: none;
    }

    button:hover,
    label.file-label:hover {
      transform: translateY(-1px);
      filter: brightness(0.99);
    }

    input[type="file"] { display: none; }

    .result {
      display: grid;
      gap: 13px;
    }

    .status {
      padding: 12px;
      border-radius: 8px;
      border: 1px solid var(--line);
      color: var(--muted);
      background: #fffaf6;
      line-height: 1.45;
    }

    .label {
      margin: 2px 0 0;
      font-size: clamp(32px, 5vw, 52px);
      line-height: 1;
      font-weight: 950;
      letter-spacing: 0;
    }

    .normal { color: var(--success); }
    .pneumonia { color: var(--danger); }

    .metric {
      display: flex;
      justify-content: space-between;
      gap: 16px;
      color: var(--muted);
      font-size: 14px;
      margin-bottom: 7px;
    }

    .metric strong {
      color: var(--ink);
    }

    .bar {
      height: 12px;
      background: #f7e2d0;
      border-radius: 999px;
      overflow: hidden;
    }

    .bar > span {
      display: block;
      height: 100%;
      width: 0;
      background: linear-gradient(90deg, var(--orange), var(--orange-dark));
      transition: width 240ms ease;
    }

    .normal-bar > span {
      background: linear-gradient(90deg, #20a35b, var(--success));
    }

    .pneumonia-bar > span {
      background: linear-gradient(90deg, #f26a21, var(--danger));
    }

    .mini-grid {
      display: grid;
      grid-template-columns: repeat(3, 1fr);
      gap: 10px;
    }

    .mini {
      min-width: 0;
      padding: 12px;
      border: 1px solid var(--line);
      border-radius: 8px;
      color: var(--muted);
      background: #ffffff;
      font-size: 12px;
    }

    .mini strong {
      display: block;
      margin-top: 5px;
      color: var(--ink);
      font-size: 15px;
      overflow-wrap: anywhere;
    }

    .note {
      padding: 12px;
      border-radius: 8px;
      color: #7a3e00;
      background: #fff4dd;
      border: 1px solid #f0c783;
      line-height: 1.48;
      font-size: 13px;
    }

    .footer {
      padding: 18px 0 0;
      color: var(--muted);
      font-size: 13px;
      line-height: 1.55;
    }

    @media (max-width: 940px) {
      .hero {
        grid-template-columns: 1fr;
      }

      .intro {
        min-height: auto;
      }
    }

    @media (max-width: 620px) {
      .topbar {
        align-items: flex-start;
        flex-direction: column;
      }

      .hero-stats,
      .mini-grid,
      .controls {
        grid-template-columns: 1fr;
      }
    }
  </style>
</head>
<body>
  <div class="page">
    <header class="topbar">
      <div class="brand">
        <div class="brand-mark">R</div>
        <div>
          <p class="brand-title"><strong>REVA University</strong> | AIML Project</p>
          <p class="brand-subtitle">Pneumonia detection using transfer learning</p>
        </div>
      </div>
      <div class="status-pill">MobileNetV2 Clinical Imaging Demo</div>
    </header>

    <main class="hero">
      <section class="intro">
        <div>
          <div class="eyebrow">Chest X-Ray Intelligence</div>
          <h1>Pneumonia Detection System</h1>
          <p class="lead">
            Upload a chest scan and let the trained model estimate whether the image
            belongs to Normal or Pneumonia class. The interface is built for academic
            demonstration, clear explanation, and confident project presentation.
          </p>
        </div>

        <div class="hero-stats">
          <div class="stat"><span>Model</span><strong>MobileNetV2</strong></div>
          <div class="stat"><span>Input</span><strong>224 x 224</strong></div>
          <div class="stat"><span>Classes</span><strong>2</strong></div>
        </div>
      </section>

      <section class="workspace">
        <div class="card">
          <div class="card-header">
            <div>
              <h2>Upload Scan</h2>
              <p class="helper">Use a PNG, JPG, or JPEG chest X-ray image.</p>
            </div>
            <div class="step">1</div>
          </div>

          <div class="upload-zone" id="dropZone">
            <div class="upload-empty" id="placeholder">
              <div class="scan-icon">+</div>
              <strong>Select a chest image</strong>
              <p>Choose your own scan or load a sample image to test the trained model.</p>
            </div>
            <img id="preview" alt="Selected scan preview">
          </div>

          <div class="controls">
            <label class="file-label" for="imageInput">Choose Image</label>
            <input id="imageInput" type="file" accept="image/png,image/jpeg">
            <button id="predictBtn">Predict</button>
            <button class="secondary" data-sample="/sample/normal">Normal Sample</button>
            <button class="secondary" data-sample="/sample/pneumonia">Pneumonia Sample</button>
          </div>
        </div>

        <div class="card result">
          <div class="card-header">
            <div>
              <h2>Prediction Result</h2>
              <p class="helper">Confidence scores update after analysis.</p>
            </div>
            <div class="step">2</div>
          </div>

          <div class="status" id="status">Waiting for an image.</div>
          <p class="label" id="label">--</p>

          <div>
            <div class="metric"><span>Confidence</span><strong id="confidence">0%</strong></div>
            <div class="bar"><span id="confidenceBar"></span></div>
          </div>

          <div>
            <div class="metric"><span>Normal probability</span><strong id="normalProb">0%</strong></div>
            <div class="bar normal-bar"><span id="normalBar"></span></div>
          </div>

          <div>
            <div class="metric"><span>Pneumonia probability</span><strong id="pneumoniaProb">0%</strong></div>
            <div class="bar pneumonia-bar"><span id="pneumoniaBar"></span></div>
          </div>

          <div class="mini-grid">
            <div class="mini">Mode<strong id="mode">--</strong></div>
            <div class="mini">Dataset<strong>Chest X-Ray</strong></div>
            <div class="mini">Architecture<strong>MobileNetV2</strong></div>
          </div>

          <div class="note" id="note">
            This tool is for academic demonstration only. It is not a medical diagnosis system.
          </div>
        </div>
      </section>
    </main>

    <footer class="footer">
      Not for clinical use. Always consult qualified healthcare professionals for medical decisions.
    </footer>
  </div>

  <script>
    const input = document.getElementById("imageInput");
    const preview = document.getElementById("preview");
    const placeholder = document.getElementById("placeholder");
    const predictBtn = document.getElementById("predictBtn");
    const statusBox = document.getElementById("status");
    let currentFile = null;

    function showPreview(file) {
      currentFile = file;
      preview.src = URL.createObjectURL(file);
      preview.style.display = "block";
      placeholder.style.display = "none";
      statusBox.textContent = "Image loaded. Click Predict to analyze the scan.";
    }

    input.addEventListener("change", () => {
      if (input.files && input.files[0]) showPreview(input.files[0]);
    });

    document.querySelectorAll("[data-sample]").forEach(button => {
      button.addEventListener("click", async () => {
        statusBox.textContent = "Loading sample image...";
        const response = await fetch(button.dataset.sample);
        const blob = await response.blob();
        showPreview(new File([blob], "sample.png", { type: blob.type || "image/png" }));
      });
    });

    function percent(value) {
      return `${Math.round(value * 10000) / 100}%`;
    }

    function setBar(id, value) {
      document.getElementById(id).style.width = `${Math.round(value * 100)}%`;
    }

    function renderResult(result) {
      const label = document.getElementById("label");
      label.textContent = result.label;
      label.className = `label ${result.label.toLowerCase()}`;
      document.getElementById("confidence").textContent = percent(result.confidence);
      document.getElementById("normalProb").textContent = percent(result.probabilities.NORMAL || 0);
      document.getElementById("pneumoniaProb").textContent = percent(result.probabilities.PNEUMONIA || 0);
      document.getElementById("mode").textContent = result.mode === "demo" ? "Demo" : "Trained";
      document.getElementById("note").textContent = result.note;
      setBar("confidenceBar", result.confidence);
      setBar("normalBar", result.probabilities.NORMAL || 0);
      setBar("pneumoniaBar", result.probabilities.PNEUMONIA || 0);
      statusBox.textContent = "Prediction complete.";
    }

    predictBtn.addEventListener("click", async () => {
      if (!currentFile) {
        statusBox.textContent = "Please choose an image first.";
        return;
      }
      statusBox.textContent = "Analyzing image with trained model...";
      const dataUrl = await new Promise((resolve, reject) => {
        const reader = new FileReader();
        reader.onload = () => resolve(reader.result);
        reader.onerror = reject;
        reader.readAsDataURL(currentFile);
      });
      const response = await fetch("/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ image_base64: dataUrl.split(",")[1] })
      });
      const result = await response.json();
      if (!response.ok) {
        statusBox.textContent = result.error || "Prediction failed.";
        return;
      }
      renderResult(result);
    });
  </script>
</body>
</html>
"""


class AppHandler(BaseHTTPRequestHandler):
    def _send(self, status: int, body: bytes, content_type: str) -> None:
        self.send_response(status)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, status: int, payload: dict) -> None:
        self._send(status, json.dumps(payload).encode("utf-8"), "application/json")

    def do_GET(self) -> None:
        path = urlparse(self.path).path
        if path == "/":
            self._send(200, INDEX_HTML.encode("utf-8"), "text/html; charset=utf-8")
            return
        if path == "/sample/normal":
            self._send_file(ASSETS_DIR / "sample_normal.png", "image/png")
            return
        if path == "/sample/pneumonia":
            self._send_file(ASSETS_DIR / "sample_pneumonia.png", "image/png")
            return
        if path == "/health":
            self._send_json(200, {"status": "ok"})
            return
        self._send_json(404, {"error": "Not found"})

    def do_POST(self) -> None:
        if urlparse(self.path).path != "/predict":
            self._send_json(404, {"error": "Not found"})
            return

        content_type = self.headers.get("Content-Type", "")
        if "application/json" not in content_type:
            self._send_json(400, {"error": "Expected application/json payload."})
            return

        content_length = int(self.headers.get("Content-Length", "0"))
        payload = json.loads(self.rfile.read(content_length).decode("utf-8"))
        encoded_image = payload.get("image_base64")
        if not encoded_image:
            self._send_json(400, {"error": "No image_base64 field found."})
            return

        image_bytes = base64.b64decode(encoded_image)
        if not image_bytes:
            self._send_json(400, {"error": "Uploaded image is empty."})
            return

        try:
            result = predict_image_bytes(image_bytes)
            self._send_json(200, result.to_dict())
        except Exception as exc:
            self._send_json(500, {"error": f"Prediction failed: {exc}"})

    def _send_file(self, path: Path, content_type: str) -> None:
        if not path.exists():
            self._send_json(404, {"error": f"Missing file: {path.name}"})
            return
        self._send(200, path.read_bytes(), content_type)

    def log_message(self, format: str, *args) -> None:
        return


def main() -> None:
    server = ThreadingHTTPServer((HOST, PORT), AppHandler)
    print(f"Pneumonia Detection app running at http://{HOST}:{PORT}")
    print("Press Ctrl+C to stop.")
    server.serve_forever()


if __name__ == "__main__":
    main()
