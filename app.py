import pickle
import re
from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

# --- same cleaning as training ---
_url = re.compile(r'https?://\S+|www\.\S+')
def clean_text(text: str) -> str:
    t = _url.sub(' ', str(text).lower())
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
    return " ".join(toks)

# --- load your trained model (pickle must be next to app.py in the container) ---
with open("Team-14.pickle", "rb") as f:
    obj = pickle.load(f)
model = obj["model"]
vectorizer = obj["vectorizer"]

app = FastAPI(title="Sentiment Prediction API", version="1.0")

class Review(BaseModel):
    text: str

# ---------- Simple, friendly UI ----------
@app.get("/", response_class=HTMLResponse)
def home():
    return """
<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8" />
<meta name="viewport" content="width=device-width, initial-scale=1" />
<title>Sentiment Demo</title>
<style>
  :root { font-family: ui-sans-serif, system-ui, -apple-system, Segoe UI, Roboto, Arial, sans-serif; }
  body { margin: 0; background: #0b1020; color: #e5e7eb; }
  .wrap { max-width: 720px; margin: 40px auto; padding: 24px; background: #111827; border-radius: 16px; box-shadow: 0 10px 30px rgba(0,0,0,0.35); }
  h1 { margin: 0 0 12px; font-size: 24px; }
  p { color: #9ca3af; margin-top: 0; }
  textarea { width: 100%; min-height: 120px; padding: 12px; border-radius: 10px; border: 1px solid #374151; background:#0b1220; color:#e5e7eb; resize: vertical; font-size: 14px; }
  .row { display:flex; gap:12px; align-items:center; margin: 12px 0 0; }
  button { background:#2563eb; color:#fff; border:none; padding:10px 16px; border-radius:10px; cursor:pointer; font-weight:600; }
  button:disabled { opacity:.6; cursor:not-allowed; }
  .json { background:#0b1220; border:1px solid #374151; border-radius:10px; padding:12px; white-space:pre-wrap; font-family: ui-monospace, SFMono-Regular, Menlo, Consolas, monospace; font-size: 13px; margin-top:12px; color:#cbd5e1; }
  .badge { display:inline-block; padding:4px 8px; border-radius:9999px; font-size: 12px; font-weight:700; }
  .pos { background:#064e3b; color:#a7f3d0; }
  .neg { background:#4c0519; color:#fecdd3; }
  .meter { height: 10px; background:#1f2937; border-radius:9999px; overflow:hidden; margin-top:6px; }
  .bar { height: 100%; background:#10b981; width:0%; transition: width .3s; }
  .hint { font-size:12px; color:#9ca3af; margin-top: 8px; }
  .small { font-size:12px; color:#9ca3af; }
</style>
</head>
<body>
  <div class="wrap">
    <h1>Movie Review Sentiment</h1>
    <p>Paste any review text below, hit Predict. The app will POST JSON (<span class="small">{ "text": "..." }</span>) to the API and show the result.</p>

    <label for="text" class="small">Your review text</label>
    <textarea id="text" placeholder="e.g., I absolutely loved this movie!"></textarea>

    <div class="row">
      <button id="btn">Predict</button>
      <span id="status" class="hint"></span>
    </div>

    <div id="result" style="display:none; margin-top:16px;">
      <div id="chip" class="badge">prediction</div>
      <div class="meter"><div id="bar" class="bar"></div></div>
      <div id="prob" class="hint"></div>

      <div class="hint">Request JSON we sent:</div>
      <div id="req" class="json"></div>

      <div class="hint">Response JSON:</div>
      <div id="res" class="json"></div>
    </div>
  </div>

<script>
const $ = (id) => document.getElementById(id);
const btn = $('btn'), text = $('text'), status = $('status');
const chip = $('chip'), bar = $('bar'), prob = $('prob'), req = $('req'), res = $('res'), box = $('result');

async function predict() {
  const t = text.value.trim();
  if (!t) { status.textContent = "Please enter some text."; return; }
  btn.disabled = true; status.textContent = "Predicting…";
  const payload = { text: t };
  req.textContent = JSON.stringify(payload, null, 2);
  try {
    const r = await fetch('/predict', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify(payload)
    });
    const data = await r.json();
    res.textContent = JSON.stringify(data, null, 2);
    const p = Number(data.probability || 0);
    prob.textContent = `Probability positive: ${(p*100).toFixed(2)}%`;
    bar.style.width = Math.max(2, Math.min(100, p*100)) + '%';
    if ((data.prediction ?? 0) === 1) { chip.textContent = 'POSITIVE'; chip.className='badge pos'; }
    else { chip.textContent = 'NEGATIVE'; chip.className='badge neg'; }
    box.style.display = '';
    status.textContent = "";
  } catch (e) {
    status.textContent = "Error contacting API. See console.";
    console.error(e);
  } finally {
    btn.disabled = false;
  }
}
btn.addEventListener('click', predict);
text.addEventListener('keydown', (e) => { if (e.ctrlKey && e.key === 'Enter') predict(); });
</script>
</body>
</html>
    """

# ---------- JSON API stays the same ----------
@app.post("/predict")
def predict(item: Review):
    txt = clean_text(item.text)
    X = vectorizer.transform([txt])
    p = float(model.predict_proba(X)[0, 1])
    return JSONResponse({"prediction": int(p >= 0.5), "probability": round(p, 4)})
