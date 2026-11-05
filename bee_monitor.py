# bee_monitor.py
# Мониторинг частот жужжания ульев: распознавание частоты + рекомендации + история 5 дней + сортировка по ульям
#
# Установка: pip install flask numpy scipy soundfile
# Для webm/opus (запись из браузера) нужен ffmpeg в PATH.
# Запуск: python bee_monitor.py  ->  http://127.0.0.1:5000

import io
import os
import sqlite3
import urllib.request
import json as _json
import time as _time
import subprocess
import tempfile
import numpy as np
import soundfile as sf
from datetime import datetime, timedelta
from scipy import signal
from flask import Flask, request, jsonify, Response, g

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(BASE_DIR, "bees.db")
KEEP_DAYS = 5                      # сколько дней хранить историю
BEE_MIN_HZ, BEE_MAX_HZ = 40, 1000  # диапазон поиска пика (вентиляция от 50 Гц)

app = Flask(__name__)


# ---------------------------------------------------------------- БД
def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(DB_PATH)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


def init_db():
    con = sqlite3.connect(DB_PATH)
    con.execute("""
        CREATE TABLE IF NOT EXISTS measurements (
            id         INTEGER PRIMARY KEY AUTOINCREMENT,
            hive       TEXT    NOT NULL,
            ts         TEXT    NOT NULL,
            peak_hz    REAL    NOT NULL,
            confidence REAL    NOT NULL,
            rms_db     REAL    NOT NULL,
            temp_c     REAL,
            state      TEXT,
            advice     TEXT
        )
    """)
    con.execute("CREATE INDEX IF NOT EXISTS idx_hive_ts ON measurements(hive, ts)")
    con.commit()
    con.close()


def cleanup_old(con):
    limit = (datetime.now() - timedelta(days=KEEP_DAYS)).isoformat(timespec="seconds")
    con.execute("DELETE FROM measurements WHERE ts < ?", (limit,))
    con.commit()


# ---------------------------------------------------------------- рекомендации
def recommend(peak_hz: float, rms_db: float, temp_c: float | None):
    """
    Диапазоны частот жужжания медоносных пчёл:
      70-180 Гц   - вентиляция (жарко / высокая влажность), база 50-100 Гц
      200-400 Гц  - стандартная работа: лёт, сбор нектара, активность в гнезде
      400-550 Гц  - возбуждение, подготовка к роению, опасность / стресс
    """
    state, advice = "Стандартная работа", (
        "Обычный полетный шум и повседневная активность гнезда. "
        "Вмешательство не требуется.")

    if peak_hz < 70:
        state = "Низкая активность / покой"
        advice = ("Частота ниже диапазона вентиляции: пчёлы малоактивны — "
                  "возможно, прохладно, ночь или зимний клуб. "
                  "Проверьте утепление и кормовые запасы.")
    elif peak_hz < 180:
        state = "Вентиляция: жарко / влажно"
        advice = ("Пчёлы активно вентилируют улей — в гнезде жарко или высокая "
                  "влажность. Откройте леток пошире, добавьте тень, "
                  "обеспечьте доступ к воде и усильте вентиляцию крыши.")
    elif peak_hz < 200:
        state = "Переходный режим"
        advice = ("Частота между вентиляцией и обычной активностью — "
                  "пограничное состояние. Наблюдайте динамику: если частота "
                  "растёт к 200+ Гц, колония успокаивается и переходит "
                  "к нормальной работе.")
    elif peak_hz <= 400:
        state = "Стандартная работа"
        advice = ("Обычный полетный шум, сбор нектара и повседневная "
                  "активность в гнезде. Всё в норме — плановый осмотр по графику.")
    elif peak_hz <= 550:
        state = "Возбуждение / риск роения"
        advice = ("Резкий рост частоты — подготовка к роению, сигнал опасности "
                  "или сильный стресс. Осмотрите улей: проверьте маточники, "
                  "свободное место под засев, наличие матки; при признаках "
                  "роения готовьте отводок и рамки-ловушки.")
    else:
        state = "Сильное возбуждение / тревога"
        advice = ("Частота выше 550 Гц — пик тревоги. Возможны нападение "
                  "(пчёлы-воры, осы), потеря матки, вибрации или шум рядом. "
                  "Осмотрите леток и прилегающую территорию, при необходимости "
                  "сократите леток и устраните раздражитель.")

    # Поправки по уровню сигнала
    if rms_db > -12:
        advice += " Сигнал очень громкий — источник близко к микрофону, проверьте позицию датчика."
    elif rms_db < -55:
        advice += " Сигнал слабый — микрофон далеко или пчёлы малоактивны, проверьте датчик."

    # Поправки по температуре
    if temp_c is not None:
        if temp_c >= 30 and peak_hz >= 200:
            advice += " При жаре (>=30 °C) высокая частота может быть следствием перегрева — проверьте вентиляцию."
        elif temp_c <= 12 and peak_hz >= 400:
            advice += " В холодную погоду такая высокая частота нетипична — перепроверьте запись или осмотрите улей."

    return state, advice


# ---------------------------------------------------------------- аудио
def read_audio(data: bytes):
    try:
        audio, sr = sf.read(io.BytesIO(data), always_2d=False)
    except Exception:
        # webm/ogg из браузера -> wav через ffmpeg
        with tempfile.NamedTemporaryFile(suffix=".webm", delete=False) as f:
            f.write(data)
            src = f.name
        dst = src + ".wav"
        try:
            subprocess.run(["ffmpeg", "-y", "-i", src, "-ar", "44100",
                            "-ac", "1", dst], check=True, capture_output=True)
            audio, sr = sf.read(dst, always_2d=False)
        except FileNotFoundError:
            raise RuntimeError("Не найден ffmpeg — установите его для чтения записей браузера")
        finally:
            for p in (src, dst):
                if os.path.exists(p):
                    os.remove(p)
    if getattr(audio, "ndim", 1) > 1:
        audio = audio.mean(axis=1)
    return np.asarray(audio, dtype=np.float32), int(sr)


def voice_check(f, pxx, peak_hz):
    """Голос vs жужжание по форме спектра."""
    db = 10 * np.log10(pxx + 1e-18)

    band = (f >= 70) & (f <= 500)
    if band.sum() < 10:
        return False, "", {}
    fb, fdb = f[band], db[band]

    thr = float(np.median(fdb)) + 6.0
    peaks = []
    for i in range(1, len(fdb) - 1):
        if fdb[i] > fdb[i - 1] and fdb[i] > fdb[i + 1] and fdb[i] > thr:
            peaks.append((float(fb[i]), float(fdb[i])))
    peaks.sort(key=lambda x: -x[1])
    peaks = peaks[:6]
    peaks.sort()

    comb_hits, comb_info = 0, []
    for i, (f1, _) in enumerate(peaks):
        for f2, _ in peaks[i + 1:]:
            if f1 <= 0:
                continue
            ratio = f2 / f1
            for h in (2, 3, 4):
                if abs(ratio - h) < 0.12:
                    comb_hits += 1
                    comb_info.append("%.0fx%d" % (f1, h))
                    break

    bee_band = (f >= 300) & (f <= 900)
    if bee_band.sum() > 3:
        bdb = db[bee_band]
        bee_span = float(np.percentile(bdb, 90) - np.percentile(bdb, 10))
        bee_level = float(np.median(bdb))
    else:
        bee_span, bee_level = 0.0, -200.0

    low_peak_hz, low_peak_db = (peaks[0] if peaks else (0.0, -200.0))
    total_db = float(np.max(db[(f >= 40) & (f <= 2000)]))

    e_low = float(pxx[(f >= 60) & (f < 300)].sum())
    e_mid = float(pxx[(f >= 300) & (f < 900)].sum())
    low_share = e_low / (e_low + e_mid + 1e-12)

    hi_band = (f >= 700) & (f <= 1500)
    hi_level = float(np.median(db[hi_band])) if hi_band.any() else -200.0
    falloff = low_peak_db - hi_level

    ratios = {
        "comb_hits": comb_hits,
        "peaks_hz": [round(x[0], 1) for x in peaks],
        "bee_span_db": round(bee_span, 1),
        "bee_level_db": round(bee_level, 1),
        "low_peak_hz": round(low_peak_hz, 1),
        "low_peak_db": round(low_peak_db, 1),
        "low_share": round(low_share, 3),
        "falloff_db": round(falloff, 1),
    }

    reasons = []
    if comb_hits >= 2 and low_peak_hz <= 320:
        reasons.append("гармоническая решётка (%s)" % ", ".join(comb_info[:3]))
    if low_peak_hz <= 260 and (low_peak_db - bee_level) > 6:
        reasons.append("пик %.0f Гц выше плато на %.0f дБ" % (low_peak_hz, low_peak_db - bee_level))
    if low_share > 0.7 and low_peak_hz <= 280:
        reasons.append("доминирует энергия ниже 300 Гц (%.0f%%)" % (low_share * 100))
    if falloff > 25 and low_peak_hz <= 300:
        reasons.append("спектр спадает с частотой на %.0f дБ" % falloff)
    if bee_span < 3 and bee_level > total_db - 25:
        reasons.append("спектр плоский — похоже на ветер или гул")

    return bool(reasons), "; ".join(reasons), ratios


def analyze_signal(audio: np.ndarray, sr: int):
    if len(audio) < sr * 0.3:
        raise RuntimeError("Слишком короткая запись (нужно >= 0.3 с)")
    audio = audio - float(np.mean(audio))
    f, pxx = signal.welch(audio, fs=sr, nperseg=min(8192, len(audio)))
    mag_db = 10 * np.log10(pxx + 1e-12)

    band = (f >= BEE_MIN_HZ) & (f <= BEE_MAX_HZ)
    fb, pb = f[band], pxx[band]
    peak_hz = float(fb[int(np.argmax(pb))])

    win = (f >= peak_hz - 25) & (f <= peak_hz + 25)
    confidence = float(pxx[win].sum() / (pxx[f <= 2000].sum() + 1e-12))
    rms_db = float(20 * np.log10(np.sqrt(np.mean(audio ** 2)) + 1e-12))

    is_voice, voice_reason, ratios = voice_check(f, pxx, peak_hz)

    step = max(1, len(f) // 1200)
    return {
        "peak_hz": peak_hz,
        "confidence": min(confidence, 1.0),
        "rms_db": rms_db,
        "chart_f": f[::step].tolist(),
        "chart_db": mag_db[::step].tolist(),
        "is_voice": is_voice,
        "voice_reason": voice_reason,
        "ratios": ratios,
    }


# ---------------------------------------------------------------- API
init_db()


# ---------------- Погода (wttr.in, без ключа) ----------------
WEATHER_LAT = 55.75   # ← координаты вашей пасеки
WEATHER_LON = 37.62
_weather_cache = {"ts": 0.0, "data": None}


def fetch_weather(lat=WEATHER_LAT, lon=WEATHER_LON):
    """Текущая погода с кэшем 30 минут."""
    now = _time.time()
    if _weather_cache["data"] and now - _weather_cache["ts"] < 1800:
        return _weather_cache["data"]

    url = "https://wttr.in/%.4f,%.4f?format=j1" % (lat, lon)
    try:
        req = urllib.request.Request(url, headers={"User-Agent": "curl/8"})
        with urllib.request.urlopen(req, timeout=8) as r:
            raw = _json.loads(r.read().decode("utf-8"))
        cur = (raw.get("current_condition") or [{}])[0]
        data = {
            "temp_c": float(cur["temp_C"]) if cur.get("temp_C") else None,
            "wind_ms": round(float(cur.get("windspeedKmph", 0)) / 3.6, 1),
            "precip_mm": float(cur.get("precipMM", 0) or 0),
            "cloud_pct": int(cur.get("cloudcover", 0) or 0),
            "humidity": int(cur.get("humidity", 0) or 0),
            "source": "wttr.in",
        }
    except Exception as e:
        data = {"temp_c": None, "error": str(e), "source": "wttr.in"}

    _weather_cache["ts"] = now
    _weather_cache["data"] = data
    return data


@app.route("/api/weather")
def api_weather():
    lat = float(request.args.get("lat", WEATHER_LAT))
    lon = float(request.args.get("lon", WEATHER_LON))
    return jsonify(fetch_weather(lat, lon))


@app.route("/")
def index():
    return Response(PAGE, mimetype="text/html")


@app.route("/health")
def health():
    import shutil
    ffmpeg = shutil.which("ffmpeg")
    if not ffmpeg:
        try:
            import imageio_ffmpeg
            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
        except Exception:
            ffmpeg = None
    return jsonify({
        "ok": True,
        "db": os.path.abspath(DB_PATH),
        "ffmpeg": bool(ffmpeg),
    })


@app.route("/api/analyze", methods=["POST"])
def api_analyze():
    hive = (request.form.get("hive") or "").strip()
    if not hive:
        return jsonify(error="Не указан улей")
    save = request.form.get("save", "1") == "1"
    temp_raw = request.form.get("temp_c")
    temp_c = float(temp_raw) if temp_raw not in (None, "") else None
    if temp_c is None:
        temp_c = fetch_weather().get("temp_c")

    if "audio" not in request.files:
        return jsonify(error="Аудио не передано")
    try:
        audio, sr = read_audio(request.files["audio"].read())
        res = analyze_signal(audio, sr)
    except Exception as e:
        return jsonify(error=str(e))

    peak_hz = res["peak_hz"]
    conf    = res["confidence"]
    rms_db  = res["rms_db"]

    if res["is_voice"] and save:
        return jsonify(
            error=("Похоже на голос или посторонний шум, а не на пчёл ("
                   + res["voice_reason"] + "). Повторите замер молча, "
                   "микрофон ближе к летку."),
            is_voice=True,
            ratios=res["ratios"],
        )

    state, advice = recommend(peak_hz, rms_db, temp_c)
    rec_id = None

    if save:
        con = get_db()
        cleanup_old(con)
        cur = con.execute(
            "INSERT INTO measurements (hive, ts, peak_hz, confidence, rms_db, temp_c, state, advice)"
            " VALUES (?,?,?,?,?,?,?,?)",
            (hive, datetime.now().isoformat(timespec="seconds"),
             peak_hz, conf, rms_db, temp_c, state, advice),
        )
        con.commit()
        rec_id = cur.lastrowid

    return jsonify(id=rec_id, hive=hive, peak_hz=peak_hz, confidence=conf,
                   rms_db=rms_db, temp_c=temp_c, state=state, advice=advice,
                   saved=bool(save), freqs=res["chart_f"], mag_db=res["chart_db"],
                   is_voice=res["is_voice"], voice_reason=res["voice_reason"],
                   ratios=res["ratios"])


@app.route("/api/history")
def api_history():
    con = get_db()
    cleanup_old(con)
    hive = request.args.get("hive", "").strip()
    days = min(max(int(request.args.get("days", KEEP_DAYS)), 1), KEEP_DAYS)
    since = (datetime.now() - timedelta(days=days)).isoformat(timespec="seconds")

    sql = "SELECT * FROM measurements WHERE ts >= ?"
    params = [since]
    if hive and hive != "all":
        sql += " AND hive = ?"
        params.append(hive)
    sql += " ORDER BY ts DESC LIMIT 500"
    rows = [dict(r) for r in con.execute(sql, params).fetchall()]

    hives = [r["hive"] for r in con.execute(
        "SELECT DISTINCT hive FROM measurements ORDER BY hive")]
    return jsonify(rows=rows, hives=hives)


@app.route("/api/delete/<int:rec_id>", methods=["POST"])
def api_delete(rec_id):
    con = get_db()
    con.execute("DELETE FROM measurements WHERE id = ?", (rec_id,))
    con.commit()
    return jsonify(ok=True)


@app.route("/api/stats")
def api_stats():
    con = get_db()
    cleanup_old(con)
    hive = request.args.get("hive", "all")
    sql = "SELECT peak_hz FROM measurements WHERE ts >= ?"
    params = [(datetime.now() - timedelta(days=KEEP_DAYS)).isoformat(timespec="seconds")]
    if hive != "all":
        sql += " AND hive = ?"
        params.append(hive)
    rows = con.execute(sql, params).fetchall()
    if not rows:
        return jsonify(count=0)
    peaks = [r["peak_hz"] for r in rows]
    return jsonify(count=len(rows),
                   avg=float(np.mean(peaks)),
                   min=float(np.min(peaks)),
                   max=float(np.max(peaks)))


# ---------------------------------------------------------------- HTML
PAGE = r"""<!DOCTYPE html>
<html lang="ru">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Мониторинг ульев</title>
<style>
  * { box-sizing: border-box; }
  body { font-family: system-ui, -apple-system, sans-serif; margin: 0;
         background: #faf7f0; color: #2b2b2b; }
  header { background: #f2b705; padding: 16px 24px; }
  header h1 { font-size: 20px; margin: 0; }
  main { padding: 20px 24px; max-width: 1180px; margin: 0 auto; }
  .card { background: #fff; border-radius: 12px; padding: 18px;
          box-shadow: 0 1px 4px rgba(0,0,0,.08); margin-bottom: 16px; }
  button { padding: 10px 18px; border: none; border-radius: 8px;
           background: #f2b705; font-weight: 600; cursor: pointer; font-size: 14px;
           font-family: inherit; }
  button:hover { filter: brightness(.95); }
  button.rec { background: #e0574d; color: #fff; }
  button.ghost { background: #eee; }
  button.sm { padding: 4px 10px; font-size: 12px; }
  input, select { padding: 9px 12px; border: 1px solid #ddd; border-radius: 8px;
                  font-size: 14px; font-family: inherit; }
  .row { display: flex; gap: 12px; flex-wrap: wrap; align-items: center; }
  .grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px,1fr)); gap: 14px; }
  .metric .label { font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #999; }
  .metric .value { font-size: 28px; font-weight: 700; color: #c98b00; }
  #status { color: #777; font-size: 14px; }
  canvas { width: 100%; height: 200px; display: block; }
  table { width: 100%; border-collapse: collapse; font-size: 14px; }
  th, td { padding: 9px 10px; text-align: left; border-bottom: 1px solid #f0eee8; }
  th { font-size: 11px; text-transform: uppercase; letter-spacing: .5px; color: #999;
       cursor: pointer; user-select: none; white-space: nowrap; }
  th:hover { color: #c98b00; }
  tr:hover td { background: #fffdf5; }
  .badge { display: inline-block; padding: 3px 9px; border-radius: 20px;
           font-size: 12px; font-weight: 600; white-space: nowrap; }
  .st-norm { background: #e3f5e3; color: #2e7d32; }
  .st-warm { background: #ffe9cc; color: #b26a00; }
  .st-hot  { background: #fde0dc; color: #c62828; }
  .st-cold { background: #dceaf9; color: #1565c0; }
  .st-vent { background: #e8e0f5; color: #6a3fa0; }
  .advice { background: #fffbe8; border-left: 4px solid #f2b705;
            padding: 12px 14px; border-radius: 6px; line-height: 1.5; }
  .advice b { color: #b26a00; }
  .legend { display: flex; gap: 14px; flex-wrap: wrap; font-size: 12px; color: #777; }
  .legend span i { display: inline-block; width: 10px; height: 10px;
                   border-radius: 3px; margin-right: 5px; }
  .empty { text-align: center; color: #999; padding: 28px; }
  .spark { height: 90px; width: 100%; }
  @media (max-width: 700px) { .hide-sm { display: none; } }
</style>
</head>
<body>
<header>
  <h1>&#128029; Мониторинг частот жужжания ульев</h1>
</header>

<main>
  <!-- ЗАПИСЬ -->
  <div class="card">
    <div class="row" style="margin-bottom:14px">
      <input id="hive" list="hiveList" placeholder="Улей (например: Улей 1)" style="flex:1;min-width:180px">
      <datalist id="hiveList"></datalist>
      <input id="temp" type="number" step="0.1" placeholder="Температура °C (необяз.)" style="width:200px">
    </div>
    <div class="row">
      <input type="file" id="file" accept="audio/*">
      <button id="recBtn" class="rec">&#9679; Записать с микрофона</button>
      <span id="status"></span>
    </div>
  </div>

  <!-- РЕЗУЛЬТАТ -->
  <div class="card" id="resultCard" style="display:none">
    <div class="grid" style="margin-bottom:14px">
      <div class="metric"><div class="label">Основная частота</div>
        <div class="value" id="rPeak">–</div></div>
      <div class="metric"><div class="label">Уверенность</div>
        <div class="value" id="rConf">–</div></div>
      <div class="metric"><div class="label">Уровень сигнала</div>
        <div class="value" id="rRms">–</div></div>
      <div class="metric"><div class="label">Состояние</div>
        <div class="value" id="rState" style="font-size:18px">–</div></div>
    </div>
    <div class="advice"><b>Рекомендация:</b> <span id="rAdvice"></span></div>
    <div style="margin-top:14px">
      <div class="label" style="font-size:11px;color:#999;text-transform:uppercase;margin-bottom:6px">Спектр</div>
      <canvas id="spec" width="1000" height="200"></canvas>
      <div class="legend" style="margin-top:8px">
        <span><i style="background:#e8e0f5"></i>70–180 Гц вентиляция</span>
        <span><i style="background:#e3f5e3"></i>200–400 Гц норма</span>
        <span><i style="background:#ffe9cc"></i>400–550 Гц роение/стресс</span>
        <span><i style="background:#fde0dc"></i>&gt;550 Гц тревога</span>
      </div>
    </div>
  </div>

  <!-- ИСТОРИЯ -->
  <div class="card">
    <div class="row" style="margin-bottom:14px">
      <h2 style="font-size:16px;margin:0;flex:1">История (последние 5 дней)</h2>
      <select id="hiveFilter"><option value="all">Все ульи</option></select>
      <select id="daysFilter">
        <option value="1">За сутки</option>
        <option value="2">За 2 дня</option>
        <option value="5" selected>За 5 дней</option>
      </select>
      <button class="ghost" id="refreshBtn">Обновить</button>
    </div>

    <div class="grid" id="stats" style="margin-bottom:14px"></div>

    <div style="overflow-x:auto">
      <table>
        <thead>
          <tr>
            <th data-sort="ts">Время</th>
            <th data-sort="hive">Улей</th>
            <th data-sort="peak_hz">Частота</th>
            <th data-sort="confidence" class="hide-sm">Увер.</th>
            <th data-sort="rms_db" class="hide-sm">Сигнал</th>
            <th data-sort="temp_c" class="hide-sm">Темп.</th>
            <th data-sort="state">Состояние</th>
            <th></th>
          </tr>
        </thead>
        <tbody id="tbody"></tbody>
      </table>
    </div>
    <div id="empty" class="empty" style="display:none">Пока нет записей. Сделайте первое измерение.</div>

    <div style="margin-top:16px">
      <div class="label" style="font-size:11px;color:#999;text-transform:uppercase;margin-bottom:6px">
        Динамика частоты по времени</div>
      <canvas id="trend" class="spark" width="1000" height="90"></canvas>
    </div>
  </div>
</main>

<script>
function safeJson(r){
  var ct = (r.headers.get('content-type')||'');
  if(!ct.includes('application/json')){
    return r.text().then(function(){ throw new Error('Сервер просыпается — повторите через 30 секунд'); });
  }
  return r.json();
}

const $ = id => document.getElementById(id);
let currentRows = [], sortKey = 'ts', sortDir = -1, rec = null, chunks = [];

$('file').onchange = e => { const f = e.target.files[0]; if (f) sendAudio(f, f.name); };

$('recBtn').onclick = async () => {
  if (rec && rec.state === 'recording') { rec.stop(); return; }
  if (!$('hive').value.trim()) { $('status').textContent = 'Сначала укажите улей'; return; }
  try {
    const stream = await navigator.mediaDevices.getUserMedia({ audio: true });
    rec = new MediaRecorder(stream); chunks = [];
    rec.ondataavailable = e => chunks.push(e.data);
    rec.onstop = () => {
      stream.getTracks().forEach(t => t.stop());
      $('recBtn').textContent = '\u25CF Записать с микрофона';
      $('recBtn').classList.add('rec');
      sendAudio(new Blob(chunks, { type: 'audio/webm' }), 'mic.webm');
    };
    rec.start();
    $('recBtn').textContent = '\u25A0 Стоп (запись...)';
    $('recBtn').classList.remove('rec');
    $('status').textContent = 'Идёт запись 3–10 секунд...';
  } catch { $('status').textContent = 'Нет доступа к микрофону'; }
};

function sendAudio(blob, name) {
  const hive = $('hive').value.trim();
  if (!hive) { $('status').textContent = 'Укажите улей'; return; }
  $('status').textContent = 'Анализ...';
  const fd = new FormData();
  fd.append('audio', blob, name);
  fd.append('hive', hive);
  fd.append('save', '1');
  // температура подставляется из погоды, если поле пустое
  if (!$('temp').dataset.loaded) {
    $('temp').dataset.loaded = '1';
    fetch('/api/weather').then(safeJson).then(w => {
      if (w.temp_c != null && !$('temp').value) {
        $('temp').value = Math.round(w.temp_c * 10) / 10;
        $('temp').placeholder = 'API: ' + Math.round(w.temp_c) + ' °C';
      }
    }).catch(() => {});
  }

  if ($('temp').value) fd.append('temp_c', $('temp').value);
  fetch('/api/analyze', { method: 'POST', body: fd })
    .then(safeJson)
    .then(d => {
      if (d.error) {
        var r = d.ratios || {};
        var extra = d.ratios
          ? '<div style="margin-top:6px;font-size:12px;color:#8a6d3b">'
            + 'гармоник ' + (r.comb_hits ?? 0)
            + ' · пик ' + (r.low_peak_hz ?? '?') + ' Гц'
            + ' · плато ' + (r.bee_span_db ?? '?') + ' дБ'
            + ' · низ ' + ((r.low_share ?? 0) * 100).toFixed(0) + '%'
            + ' · спад ' + (r.falloff_db ?? '?') + ' дБ'
            + (r.peaks_hz && r.peaks_hz.length ? '<br>пики: ' + r.peaks_hz.join(', ') + ' Гц' : '')
            + '</div>'
          : '';
        $('status').innerHTML =
          '<div style="background:#fff8e1;border-left:4px solid #f2b705;'
          + 'padding:10px 14px;border-radius:6px;line-height:1.5;margin-top:8px">'
          + '<b style="color:#b26a00">\u26A0\uFE0F Замер отклонён</b> — похоже на голос '
          + 'или посторонний шум, а не на пчёл.<br>'
          + String(d.error).split('(')[1]?.split(')')[0] || ''
          + '<br><span style="color:#777">Повторите замер молча, микрофон ближе к летку.</span>'
          + extra + '</div>';
        return;
      }
      var r = d.ratios || {};
      $('status').innerHTML = 'Готово, сохранено'
        + ' <small style="color:#999">(гармоник ' + (r.comb_hits ?? 0)
        + ' · пик ' + (r.low_peak_hz ?? '?') + ' Гц'
        + ' · плато ' + (r.bee_span_db ?? '?') + ' дБ'
        + ' · низ ' + ((r.low_share ?? 0) * 100).toFixed(0) + '%'
        + ' · спад ' + (r.falloff_db ?? '?') + ' дБ)</small>';
      showResult(d);
      loadHistory();
    })
    .catch(e => $('status').textContent = 'Ошибка: ' + e);
}

function stateClass(s) {
  if (!s) return 'st-norm';
  if (s.includes('тревог') || s.includes('Сильное')) return 'st-hot';
  if (s.includes('роения') || s.includes('Возбужд')) return 'st-warm';
  if (s.includes('Вентиляц')) return 'st-vent';
  if (s.includes('Низкая') || s.includes('Переходн')) return 'st-cold';
  return 'st-norm';
}

function stateColor(s) {
  return { 'st-hot': '#c62828', 'st-warm': '#b26a00',
           'st-vent': '#6a3fa0', 'st-cold': '#1565c0' }[stateClass(s)] || '#2e7d32';
}

function showResult(d) {
  $('resultCard').style.display = 'block';
  $('rPeak').textContent = d.peak_hz.toFixed(1) + ' Гц';
  $('rConf').textContent = (d.confidence * 100).toFixed(0) + ' %';
  $('rRms').textContent = d.rms_db.toFixed(1) + ' дБ';
  const st = $('rState');
  st.textContent = d.state;
  st.style.color = stateColor(d.state);
  $('rAdvice').textContent = d.advice;
  drawSpectrum(d.freqs, d.mag_db, d.peak_hz);
}

function drawSpectrum(freqs, mag, peak) {
  const c = $('spec'), ctx = c.getContext('2d');
  const W = c.width, H = c.height, maxF = 1000;
  ctx.clearRect(0, 0, W, H);
  let idx = freqs.findIndex(f => f >= maxF);
  if (idx < 0) idx = freqs.length;
  const seg = mag.slice(0, idx);
  if (!seg.length) return;
  const mMax = Math.max(...seg), mMin = mMax - 70;

  // цветные зоны диапазонов
  const zones = [[0,70,'#dceaf9'],[70,180,'#e8e0f5'],[180,200,'#f5f1e0'],
                 [200,400,'#e3f5e3'],[400,550,'#ffe9cc'],[550,1000,'#fde0dc']];
  zones.forEach(([a, b, col]) => {
    ctx.fillStyle = col;
    const x0 = a / maxF * W, x1 = Math.min(b / maxF * W, W);
    ctx.fillRect(x0, 0, x1 - x0, H);
  });

  // сетка и подписи
  ctx.strokeStyle = 'rgba(0,0,0,.06)'; ctx.lineWidth = 1;
  ctx.fillStyle = '#999'; ctx.font = '11px system-ui';
  for (let f = 0; f <= maxF; f += 100) {
    const x = f / maxF * W;
    ctx.beginPath(); ctx.moveTo(x, 0); ctx.lineTo(x, H); ctx.stroke();
    if (f % 200 === 0) ctx.fillText(f + '', x + 3, H - 3);
  }

  // спектр
  ctx.beginPath(); ctx.strokeStyle = '#8a5b00'; ctx.lineWidth = 1.6;
  for (let i = 0; i < idx; i++) {
    const x = i / idx * W;
    const y = H - (mag[i] - mMin) / (mMax - mMin) * (H - 6) - 3;
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  }
  ctx.stroke();

  // линия пика
  const px = peak / maxF * W;
  ctx.strokeStyle = '#e0574d'; ctx.setLineDash([6, 4]); ctx.lineWidth = 1.5;
  ctx.beginPath(); ctx.moveTo(px, 0); ctx.lineTo(px, H); ctx.stroke();
  ctx.setLineDash([]);
  ctx.fillStyle = '#e0574d'; ctx.font = 'bold 13px system-ui';
  ctx.fillText(peak.toFixed(0) + ' Гц', Math.min(px + 6, W - 70), 16);
}

// ---------- история ----------
function loadHistory() {
  const hive = $('hiveFilter').value, days = $('daysFilter').value;
  fetch(`/api/history?hive=${encodeURIComponent(hive)}&days=${days}`)
    .then(safeJson)
    .then(d => {
      currentRows = d.rows;
      const sel = $('hiveFilter');
      sel.innerHTML = '<option value="all">Все ульи</option>' +
        d.hives.map(h => `<option ${h === hive ? 'selected' : ''}>${h}</option>`).join('');
      $('hiveList').innerHTML = d.hives.map(h => `<option value="${h}">`).join('');
      renderTable();
      drawTrend();
    });
  fetch(`/api/stats?hive=${encodeURIComponent(hive)}`)
    .then(safeJson)
    .then(s => {
      const el = $('stats');
      if (!s.count) { el.innerHTML = ''; return; }
      el.innerHTML = `
        <div class="metric"><div class="label">Измерений</div><div class="value">${s.count}</div></div>
        <div class="metric"><div class="label">Средняя частота</div><div class="value">${s.avg.toFixed(0)} Гц</div></div>
        <div class="metric"><div class="label">Мин / Макс</div><div class="value" style="font-size:20px">${s.min.toFixed(0)} / ${s.max.toFixed(0)} Гц</div></div>`;
    });
}

function renderTable() {
  const rows = [...currentRows].sort((a, b) => {
    let x = a[sortKey], y = b[sortKey];
    if (x == null) x = -Infinity;
    if (y == null) y = -Infinity;
    return (x > y ? 1 : x < y ? -1 : 0) * sortDir;
  });
  $('empty').style.display = rows.length ? 'none' : 'block';
  $('tbody').innerHTML = rows.map(r => {
    const t = new Date(r.ts).toLocaleString('ru-RU', { day:'2-digit', month:'2-digit',
                hour:'2-digit', minute:'2-digit', second:'2-digit' });
    return `<tr title="${(r.advice || '').replace(/"/g, '&quot;')}">
      <td>${t}</td>
      <td><b>${r.hive}</b></td>
      <td><b style="color:#c98b00">${r.peak_hz.toFixed(0)} Гц</b></td>
      <td class="hide-sm">${(r.confidence*100).toFixed(0)}%</td>
      <td class="hide-sm">${r.rms_db.toFixed(0)} дБ</td>
      <td class="hide-sm">${r.temp_c != null ? r.temp_c + '°' : '–'}</td>
      <td><span class="badge ${stateClass(r.state)}">${r.state}</span></td>
      <td><button class="sm ghost" onclick="delRec(${r.id})">✕</button></td>
    </tr>`;
  }).join('');
}

document.querySelectorAll('th[data-sort]').forEach(th => {
  th.onclick = () => {
    const k = th.dataset.sort;
    sortDir = (sortKey === k) ? -sortDir : -1;
    sortKey = k;
    renderTable();
  };
});

function drawTrend() {
  const c = $('trend'), ctx = c.getContext('2d');
  const W = c.width, H = c.height;
  ctx.clearRect(0, 0, W, H);
  const pts = [...currentRows].reverse().filter(r => r.peak_hz);
  if (pts.length < 2) {
    ctx.fillStyle = '#bbb'; ctx.font = '12px system-ui';
    ctx.fillText('Нужно минимум 2 измерения', 10, H / 2);
    return;
  }
  const vals = pts.map(p => p.peak_hz);
  const lo = Math.min(...vals) - 20, hi = Math.max(...vals) + 20;
  ctx.strokeStyle = '#f2b705'; ctx.lineWidth = 2; ctx.beginPath();
  pts.forEach((p, i) => {
    const x = i / (pts.length - 1) * (W - 10) + 5;
    const y = H - 8 - (p.peak_hz - lo) / (hi - lo) * (H - 20);
    i ? ctx.lineTo(x, y) : ctx.moveTo(x, y);
  });
  ctx.stroke();
  ctx.fillStyle = '#bbb'; ctx.font = '10px system-ui';
  ctx.fillText(hi.toFixed(0) + ' Гц', 4, 12);
  ctx.fillText(lo.toFixed(0) + ' Гц', 4, H - 2);
}

function delRec(id) {
  if (!confirm('Удалить запись?')) return;
  fetch('/api/delete/' + id, { method: 'POST' }).then(loadHistory);
}

$('hiveFilter').onchange = loadHistory;
$('daysFilter').onchange = loadHistory;
$('refreshBtn').onclick = loadHistory;
loadHistory();
</script>
</body>
</html>
"""

if __name__ == "__main__":
    init_db()
    print("Сервер запущен: http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=True)
