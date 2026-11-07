import json
import os
import threading
import time
from datetime import datetime
from pathlib import Path

import cv2
import keyboard
import mss
import numpy as np
from flask import Flask, jsonify, render_template, request

app = Flask(__name__)

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "stats.json"
TEMPLATES_DIR = BASE_DIR / "templates"

# Настройки детектора по умолчанию (для 1920x1080)
# Область, где в Fortnite всплывает "+600 XP CHEST" справа сверху от прицела
DEFAULT_CONFIG = {
    "session_chests": 0,
    "total_chests": 0,
    "goal_chests": 100,
    "threshold": 0.76,
    "cooldown_sec": 2.4,
    "roi": {
        "left": 1080,
        "top": 180,
        "width": 520,
        "height": 340
    },
    "hotkeys": {
        "increment": "f9",
        "decrement": "f10",
        "reset_session": "f11"
    }
}

state_lock = threading.Lock()
state = {
    "session_chests": 0,
    "total_chests": 0,
    "goal_chests": 100,
    "threshold": 0.76,
    "cooldown_sec": 2.4,
    "roi": DEFAULT_CONFIG["roi"].copy(),
    "hotkeys": DEFAULT_CONFIG["hotkeys"].copy(),
    "last_detected_ts": 0.0,
    "last_score": 0.0,
    "scanner_active": True,
    "event_id": 0,  # увеличивается при каждом открытии для триггера анимации в OBS
    "history": []
}


def load_state():
    if DATA_FILE.exists():
        try:
            with open(DATA_FILE, "r", encoding="utf-8") as f:
                saved = json.load(f)
                with state_lock:
                    for k in ("session_chests", "total_chests", "goal_chests", "threshold", "cooldown_sec", "roi", "hotkeys"):
                        if k in saved:
                            state[k] = saved[k]
        except Exception as e:
            print(f"[!] Ошибка чтения stats.json: {e}")


def save_state():
    with state_lock:
        data_to_save = {
            "session_chests": state["session_chests"],
            "total_chests": state["total_chests"],
            "goal_chests": state["goal_chests"],
            "threshold": state["threshold"],
            "cooldown_sec": state["cooldown_sec"],
            "roi": state["roi"],
            "hotkeys": state["hotkeys"],
        }
    try:
        with open(DATA_FILE, "w", encoding="utf-8") as f:
            json.dump(data_to_save, f, ensure_ascii=False, indent=2)
    except Exception as e:
        print(f"[!] Ошибка сохранения stats.json: {e}")


def register_chest_open(source="auto", score=1.0):
    now = time.time()
    with state_lock:
        state["session_chests"] += 1
        state["total_chests"] += 1
        state["last_detected_ts"] = now
        state["event_id"] += 1
        entry = {
            "time": datetime.now().strftime("%H:%M:%S"),
            "source": source,
            "score": round(float(score), 3),
            "session": state["session_chests"],
            "total": state["total_chests"]
        }
        state["history"].insert(0, entry)
        state["history"] = state["history"][:15]
        s_cnt = state["session_chests"]
        t_cnt = state["total_chests"]

    save_state()
    print(f"[+] СУНДУК ОТКРЫТ! ({source}, точность: {score:.2f}) | За сессию: {s_cnt} | Всего: {t_cnt}")


def adjust_chests(delta: int):
    with state_lock:
        state["session_chests"] = max(0, state["session_chests"] + delta)
        state["total_chests"] = max(0, state["total_chests"] + delta)
        if delta > 0:
            state["event_id"] += 1
        s_cnt = state["session_chests"]
        t_cnt = state["total_chests"]
    save_state()
    print(f"[*] Ручная корректировка ({delta:+d}) | За сессию: {s_cnt} | Всего: {t_cnt}")


def reset_session():
    with state_lock:
        state["session_chests"] = 0
    save_state()
    print("[*] Счётчик текущей сессии сброшен в 0.")


def preprocess_text_mask(bgr_img: np.ndarray) -> np.ndarray:
    """
    Выделяет яркие бело-розовые/фиолетовые пиксели текста '+600 XP CHEST'
    на фоне характерной фиолетовой обводки Fortnite XP.
    Возвращает одноканальное изображение для устойчивого matchTemplate.
    """
    gray = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2GRAY)
    return gray


def has_purple_xp_glow(bgr_patch: np.ndarray) -> bool:
    """
    Проверяет, что вокруг найденного слова 'CHEST' действительно есть
    фиолетово-розовое свечение опыта Fortnite (чтобы не путать с обычным белым текстом).
    В HSV фиолетовый/пурпурный оттенок Fortnite XP имеет Hue ~ 125..165, Saturation > 60.
    """
    hsv = cv2.cvtColor(bgr_patch, cv2.COLOR_BGR2HSV)
    # Маска фиолетово-розовой обводки
    lower_purple = np.array([120, 50, 70], dtype=np.uint8)
    upper_purple = np.array([168, 255, 255], dtype=np.uint8)
    mask = cv2.inRange(hsv, lower_purple, upper_purple)
    purple_ratio = np.count_nonzero(mask) / mask.size
    return purple_ratio > 0.08


def load_multiscale_templates():
    """
    Загружает шаблон надписи CHEST и создаёт несколько масштабов (0.85x .. 1.15x),
    так как плашка '+600 XP CHEST' слегка пульсирует при появлении.
    """
    tpl_path = TEMPLATES_DIR / "chest_label.png"
    if not tpl_path.exists():
        print(f"[!] Шаблон не найден: {tpl_path}")
        return []

    base_bgr = cv2.imread(str(tpl_path))
    if base_bgr is None:
        return []

    base_gray = preprocess_text_mask(base_bgr)
    templates = []
    for scale in (0.88, 0.94, 1.0, 1.06, 1.12):
        w = max(10, int(base_gray.shape[1] * scale))
        h = max(8, int(base_gray.shape[0] * scale))
        resized = cv2.resize(base_gray, (w, h), interpolation=cv2.INTER_CUBIC)
        templates.append(resized)
    return templates


def screen_scanner_loop():
    """
    Безопасный фоновый сканер экрана через MSS (Desktop Duplication / GDI).
    Не взаимодействует с процессом Fortnite и античитом.
    """
    templates = load_multiscale_templates()
    if not templates:
        print("[!] Не удалось загрузить шаблоны для распознавания!")
        return

    print("[*] Визуальный сканер экрана запущен (1920x1080, безопасный режим OBS/MSS).")
    consecutive_hits = 0

    with mss.mss() as sct:
        while True:
            try:
                with state_lock:
                    active = state["scanner_active"]
                    roi = state["roi"].copy()
                    threshold = float(state["threshold"])
                    cooldown = float(state["cooldown_sec"])
                    last_ts = state["last_detected_ts"]

                if not active:
                    time.sleep(0.25)
                    continue

                now = time.time()
                if now - last_ts < cooldown:
                    time.sleep(0.1)
                    continue

                monitor = {
                    "left": int(roi["left"]),
                    "top": int(roi["top"]),
                    "width": int(roi["width"]),
                    "height": int(roi["height"]),
                }

                sct_img = sct.grab(monitor)
                frame_bgr = np.array(sct_img)[:, :, :3]
                frame_gray = preprocess_text_mask(frame_bgr)

                best_score = 0.0
                best_loc = None
                best_shape = None

                for tpl in templates:
                    if frame_gray.shape[0] < tpl.shape[0] or frame_gray.shape[1] < tpl.shape[1]:
                        continue
                    res = cv2.matchTemplate(frame_gray, tpl, cv2.TM_CCOEFF_NORMED)
                    _, max_val, _, max_loc = cv2.minMaxLoc(res)
                    if max_val > best_score:
                        best_score = float(max_val)
                        best_loc = max_loc
                        best_shape = tpl.shape

                with state_lock:
                    state["last_score"] = round(best_score, 3)

                if best_score >= threshold and best_loc is not None and best_shape is not None:
                    x, y = best_loc
                    th, tw = best_shape
                    patch_bgr = frame_bgr[y:y + th, x:x + tw]

                    if has_purple_xp_glow(patch_bgr):
                        consecutive_hits += 1
                        # Требуем 2 кадра подряд (~80 мс), чтобы исключить случайные вспышки
                        if consecutive_hits >= 2:
                            register_chest_open(source="auto", score=best_score)
                            consecutive_hits = 0
                    else:
                        consecutive_hits = 0
                else:
                    consecutive_hits = 0

            except Exception as e:
                print(f"[!] Ошибка в цикле сканера: {e}")
                time.sleep(0.5)

            # ~20 проверок в секунду — менее 1% нагрузки на CPU
            time.sleep(0.05)


def setup_hotkeys():
    try:
        keyboard.add_hotkey(state["hotkeys"]["increment"], lambda: adjust_chests(+1))
        keyboard.add_hotkey(state["hotkeys"]["decrement"], lambda: adjust_chests(-1))
        keyboard.add_hotkey(state["hotkeys"]["reset_session"], reset_session)
        print(
            f"[*] Горячие клавиши активны: "
            f"[{state['hotkeys']['increment'].upper()}] +1 сундук | "
            f"[{state['hotkeys']['decrement'].upper()}] -1 сундук | "
            f"[{state['hotkeys']['reset_session'].upper()}] Сброс сессии"
        )
    except Exception as e:
        print(f"[!] Не удалось зарегистрировать глобальные хоткеи: {e}")


# ----------------- Маршруты Flask для OBS и Панели управления -----------------

@app.route("/")
@app.route("/widget")
def widget_page():
    """Страница виджета для добавления в OBS (Browser Source)."""
    return render_template("widget.html")


@app.route("/dashboard")
def dashboard_page():
    """Панель управления и тестирования в браузере."""
    return render_template("dashboard.html")


@app.route("/api/state", methods=["GET"])
def api_get_state():
    with state_lock:
        return jsonify(state)


@app.route("/api/action", methods=["POST"])
def api_action():
    data = request.get_json(force=True, silent=True) or {}
    action = data.get("action")

    if action == "increment":
        register_chest_open(source="manual", score=1.0)
    elif action == "decrement":
        adjust_chests(-1)
    elif action == "reset_session":
        reset_session()
    elif action == "reset_all":
        with state_lock:
            state["session_chests"] = 0
            state["total_chests"] = 0
        save_state()
    elif action == "toggle_scanner":
        with state_lock:
            state["scanner_active"] = not state["scanner_active"]
    elif action == "update_settings":
        with state_lock:
            if "goal_chests" in data:
                state["goal_chests"] = max(1, int(data["goal_chests"]))
            if "threshold" in data:
                state["threshold"] = float(data["threshold"])
            if "cooldown_sec" in data:
                state["cooldown_sec"] = float(data["cooldown_sec"])
        save_state()

    with state_lock:
        return jsonify(state)


if __name__ == "__main__":
    load_state()
    setup_hotkeys()

    scanner_thread = threading.Thread(target=screen_scanner_loop, daemon=True)
    scanner_thread.start()

    print("\n" + "=" * 62)
    print("  FORTNITE CHEST TRACKER (БЕЗОПАСНЫЙ ВИДЖЕТ ДЛЯ OBS)")
    print("=" * 62)
    print("  1. Ссылка для OBS (Browser Source): http://localhost:5050/widget")
    print("     Рекомендуемый размер в OBS: Ширина 420, Высота 160")
    print("  2. Панель управления и настроек:    http://localhost:5050/dashboard")
    print("=" * 62 + "\n")

    app.run(host="127.0.0.1", port=5050, debug=False, use_reloader=False)
