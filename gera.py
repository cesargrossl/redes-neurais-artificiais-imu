import os
import json
import numpy as np
from pathlib import Path
import shutil

# =========================
# CONFIGURAÇÃO
# =========================
OUT_DIR = Path("dataset_5min_depth_imu")

DURATION_S = 5 * 60     # 5 minutos
DEPTH_FPS = 10
IMU_HZ = 100
WINDOW_FRAMES = 8
FRAME_W, FRAME_H = 160, 120

np.random.seed(42)

def ensure_dir(p):
    p.mkdir(parents=True, exist_ok=True)

def save_pgm(path, img):
    h, w = img.shape
    header = f"P5\n{w} {h}\n255\n".encode()
    with open(path, "wb") as f:
        f.write(header)
        f.write(img.astype(np.uint8).tobytes())

# Remove dataset antigo
if OUT_DIR.exists():
    shutil.rmtree(OUT_DIR)

ensure_dir(OUT_DIR)

# =========================
# TRAJETÓRIA
# =========================
n_depth = DURATION_S * DEPTH_FPS
t_depth = np.arange(n_depth) / DEPTH_FPS

turn = np.zeros(n_depth)
for i in range(0, n_depth, 80):
    turn[i:i+40] = np.random.choice([-1, 1])

# =========================
# IMU
# =========================
n_imu = DURATION_S * IMU_HZ
t_imu = np.arange(n_imu) / IMU_HZ
yaw_rate = np.interp(t_imu, t_depth, turn * 15)

ax = np.random.normal(0, 0.1, n_imu)
ay = yaw_rate / 90 + np.random.normal(0, 0.1, n_imu)
az = 9.81 + np.random.normal(0, 0.1, n_imu)

gx = np.random.normal(0, 0.2, n_imu)
gy = np.random.normal(0, 0.2, n_imu)
gz = yaw_rate + np.random.normal(0, 0.5, n_imu)

# =========================
# DEPTH FRAMES
# =========================
base = np.tile(np.linspace(40, 220, FRAME_W), (FRAME_H, 1))
frames = []

for i in range(n_depth):
    shift = int(turn[i] * 6)
    frame = np.roll(base, shift, axis=1)
    frame += np.random.normal(0, 5, frame.shape)
    frames.append(np.clip(frame, 0, 255))

# =========================
# DATASET POR JANELA
# =========================
sample_id = 0
for i in range(0, n_depth - WINDOW_FRAMES, WINDOW_FRAMES):
    sample_id += 1
    sdir = OUT_DIR / f"sample_{sample_id:05d}"
    ensure_dir(sdir / "depth_frames")

    for k in range(WINDOW_FRAMES):
        save_pgm(sdir / "depth_frames" / f"{k+1:04d}.pgm", frames[i+k])

    t0 = i / DEPTH_FPS
    t1 = (i + WINDOW_FRAMES) / DEPTH_FPS

    ia = int(t0 * IMU_HZ)
    ib = int(t1 * IMU_HZ)

    imu = np.column_stack([t_imu[ia:ib], ax[ia:ib], ay[ia:ib], az[ia:ib],
                           gx[ia:ib], gy[ia:ib], gz[ia:ib]])

    np.savetxt(sdir / "imu_window.csv", imu,
               delimiter=",", header="t,ax,ay,az,gx,gy,gz",
               comments="")

    label = 1 if np.mean(gz[ia:ib]) > 0 else 0
    (sdir / "label.txt").write_text(str(label))

    meta = {
        "label": "right" if label == 1 else "left",
        "frames": WINDOW_FRAMES,
        "depth_fps": DEPTH_FPS,
        "imu_hz": IMU_HZ
    }
    (sdir / "meta.json").write_text(json.dumps(meta, indent=2))

print("✅ Dataset gerado com sucesso!")
print(f"📁 Pasta criada: {OUT_DIR.resolve()}")
print(f"📦 Total de amostras: {sample_id}")
