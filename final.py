# Simulacao_RNA.py
# Rede Neural Artificial (MLP) + Simulacao temporal com dataset_5min_depth_imu
# Requisitos: numpy, matplotlib
#
# O que este script gera (na pasta outputs/):
# - entrada_exemplo.png  -> ilustração das ENTRADAS (SEM mostrar imagem da câmera)
# - loss_exec*.png       -> loss por execução (3 rodadas)
# - direcao_exec*.png    -> real vs previsto no tempo
# - traj_exec*.png       -> trajetória referência (labels) vs prevista (MLP)
# - cm_exec*.png         -> matriz de confusão (opcional, mas mantida)
#
# Métrica "robótica" (além da matriz de confusão):
# - erro de trajetória (MAE e RMSE) comparando trajetória integrada pelos labels vs pelas previsões
# - taxa de trocas (switch_rate) para avaliar estabilidade das decisões no tempo

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CORRECAO ENCODING (Windows)
# =========================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# =========================
# CONFIG
# =========================
DATASET_DIR = Path(__file__).parent / "dataset_5min_depth_imu"

WINDOW_FRAMES = 8

# Para SIMULACAO 2D (aproximada)
STEP_METERS = 0.25
TURN_DEG = 12

# Treino/teste
TRAIN_SPLIT = 0.80

# Saída de figuras
OUT_DIR = Path(__file__).parent / "outputs"
OUT_DIR.mkdir(exist_ok=True)

# =========================
# LEITOR PGM (P5)
# =========================
def read_pgm_p5(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Formato inválido (esperado P5): {path}")

        def read_non_comment():
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            return line

        dims = read_non_comment().strip()
        w, h = map(int, dims.split())
        maxval = int(read_non_comment().strip())
        if maxval != 255:
            raise ValueError(f"Maxval inesperado {maxval} em {path}")

        data = f.read(w * h)
        img = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
        return img

# =========================
# FEATURE ENGINEERING
# - IMU: media e desvio de ax..gz (12 features)
# - Depth: features simples por frame: media e desvio (2*WINDOW_FRAMES)
# Total: 12 + 16 = 28 features
# =========================
def imu_features(csv_path: Path) -> np.ndarray:
    # header: t,ax,ay,az,gx,gy,gz
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    imu = data[:, 1:7]  # ax..gz
    feat = np.concatenate([imu.mean(axis=0), imu.std(axis=0)], axis=0)  # 12
    return feat.astype(np.float32)

def depth_features(depth_dir: Path) -> np.ndarray:
    feats = []
    for k in range(1, WINDOW_FRAMES + 1):
        img = read_pgm_p5(depth_dir / f"{k:04d}.pgm").astype(np.float32) / 255.0
        feats.append(img.mean())
        feats.append(img.std())
    return np.array(feats, dtype=np.float32)  # 16

# =========================
# CARREGAR DATASET EM ORDEM TEMPORAL
# =========================
def load_dataset(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Nao achei a pasta: {dataset_dir.resolve()}")

    samples = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    if not samples:
        raise FileNotFoundError("Nao achei pastas sample_ dentro do dataset.")

    X = []
    y = []

    for s in samples:
        depth_dir = s / "depth_frames"
        imu_csv = s / "imu_window.csv"
        label_file = s / "label.txt"

        if not (depth_dir.exists() and imu_csv.exists() and label_file.exists()):
            raise FileNotFoundError(f"Sample incompleto: {s}")

        f_imu = imu_features(imu_csv)
        f_depth = depth_features(depth_dir)
        feat = np.concatenate([f_imu, f_depth], axis=0)  # 28

        label = int(label_file.read_text().strip())  # 0 esq, 1 dir
        X.append(feat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, samples

# =========================
# NORMALIZACAO
# =========================
def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma

def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma

# =========================
# MLP (Rede Neural Artificial) em NumPy
# - 1 camada escondida
# - sigmoid na saída (binária)
# =========================
class MLPBinary:
    def __init__(self, in_dim, hidden_dim=32, lr=0.01, seed=None):
        if seed is not None:
            np.random.seed(seed)

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Xavier/He-like init
        self.W1 = np.random.randn(in_dim, hidden_dim).astype(np.float32) * np.sqrt(2.0 / in_dim)
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)

        self.W2 = np.random.randn(hidden_dim, 1).astype(np.float32) * np.sqrt(2.0 / hidden_dim)
        self.b2 = np.zeros((1,), dtype=np.float32)

    def relu(self, z):
        return np.maximum(0, z)

    def relu_grad(self, z):
        return (z > 0).astype(np.float32)

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        p = self.sigmoid(z2)  # prob classe 1
        return z1, a1, z2, p

    def loss(self, y_true, p):
        y_true = y_true.reshape(-1, 1).astype(np.float32)
        eps = 1e-7
        return -np.mean(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps))

    def train(self, X, y, epochs=30, batch_size=32, verbose=True):
        n = X.shape[0]
        history = []

        for ep in range(1, epochs + 1):
            idx = np.random.permutation(n)
            Xs = X[idx]
            ys = y[idx]

            for i in range(0, n, batch_size):
                Xb = Xs[i:i+batch_size]
                yb = ys[i:i+batch_size].reshape(-1, 1).astype(np.float32)

                z1, a1, _, p = self.forward(Xb)

                dz2 = (p - yb) / len(yb)  # BCE+sigmoid
                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * self.relu_grad(z1)

                dW1 = Xb.T @ dz1
                db1 = dz1.sum(axis=0)

                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            _, _, _, p_all = self.forward(X)
            L = self.loss(y, p_all)
            history.append(L)

            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                pred = (p_all.flatten() >= 0.5).astype(int)
                acc = (pred == y).mean()
                print(f"Epoca {ep:02d} | loss={L:.4f} | acc={acc:.3f}")

        return history

    def predict_proba(self, X):
        _, _, _, p = self.forward(X)
        return p.flatten()

    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)

# =========================
# (1) ILUSTRACAO DOS DADOS DE ENTRADA (SEM IMAGEM)
# Mostra:
# - Depth por frame: média e desvio (8 pontos)
# - Depth por setores (Esq/Centro/Dir): heatmap 8x3
# - IMU no tempo (ax..gz)
# - Vetor de features 28D (entrada da MLP)
# =========================
def plot_input_example_no_image(sample_dir: Path, out_path: Path):
    depth_dir = sample_dir / "depth_frames"
    imu_csv = sample_dir / "imu_window.csv"
    label_path = sample_dir / "label.txt"
    label = int(label_path.read_text().strip())  # 0 esq, 1 dir

    # Depth -> resumo por frame + setores
    depth_means = []
    depth_stds = []
    sector_means = []  # (WINDOW_FRAMES, 3) = esq/centro/dir

    for k in range(1, WINDOW_FRAMES + 1):
        img = read_pgm_p5(depth_dir / f"{k:04d}.pgm").astype(np.float32) / 255.0

        depth_means.append(float(img.mean()))
        depth_stds.append(float(img.std()))

        h, w = img.shape
        s1 = float(img[:, : w//3].mean())
        s2 = float(img[:, w//3 : 2*w//3].mean())
        s3 = float(img[:, 2*w//3 :].mean())
        sector_means.append([s1, s2, s3])

    depth_means = np.array(depth_means, dtype=np.float32)
    depth_stds = np.array(depth_stds, dtype=np.float32)
    sector_means = np.array(sector_means, dtype=np.float32)

    # IMU série temporal
    data = np.genfromtxt(imu_csv, delimiter=",", skip_header=1)
    t = data[:, 0]
    imu = data[:, 1:7]  # ax..gz

    # features 28D (entrada da MLP)
    f_imu = imu_features(imu_csv)
    f_depth = depth_features(depth_dir)
    feat = np.concatenate([f_imu, f_depth], axis=0)

    # Plot painel 2x2
    plt.figure(figsize=(12, 9))

    # (a) Depth: média e desvio por frame
    ax1 = plt.subplot(2, 2, 1)
    x = np.arange(1, WINDOW_FRAMES + 1)
    ax1.plot(x, depth_means, marker="o", label="média(depth)")
    ax1.plot(x, depth_stds, marker="o", label="desvio(depth)")
    ax1.set_title(f"Depth na janela (resumo por frame) | label={label} ({'Esq' if label==0 else 'Dir'})")
    ax1.set_xlabel("Frame na janela (1..8)")
    ax1.set_ylabel("valor normalizado (0..1)")
    ax1.grid(True)
    ax1.legend()

    # (b) Depth por setores: heatmap 8x3
    ax2 = plt.subplot(2, 2, 2)
    im = ax2.imshow(sector_means, aspect="auto")
    ax2.set_title("Depth por setores (Esq/Centro/Dir) — heatmap")
    ax2.set_xlabel("Setor")
    ax2.set_ylabel("Frame na janela")
    ax2.set_xticks([0, 1, 2])
    ax2.set_xticklabels(["Esq", "Centro", "Dir"])
    ax2.set_yticks(np.arange(WINDOW_FRAMES))
    ax2.set_yticklabels([str(i) for i in range(1, WINDOW_FRAMES + 1)])
    plt.colorbar(im, ax=ax2, fraction=0.046, pad=0.04)

    # (c) IMU série temporal
    ax3 = plt.subplot(2, 2, 3)
    for k in range(imu.shape[1]):
        ax3.plot(t, imu[:, k])
    ax3.set_title("IMU na janela (ax..gz)")
    ax3.set_xlabel("t")
    ax3.set_ylabel("valor")
    ax3.grid(True)

    # (d) Vetor 28D
    ax4 = plt.subplot(2, 2, 4)
    ax4.bar(np.arange(len(feat)), feat)
    ax4.set_title("Vetor de entrada da MLP (features 28D)")
    ax4.set_xlabel("índice da feature")
    ax4.set_ylabel("valor")
    ax4.grid(True)

    plt.tight_layout()
    plt.savefig(out_path, dpi=200)
    plt.close()

# =========================
# Trajetoria 2D a partir de comandos (0=esq, 1=dir)
# =========================
def integrate_trajectory(commands, step=STEP_METERS, turn_deg=TURN_DEG):
    xs, ys = [0.0], [0.0]
    theta = 0.0
    for cmd in commands:
        if cmd == 1:  # direita
            theta -= np.deg2rad(turn_deg)
        else:         # esquerda
            theta += np.deg2rad(turn_deg)

        xs.append(xs[-1] + step * np.cos(theta))
        ys.append(ys[-1] + step * np.sin(theta))
    return np.array(xs), np.array(ys)

# =========================
# METRICA ROBOTICA (NAO matriz de confusao)
# Erro de trajetória: distancia ponto-a-ponto entre:
# - trajeto integrando labels reais
# - trajeto integrando previsoes
# =========================
def trajectory_error(xs_ref, ys_ref, xs_pred, ys_pred):
    n = min(len(xs_ref), len(xs_pred))
    dx = xs_ref[:n] - xs_pred[:n]
    dy = ys_ref[:n] - ys_pred[:n]
    dist = np.sqrt(dx*dx + dy*dy)
    return float(dist.mean()), float(np.sqrt((dist**2).mean()))  # MAE e RMSE

# =========================
# VARIAS RODADAS
# =========================
def run_one(seed, X_train_n, y_train, X_test_n, y_test, run_id):
    print(f"\n=== Execucao {run_id} | SEED={seed} ===")
    mlp = MLPBinary(in_dim=X_train_n.shape[1], hidden_dim=32, lr=0.02, seed=seed)
    loss_hist = mlp.train(X_train_n, y_train, epochs=30, batch_size=32, verbose=True)

    pred_test = mlp.predict(X_test_n)
    acc_test = (pred_test == y_test).mean()

    # matriz de confusão (mantém)
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_test, pred_test):
        cm[yt, yp] += 1

    # trajetoria: referencia = labels reais; prevista = predicoes
    xs_ref, ys_ref = integrate_trajectory(y_test)
    xs_pred, ys_pred = integrate_trajectory(pred_test)
    traj_mae, traj_rmse = trajectory_error(xs_ref, ys_ref, xs_pred, ys_pred)

    # “chattering” (trocas de decisão)
    switches = int(np.sum(pred_test[1:] != pred_test[:-1]))
    switch_rate = float(switches / max(1, (len(pred_test) - 1)))

    # --- salvar figuras desta execução ---
    # loss
    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("Epoca")
    plt.ylabel("Loss (BCE)")
    plt.title(f"Treinamento - Loss | Execucao {run_id}")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"loss_exec{run_id}.png", dpi=200)
    plt.close()

    # real vs previsto (temporal)
    t_test = np.arange(len(y_test))
    plt.figure()
    plt.step(t_test, y_test, where="post", label="Real (label)")
    plt.step(t_test, pred_test, where="post", label="Previsto (MLP)")
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    plt.xlabel("Amostra (ordem temporal no teste)")
    plt.ylabel("Direcao")
    plt.title(f"Direcao real vs prevista | Execucao {run_id}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"direcao_exec{run_id}.png", dpi=200)
    plt.close()

    # trajetorias: ref vs pred (métrica robótica)
    plt.figure()
    plt.plot(xs_ref, ys_ref, label="Trajetoria referencia (labels)")
    plt.plot(xs_pred, ys_pred, label="Trajetoria prevista (MLP)")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Trajetoria 2D: referencia vs prevista | Execucao {run_id}")
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"traj_exec{run_id}.png", dpi=200)
    plt.close()

    # matriz de confusão como imagem
    plt.figure()
    plt.imshow(cm)
    plt.title(f"Matriz de Confusao (Teste) | Execucao {run_id}")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.xticks([0, 1], ["Esquerda", "Direita"])
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig(OUT_DIR / f"cm_exec{run_id}.png", dpi=200)
    plt.close()

    return {
        "run": run_id,
        "seed": seed,
        "acc_test": float(acc_test),
        "traj_mae": traj_mae,
        "traj_rmse": traj_rmse,
        "switches": switches,
        "switch_rate": switch_rate,
        "cm": cm
    }

# =========================
# MAIN
# =========================
def main():
    print("Lendo dataset...")
    X, y, samples = load_dataset(DATASET_DIR)

    print("Dataset carregado.")
    print("Total amostras:", len(y))
    print("Esquerda (0):", int((y == 0).sum()), "| Direita (1):", int((y == 1).sum()))

    # Split temporal: treina com início, testa com final
    N = len(y)
    n_train = int(N * TRAIN_SPLIT)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # Normalização baseada no treino
    mu, sigma = standardize_fit(X_train)
    X_train_n = standardize_apply(X_train, mu, sigma)
    X_test_n = standardize_apply(X_test, mu, sigma)

    # (1) Figura de ilustração das entradas (SEM imagem)
    example_sample = samples[0]
    plot_input_example_no_image(example_sample, OUT_DIR / "entrada_exemplo.png")
    print(f"\n[OK] Figura de entrada (sem imagem) salva em: {OUT_DIR / 'entrada_exemplo.png'}")

    # (2) Rodar varias execuções (3 rodadas)
    seeds = [42, 123, 999]
    results = []
    for i, sd in enumerate(seeds, start=1):
        res = run_one(sd, X_train_n, y_train, X_test_n, y_test, run_id=i)
        results.append(res)

    # Resumo final (tabela simples)
    print("\n====================")
    print("RESUMO DAS EXECUCOES")
    print("====================")
    print("run | seed | acc_test | traj_MAE | traj_RMSE | switches | switch_rate")
    for r in results:
        print(f"{r['run']:>3} | {r['seed']:>4} | {r['acc_test']:.4f} | "
              f"{r['traj_mae']:.4f} | {r['traj_rmse']:.4f} | "
              f"{r['switches']:>8} | {r['switch_rate']:.4f}")

    accs = np.array([r["acc_test"] for r in results], dtype=np.float32)
    maes = np.array([r["traj_mae"] for r in results], dtype=np.float32)
    rmses = np.array([r["traj_rmse"] for r in results], dtype=np.float32)

    print("\nMEDIAS (3 rodadas):")
    print(f"Acuracia teste: {accs.mean():.4f} ± {accs.std():.4f}")
    print(f"Erro traj (MAE): {maes.mean():.4f} ± {maes.std():.4f}")
    print(f"Erro traj (RMSE): {rmses.mean():.4f} ± {rmses.std():.4f}")

    print(f"\nFiguras salvas em: {OUT_DIR.resolve()}")
    print("Fim.")# Simulacao_RNA.py
# Rede Neural Artificial (MLP) + Simulacao temporal com dataset_5min_depth_imu
# + MULTIPLAS RODADAS (seeds diferentes) + FIGURAS DOS EXPERIMENTOS
# Requisitos: numpy, matplotlib
#
# Saídas geradas (PNG/TXT) na mesma pasta do script:
# - fig_loss_multiplas_rodadas.png
# - fig_boxplot_acuracia_teste.png
# - fig_temporal_melhor_rodada.png
# - fig_matriz_confusao_melhor_rodada.png
# - fig_trajetoria_melhor_rodada.png
# - resumo_rodadas.txt

import sys
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# =========================
# CORRECAO ENCODING (Windows)
# =========================
try:
    sys.stdout.reconfigure(encoding="utf-8")
except Exception:
    pass

# =========================
# CONFIG GERAL
# =========================
DATASET_DIR = Path(__file__).parent / "dataset_5min_depth_imu"

WINDOW_FRAMES = 8

# Para SIMULACAO 2D (aproximada)
STEP_METERS = 0.25
TURN_DEG = 12

# Split temporal
TRAIN_SPLIT = 0.80

# =========================
# CONFIG EXPERIMENTO (RODADAS)
# =========================
N_RODADAS = 10
BASE_SEED = 100

EPOCHS = 30
BATCH_SIZE = 32
HIDDEN_DIM = 32
LR = 0.02


# =========================
# LEITOR PGM (P5)
# =========================
def read_pgm_p5(path: Path) -> np.ndarray:
    with open(path, "rb") as f:
        magic = f.readline().strip()
        if magic != b"P5":
            raise ValueError(f"Formato inválido (esperado P5): {path}")

        def read_non_comment():
            line = f.readline()
            while line.startswith(b"#"):
                line = f.readline()
            return line

        dims = read_non_comment().strip()
        w, h = map(int, dims.split())
        maxval = int(read_non_comment().strip())
        if maxval != 255:
            raise ValueError(f"Maxval inesperado {maxval} em {path}")

        data = f.read(w * h)
        img = np.frombuffer(data, dtype=np.uint8).reshape((h, w))
        return img


# =========================
# FEATURE ENGINEERING
# - IMU: media e desvio de ax..gz (12 features)
# - Depth: media e desvio por frame (2*WINDOW_FRAMES) = 16
# Total: 28 features
# =========================
def imu_features(csv_path: Path) -> np.ndarray:
    # header: t,ax,ay,az,gx,gy,gz
    data = np.genfromtxt(csv_path, delimiter=",", skip_header=1)
    imu = data[:, 1:7]  # ax..gz
    feat = np.concatenate([imu.mean(axis=0), imu.std(axis=0)], axis=0)  # 12
    return feat.astype(np.float32)


def depth_features(depth_dir: Path) -> np.ndarray:
    feats = []
    for k in range(1, WINDOW_FRAMES + 1):
        img = read_pgm_p5(depth_dir / f"{k:04d}.pgm").astype(np.float32) / 255.0
        feats.append(float(img.mean()))
        feats.append(float(img.std()))
    return np.array(feats, dtype=np.float32)  # 16


# =========================
# CARREGAR DATASET EM ORDEM TEMPORAL
# =========================
def load_dataset(dataset_dir: Path):
    if not dataset_dir.exists():
        raise FileNotFoundError(f"Nao achei a pasta: {dataset_dir.resolve()}")

    samples = sorted([p for p in dataset_dir.iterdir() if p.is_dir() and p.name.startswith("sample_")])
    if not samples:
        raise FileNotFoundError("Nao achei pastas sample_ dentro do dataset.")

    X = []
    y = []

    for s in samples:
        depth_dir = s / "depth_frames"
        imu_csv = s / "imu_window.csv"
        label_file = s / "label.txt"

        if not (depth_dir.exists() and imu_csv.exists() and label_file.exists()):
            raise FileNotFoundError(f"Sample incompleto: {s}")

        f_imu = imu_features(imu_csv)
        f_depth = depth_features(depth_dir)
        feat = np.concatenate([f_imu, f_depth], axis=0)  # 28

        label = int(label_file.read_text().strip())  # 0 esq, 1 dir

        X.append(feat)
        y.append(label)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int32)
    return X, y, samples


# =========================
# NORMALIZACAO
# =========================
def standardize_fit(X):
    mu = X.mean(axis=0)
    sigma = X.std(axis=0)
    sigma[sigma < 1e-8] = 1.0
    return mu, sigma


def standardize_apply(X, mu, sigma):
    return (X - mu) / sigma


# =========================
# MLP (Binária) em NumPy
# =========================
class MLPBinary:
    def __init__(self, in_dim, hidden_dim=32, lr=0.01):
        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.lr = float(lr)

        # Inicialização (He/Xavier-like)
        self.W1 = (np.random.randn(self.in_dim, self.hidden_dim).astype(np.float32)
                   * np.sqrt(2.0 / self.in_dim))
        self.b1 = np.zeros((self.hidden_dim,), dtype=np.float32)

        self.W2 = (np.random.randn(self.hidden_dim, 1).astype(np.float32)
                   * np.sqrt(2.0 / self.hidden_dim))
        self.b2 = np.zeros((1,), dtype=np.float32)

    @staticmethod
    def relu(z):
        return np.maximum(0, z)

    @staticmethod
    def relu_grad(z):
        return (z > 0).astype(np.float32)

    @staticmethod
    def sigmoid(z):
        return 1.0 / (1.0 + np.exp(-z))

    def forward(self, X):
        z1 = X @ self.W1 + self.b1
        a1 = self.relu(z1)
        z2 = a1 @ self.W2 + self.b2
        p = self.sigmoid(z2)  # prob classe 1
        return z1, a1, z2, p

    @staticmethod
    def loss(y_true, p):
        y_true = y_true.reshape(-1, 1).astype(np.float32)
        eps = 1e-7
        return float(-np.mean(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps)))

    def train(self, X, y, epochs=30, batch_size=32, verbose=True):
        n = X.shape[0]
        history = []

        for ep in range(1, epochs + 1):
            idx = np.random.permutation(n)
            Xs = X[idx]
            ys = y[idx]

            for i in range(0, n, batch_size):
                Xb = Xs[i:i + batch_size]
                yb = ys[i:i + batch_size].reshape(-1, 1).astype(np.float32)

                z1, a1, z2, p = self.forward(Xb)

                # dL/dz2 (BCE+sigmoid) = p - y
                dz2 = (p - yb) / max(1, len(yb))

                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * self.relu_grad(z1)

                dW1 = Xb.T @ dz1
                db1 = dz1.sum(axis=0)

                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            _, _, _, p_all = self.forward(X)
            L = self.loss(y, p_all)
            history.append(L)

            if verbose and (ep == 1 or ep % 5 == 0 or ep == epochs):
                pred = (p_all.flatten() >= 0.5).astype(int)
                acc = float((pred == y).mean())
                print(f"Epoca {ep:02d} | loss={L:.4f} | acc={acc:.3f}")

        return history

    def predict_proba(self, X):
        _, _, _, p = self.forward(X)
        return p.flatten()

    def predict(self, X):
        p = self.predict_proba(X)
        return (p >= 0.5).astype(int)


# =========================
# UTIL
# =========================
def set_seed(seed: int):
    np.random.seed(int(seed))


def confusion_matrix_2x2(y_true, y_pred):
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_true, y_pred):
        cm[int(yt), int(yp)] += 1
    return cm


def run_one_experiment(seed: int, hidden_dim=32, lr=0.02, epochs=30, batch_size=32, verbose_train=False):
    set_seed(seed)

    X, y, _ = load_dataset(DATASET_DIR)

    N = len(y)
    n_train = int(N * TRAIN_SPLIT)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    mu, sigma = standardize_fit(X_train)
    X_train_n = standardize_apply(X_train, mu, sigma)
    X_test_n = standardize_apply(X_test, mu, sigma)

    mlp = MLPBinary(in_dim=X.shape[1], hidden_dim=hidden_dim, lr=lr)
    loss_hist = mlp.train(X_train_n, y_train, epochs=epochs, batch_size=batch_size, verbose=verbose_train)

    pred_test = mlp.predict(X_test_n)
    acc_test = float((pred_test == y_test).mean())
    cm = confusion_matrix_2x2(y_test, pred_test)

    return {
        "seed": seed,
        "acc_test": acc_test,
        "loss_hist": np.array(loss_hist, dtype=np.float32),
        "cm": cm,
        "y_test": y_test,
        "pred_test": pred_test
    }


def simulate_trajectory(commands):
    xs, ys = [0.0], [0.0]
    theta = 0.0
    for cmd in commands:
        if int(cmd) == 1:  # direita
            theta -= np.deg2rad(TURN_DEG)
        else:  # esquerda
            theta += np.deg2rad(TURN_DEG)
        xs.append(xs[-1] + STEP_METERS * np.cos(theta))
        ys.append(ys[-1] + STEP_METERS * np.sin(theta))
    return np.array(xs, dtype=np.float32), np.array(ys, dtype=np.float32)


# =========================
# MAIN
# =========================
def main():
    print("===============================================")
    print("MLP NumPy + Simulação temporal + Múltiplas Rodadas")
    print("===============================================")
    print(f"Dataset: {DATASET_DIR.resolve()}")
    print(f"Rodadas: {N_RODADAS} | BASE_SEED={BASE_SEED}")
    print(f"Arquitetura: 28 -> {HIDDEN_DIM} -> 1 | LR={LR} | epochs={EPOCHS} | batch={BATCH_SIZE}")
    print("")

    # Sanidade: carrega uma vez para mostrar info
    X, y, _ = load_dataset(DATASET_DIR)
    print("Dataset carregado.")
    print("Total amostras:", len(y))
    print("Esquerda (0):", int((y == 0).sum()), "| Direita (1):", int((y == 1).sum()))
    print("")

    # =========================
    # Rodar múltiplas rodadas
    # =========================
    results = []
    for k in range(N_RODADAS):
        seed = BASE_SEED + k
        r = run_one_experiment(
            seed=seed,
            hidden_dim=HIDDEN_DIM,
            lr=LR,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose_train=False
        )
        results.append(r)
        print(f"Rodada {k+1:02d}/{N_RODADAS} | seed={seed} | acc_test={r['acc_test']:.4f}")

    accs = np.array([r["acc_test"] for r in results], dtype=np.float32)
    mean_acc = float(accs.mean())
    std_acc = float(accs.std(ddof=1)) if len(accs) > 1 else 0.0

    print("\nResumo (Teste):")
    print(f"Acurácia média = {mean_acc:.4f}  | desvio padrão = {std_acc:.4f}")

    # =========================
    # FIGURA A) Loss de todas as rodadas
    # =========================
    plt.figure()
    for r in results:
        plt.plot(r["loss_hist"], alpha=0.6)
    plt.xlabel("Época")
    plt.ylabel("Loss (BCE)")
    plt.title(f"Treinamento - Loss em {N_RODADAS} rodadas (seeds diferentes)")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig("fig_loss_multiplas_rodadas.png", dpi=200)
    plt.show()

    # =========================
    # FIGURA B) Boxplot acurácia no teste
    # =========================
    plt.figure()
    plt.boxplot(accs, vert=True, showmeans=True)
    plt.ylabel("Acurácia no teste")
    plt.title(f"Acurácia no teste em {N_RODADAS} rodadas\nmédia={mean_acc:.3f} | dp={std_acc:.3f}")
    plt.grid(True, axis="y")
    plt.tight_layout()
    plt.savefig("fig_boxplot_acuracia_teste.png", dpi=200)
    plt.show()

    # =========================
    # TXT com resumo (pra colar no relatório)
    # =========================
    with open("resumo_rodadas.txt", "w", encoding="utf-8") as f:
        f.write(f"Experimento com {N_RODADAS} rodadas (seeds diferentes)\n")
        f.write(f"Arquitetura: 28 -> {HIDDEN_DIM} -> 1 | LR={LR} | epochs={EPOCHS} | batch={BATCH_SIZE}\n")
        f.write("Split temporal: treino no início da sequência e teste no final (preservando ordem temporal)\n\n")
        for r in results:
            f.write(f"seed={r['seed']}, acc_test={r['acc_test']:.6f}\n")
        f.write("\n")
        f.write(f"MEDIA_ACURACIA_TESTE={mean_acc:.6f}\n")
        f.write(f"DP_ACURACIA_TESTE={std_acc:.6f}\n")

    # =========================
    # Escolher a melhor rodada para figuras detalhadas
    # =========================
    best_idx = int(np.argmax(accs))
    best = results[best_idx]
    y_test = best["y_test"]
    pred_test = best["pred_test"]
    cm = best["cm"]

    print(f"\nMelhor rodada para figuras detalhadas: seed={best['seed']} | acc={best['acc_test']:.4f}")

    # =========================
    # FIGURA C) Simulação temporal (real vs previsto)
    # =========================
    t_test = np.arange(len(y_test))
    plt.figure()
    plt.step(t_test, y_test, where="post", label="Real (label)")
    plt.step(t_test, pred_test, where="post", label="Previsto (MLP)")
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    plt.xlabel("Amostra (ordem temporal no teste)")
    plt.ylabel("Direção")
    plt.title("Simulação temporal (melhor rodada): real vs previsto")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig("fig_temporal_melhor_rodada.png", dpi=200)
    plt.show()

    # =========================
    # FIGURA D) Matriz de confusão (melhor rodada)
    # =========================
    plt.figure()
    plt.imshow(cm)
    plt.title("Matriz de Confusão (Teste) - melhor rodada")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.xticks([0, 1], ["Esquerda", "Direita"])
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.tight_layout()
    plt.savefig("fig_matriz_confusao_melhor_rodada.png", dpi=200)
    plt.show()

    # =========================
    # FIGURA E) Trajetória 2D aproximada (melhor rodada)
    # =========================
    xs, ys = simulate_trajectory(pred_test)
    plt.figure()
    plt.plot(xs, ys)
    plt.scatter(xs[0], ys[0], marker="o")
    plt.scatter(xs[-1], ys[-1], marker="x")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajetória 2D aproximada (melhor rodada)")
    plt.tight_layout()
    plt.savefig("fig_trajetoria_melhor_rodada.png", dpi=200)
    plt.show()

    print("\nArquivos gerados:")
    print("- fig_loss_multiplas_rodadas.png")
    print("- fig_boxplot_acuracia_teste.png")
    print("- fig_temporal_melhor_rodada.png")
    print("- fig_matriz_confusao_melhor_rodada.png")
    print("- fig_trajetoria_melhor_rodada.png")
    print("- resumo_rodadas.txt")
    print("\nFim.")


if __name__ == "__main__":
    main()


if __name__ == "__main__":
    main()
