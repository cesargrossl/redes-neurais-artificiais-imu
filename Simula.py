# Simulacao_RNA.py
# Rede Neural Artificial (MLP) + Simulacao temporal com dataset_5min_depth_imu
# Requisitos: numpy, matplotlib

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
    pass  # se não suportar, segue sem travar

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
SEED = 42

np.random.seed(SEED)

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
    def __init__(self, in_dim, hidden_dim=32, lr=0.01):
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Xavier init
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
        # BCE
        y_true = y_true.reshape(-1, 1).astype(np.float32)
        eps = 1e-7
        return -np.mean(y_true * np.log(p + eps) + (1 - y_true) * np.log(1 - p + eps))

    def train(self, X, y, epochs=30, batch_size=32, verbose=True):
        n = X.shape[0]
        history = []

        for ep in range(1, epochs + 1):
            # shuffle treino
            idx = np.random.permutation(n)
            Xs = X[idx]
            ys = y[idx]

            for i in range(0, n, batch_size):
                Xb = Xs[i:i+batch_size]
                yb = ys[i:i+batch_size].reshape(-1, 1).astype(np.float32)

                z1, a1, z2, p = self.forward(Xb)

                # gradiente BCE com sigmoid: dL/dz2 = p - y
                dz2 = (p - yb) / len(yb)  # normaliza no batch

                dW2 = a1.T @ dz2
                db2 = dz2.sum(axis=0)

                da1 = dz2 @ self.W2.T
                dz1 = da1 * self.relu_grad(z1)

                dW1 = Xb.T @ dz1
                db1 = dz1.sum(axis=0)

                # update
                self.W1 -= self.lr * dW1
                self.b1 -= self.lr * db1
                self.W2 -= self.lr * dW2
                self.b2 -= self.lr * db2

            # loss no fim da época
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
# MAIN
# =========================
def main():
    print("Lendo dataset...")
    X, y, samples = load_dataset(DATASET_DIR)

    print("Dataset carregado.")
    print("Total amostras:", len(y))
    print("Esquerda (0):", int((y == 0).sum()), "| Direita (1):", int((y == 1).sum()))

    # Split temporal: treina com início, testa com final (simulação realista)
    N = len(y)
    n_train = int(N * TRAIN_SPLIT)

    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # Normalização baseada no treino
    mu, sigma = standardize_fit(X_train)
    X_train_n = standardize_apply(X_train, mu, sigma)
    X_test_n = standardize_apply(X_test, mu, sigma)

    # Treinar MLP
    print("\nTreinando Rede Neural Artificial (MLP em NumPy)...")
    mlp = MLPBinary(in_dim=X.shape[1], hidden_dim=32, lr=0.02)
    loss_hist = mlp.train(X_train_n, y_train, epochs=30, batch_size=32, verbose=True)

    # Avaliar
    pred_test = mlp.predict(X_test_n)
    acc_test = (pred_test == y_test).mean()
    print("\nResultado no TESTE (sequencia final):")
    print("Acuracia:", float(acc_test))

    # Matriz de confusão 2x2
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_test, pred_test):
        cm[yt, yp] += 1
    print("Matriz de confusao (linhas=real, colunas=previsto):")
    print(cm)

    # =========================
    # 1) Curva de loss
    # =========================
    plt.figure()
    plt.plot(loss_hist)
    plt.xlabel("Epoca")
    plt.ylabel("Loss (BCE)")
    plt.title("Treinamento - Loss")
    plt.grid(True)
    plt.show()

    # =========================
    # 2) Simulacao temporal (REAL vs PREVISTO)
    # =========================
    t_test = np.arange(len(y_test))  # eixo por amostra
    plt.figure()
    plt.step(t_test, y_test, where="post", label="Real (label)")
    plt.step(t_test, pred_test, where="post", label="Previsto (MLP)")
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    plt.xlabel("Amostra (ordem temporal no teste)")
    plt.ylabel("Direcao")
    plt.title("Simulacao temporal: direcao real vs prevista")
    plt.grid(True)
    plt.legend()
    plt.show()

    # =========================
    # 3) Trajetoria 2D aproximada (usando PREVISTO)
    # =========================
    xs, ys = [0.0], [0.0]
    theta = 0.0

    for cmd in pred_test:
        if cmd == 1:  # direita
            theta -= np.deg2rad(TURN_DEG)
        else:         # esquerda
            theta += np.deg2rad(TURN_DEG)

        x_new = xs[-1] + STEP_METERS * np.cos(theta)
        y_new = ys[-1] + STEP_METERS * np.sin(theta)
        xs.append(x_new)
        ys.append(y_new)

    plt.figure()
    plt.plot(xs, ys)
    plt.scatter(xs[0], ys[0], marker="o")
    plt.scatter(xs[-1], ys[-1], marker="x")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title("Trajetoria 2D aproximada (integrando previsoes esquerda/direita)")
    plt.show()

    # =========================
    # 4) Mostrar matriz de confusão como imagem
    # =========================
    plt.figure()
    plt.imshow(cm)
    plt.title("Matriz de Confusao (Teste)")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.xticks([0, 1], ["Esquerda", "Direita"])
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.show()

    print("\nFim. Simulacao concluida.")

if __name__ == "__main__":
    main()
