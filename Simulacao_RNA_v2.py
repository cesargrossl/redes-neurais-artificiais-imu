# Simulacao_RNA_v2.py
# Rede Neural Artificial (MLP) + Simulação temporal com dataset_5min_depth_imu
# Extras (atendendo às observações):
#  - Ilustração dos dados de entrada (frames depth + sinais IMU de uma amostra)
#  - Execução de múltiplas rodadas (seeds) e gráficos agregados
#  - Métricas orientadas a robótica/controle (trajetória e estabilidade temporal), além de matriz de confusão
#
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

# Experimentos (múltiplas rodadas)
RUN_SEEDS = [1, 7, 21, 42, 77]   # pode aumentar

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
# LEITURA RAW (para figura)
# =========================
def load_one_sample(sample_dir: Path):
    depth_dir = sample_dir / "depth_frames"
    imu_csv = sample_dir / "imu_window.csv"
    label_file = sample_dir / "label.txt"

    # depth
    frames = []
    for k in range(1, WINDOW_FRAMES + 1):
        img = read_pgm_p5(depth_dir / f"{k:04d}.pgm").astype(np.float32) / 255.0
        frames.append(img)

    # imu raw
    data = np.genfromtxt(imu_csv, delimiter=",", skip_header=1)
    t = data[:, 0]
    imu = data[:, 1:7]  # ax..gz

    # label
    label = int(label_file.read_text().strip())  # 0 esq, 1 dir

    return np.stack(frames, axis=0), t, imu, label

# =========================
# FEATURE ENGINEERING
# - IMU: media e desvio de ax..gz (12 features)
# - Depth: features simples por frame: media e desvio (2*WINDOW_FRAMES)
# Total: 12 + 16 = 28 features
# =========================
def imu_features(csv_path: Path) -> np.ndarray:
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
# MÉTRICAS (além da matriz de confusão)
# =========================
def binary_logloss(y_true, p):
    y_true = y_true.astype(np.float32)
    p = np.clip(p.astype(np.float32), 1e-7, 1 - 1e-7)
    return float(-np.mean(y_true * np.log(p) + (1 - y_true) * np.log(1 - p)))

def brier_score(y_true, p):
    y_true = y_true.astype(np.float32)
    return float(np.mean((p.astype(np.float32) - y_true) ** 2))

def precision_recall_f1(y_true, y_pred):
    tp = int(((y_true == 1) & (y_pred == 1)).sum())
    fp = int(((y_true == 0) & (y_pred == 1)).sum())
    fn = int(((y_true == 1) & (y_pred == 0)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return float(precision), float(recall), float(f1)

def switch_rate(seq):
    # taxa de mudanças de comando no tempo (estabilidade)
    if len(seq) <= 1:
        return 0.0
    changes = np.sum(seq[1:] != seq[:-1])
    return float(changes / (len(seq) - 1))

def build_trajectory(seq, step=STEP_METERS, turn_deg=TURN_DEG):
    # integra esquerda/direita em uma trajetória 2D (mesmo modelo do seu trabalho)
    xs, ys = [0.0], [0.0]
    theta = 0.0
    for cmd in seq:
        if cmd == 1:   # direita
            theta -= np.deg2rad(turn_deg)
        else:          # esquerda
            theta += np.deg2rad(turn_deg)
        xs.append(xs[-1] + step * np.cos(theta))
        ys.append(ys[-1] + step * np.sin(theta))
    return np.array(xs), np.array(ys)

def trajectory_errors(y_true_seq, y_pred_seq):
    # Erro entre a trajetória gerada pelos comandos "reais" e "previstos"
    xT, yT = build_trajectory(y_true_seq)
    xP, yP = build_trajectory(y_pred_seq)

    # alinhar comprimentos (devem ser iguais: len+1)
    assert len(xT) == len(xP)

    # RMSE ponto a ponto
    rmse = float(np.sqrt(np.mean((xT - xP) ** 2 + (yT - yP) ** 2)))

    # erro final (distância entre posições finais)
    end_err = float(np.sqrt((xT[-1] - xP[-1]) ** 2 + (yT[-1] - yP[-1]) ** 2))

    return rmse, end_err, (xT, yT, xP, yP)

# =========================
# MLP (Rede Neural Artificial) em NumPy
# - 1 camada escondida
# - sigmoid na saída (binária)
# =========================
class MLPBinary:
    def __init__(self, in_dim, hidden_dim=32, lr=0.01, seed=42):
        rng = np.random.default_rng(seed)
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.lr = lr

        # Xavier/He init (ReLU)
        self.W1 = (rng.standard_normal((in_dim, hidden_dim)).astype(np.float32) * np.sqrt(2.0 / in_dim))
        self.b1 = np.zeros((hidden_dim,), dtype=np.float32)

        self.W2 = (rng.standard_normal((hidden_dim, 1)).astype(np.float32) * np.sqrt(2.0 / hidden_dim))
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

                z1, a1, z2, p = self.forward(Xb)

                # gradiente BCE com sigmoid: dL/dz2 = p - y
                dz2 = (p - yb) / len(yb)

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

            # loss no fim da época (em TODO o treino, para comparar rodadas)
            _, _, _, p_all = self.forward(X)
            L = binary_logloss(y.astype(np.float32), p_all.flatten())
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
# FIGURA: ILUSTRAÇÃO DAS ENTRADAS (Depth + IMU) de uma amostra
# =========================
def plot_input_example(samples, which="middle"):
    if not samples:
        return
    if which == "first":
        sdir = samples[0]
    elif which == "last":
        sdir = samples[-1]
    else:
        sdir = samples[len(samples)//2]

    frames, t, imu, label = load_one_sample(sdir)

    # 1) grid de frames depth
    fig = plt.figure(figsize=(10, 6))
    fig.suptitle(f"Exemplo de entrada (amostra {sdir.name}) | label={'direita' if label==1 else 'esquerda'}")

    # frames (2x4)
    for i in range(WINDOW_FRAMES):
        ax = fig.add_subplot(2, 4, i+1)
        ax.imshow(frames[i], cmap="gray")
        ax.set_title(f"Depth {i+1}")
        ax.axis("off")
    plt.tight_layout()
    plt.show()

    # 2) IMU raw (ax..gz)
    names = ["ax", "ay", "az", "gx", "gy", "gz"]
    plt.figure(figsize=(10, 4))
    for k in range(6):
        plt.plot(t, imu[:, k], label=names[k])
    plt.title("Sinais IMU na janela (ax..gz)")
    plt.xlabel("t")
    plt.ylabel("valor")
    plt.grid(True)
    plt.legend()
    plt.show()

# =========================
# UMA RODADA DE TREINO/TESTE
# =========================
def run_one_experiment(X, y, seed=42, epochs=30, hidden=32, lr=0.02):
    # Split temporal: treina com início, testa com final
    N = len(y)
    n_train = int(N * TRAIN_SPLIT)
    X_train, y_train = X[:n_train], y[:n_train]
    X_test, y_test = X[n_train:], y[n_train:]

    # Normalização baseada no treino
    mu, sigma = standardize_fit(X_train)
    X_train_n = standardize_apply(X_train, mu, sigma)
    X_test_n = standardize_apply(X_test, mu, sigma)

    mlp = MLPBinary(in_dim=X.shape[1], hidden_dim=hidden, lr=lr, seed=seed)
    loss_hist = mlp.train(X_train_n, y_train, epochs=epochs, batch_size=32, verbose=False)

    # predições no teste
    p_test = mlp.predict_proba(X_test_n)
    pred_test = (p_test >= 0.5).astype(int)

    # métricas "clássicas"
    acc = float((pred_test == y_test).mean())
    ll = binary_logloss(y_test, p_test)
    brier = brier_score(y_test, p_test)
    prec, rec, f1 = precision_recall_f1(y_test, pred_test)

    # matriz de confusão
    cm = np.zeros((2, 2), dtype=int)
    for yt, yp in zip(y_test, pred_test):
        cm[int(yt), int(yp)] += 1

    # métricas mais "robótica"
    sw = switch_rate(pred_test)
    traj_rmse, traj_end, traj_pack = trajectory_errors(y_test, pred_test)

    return {
        "seed": seed,
        "loss_hist": np.array(loss_hist, dtype=np.float32),
        "acc": acc,
        "logloss": ll,
        "brier": brier,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "switch_rate": sw,
        "traj_rmse": traj_rmse,
        "traj_end_err": traj_end,
        "cm": cm,
        "y_test": y_test,
        "pred_test": pred_test,
        "p_test": p_test,
        "traj_pack": traj_pack,
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

    # 0) Ilustração de entrada (para inserir como figura no trabalho)
    plot_input_example(samples, which="middle")

    # 1) Múltiplas rodadas (seeds)
    results = []
    for sd in RUN_SEEDS:
        print(f"\nRodada seed={sd} ...")
        res = run_one_experiment(X, y, seed=sd, epochs=30, hidden=32, lr=0.02)
        results.append(res)
        print(f"  acc={res['acc']:.3f} | f1={res['f1']:.3f} | logloss={res['logloss']:.3f} | "
              f"switch_rate={res['switch_rate']:.3f} | traj_rmse={res['traj_rmse']:.3f} | end_err={res['traj_end_err']:.3f}")

    # =========================
    # 2) Gráfico: loss por época (todas as rodadas + média)
    # =========================
    H = np.stack([r["loss_hist"] for r in results], axis=0)  # (runs, epochs)
    mean_loss = H.mean(axis=0)
    std_loss = H.std(axis=0)

    plt.figure()
    for r in results:
        plt.plot(r["loss_hist"], alpha=0.35)
    plt.plot(mean_loss, linewidth=2, label="Média")
    plt.fill_between(np.arange(len(mean_loss)), mean_loss - std_loss, mean_loss + std_loss, alpha=0.2, label="±1 desvio")
    plt.xlabel("Época")
    plt.ylabel("Loss (logloss/BCE)")
    plt.title("Treinamento - Loss (múltiplas rodadas)")
    plt.grid(True)
    plt.legend()
    plt.show()

    # =========================
    # 3) Tabela-resumo (imprime no console)
    # =========================
    print("\nResumo das métricas (teste):")
    print("seed | acc  | f1   | logloss | brier  | switch | traj_rmse | end_err")
    for r in results:
        print(f"{r['seed']:>4} | {r['acc']:.3f} | {r['f1']:.3f} | {r['logloss']:.3f}  | {r['brier']:.3f} | "
              f"{r['switch_rate']:.3f}  | {r['traj_rmse']:.3f}    | {r['traj_end_err']:.3f}")

    # Escolhe "melhor" por menor logloss (pode trocar o critério)
    best = sorted(results, key=lambda d: d["logloss"])[0]
    y_test = best["y_test"]
    pred_test = best["pred_test"]
    p_test = best["p_test"]
    cm = best["cm"]

    print(f"\nMelhor rodada (menor logloss): seed={best['seed']} | logloss={best['logloss']:.3f} | acc={best['acc']:.3f}")

    # =========================
    # 4) Simulação temporal (REAL vs PREVISTO) + probabilidade
    # =========================
    t_test = np.arange(len(y_test))
    plt.figure()
    plt.step(t_test, y_test, where="post", label="Real (label)")
    plt.step(t_test, pred_test, where="post", label="Previsto (MLP)")
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    plt.xlabel("Amostra (ordem temporal no teste)")
    plt.ylabel("Direção")
    plt.title("Simulação temporal: direção real vs prevista")
    plt.grid(True)
    plt.legend()
    plt.show()

    plt.figure()
    plt.plot(t_test, p_test, label="p(direita)")
    plt.step(t_test, y_test, where="post", alpha=0.5, label="Real (0/1)")
    plt.ylim(-0.05, 1.05)
    plt.xlabel("Amostra (ordem temporal no teste)")
    plt.ylabel("Probabilidade")
    plt.title("Probabilidade ao longo do tempo (interpretação 'controle')")
    plt.grid(True)
    plt.legend()
    plt.show()

    # =========================
    # 5) Trajetória: REAL vs PREVISTA
    # =========================
    xT, yT, xP, yP = best["traj_pack"]
    plt.figure()
    plt.plot(xT, yT, label="Trajetória (real)")
    plt.plot(xP, yP, label="Trajetória (prevista)")
    plt.scatter(xT[0], yT[0], marker="o", label="Início")
    plt.scatter(xT[-1], yT[-1], marker="x", label="Fim real")
    plt.scatter(xP[-1], yP[-1], marker="x", label="Fim previsto")
    plt.axis("equal")
    plt.grid(True)
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.title(f"Trajetórias (seed={best['seed']}) | RMSE={best['traj_rmse']:.3f} | EndErr={best['traj_end_err']:.3f}")
    plt.legend()
    plt.show()

    # =========================
    # 6) Matriz de confusão (mantida, mas não é a única métrica)
    # =========================
    plt.figure()
    plt.imshow(cm)
    plt.title("Matriz de Confusão (Teste)")
    plt.xlabel("Previsto")
    plt.ylabel("Real")
    plt.xticks([0, 1], ["Esquerda", "Direita"])
    plt.yticks([0, 1], ["Esquerda", "Direita"])
    for i in range(2):
        for j in range(2):
            plt.text(j, i, str(cm[i, j]), ha="center", va="center")
    plt.show()

    print("\nFim. Simulação concluída.")

if __name__ == "__main__":
    main()
