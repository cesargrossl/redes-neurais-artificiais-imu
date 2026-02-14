import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

DATASET = Path("dataset_5min_depth_imu/sample_00001")

WINDOW = 8

def read_pgm(path):
    with open(path, "rb") as f:
        f.readline()
        w,h = map(int,f.readline().split())
        f.readline()
        data = np.frombuffer(f.read(),dtype=np.uint8)
        return data.reshape(h,w)

# ---- Depth ----
depth_dir = DATASET/"depth_frames"

means=[]
stds=[]
sectors=[]

for i in range(1,WINDOW+1):
    img = read_pgm(depth_dir/f"{i:04d}.pgm")/255.0

    means.append(img.mean())
    stds.append(img.std())

    h,w = img.shape
    sectors.append([
        img[:,0:w//3].mean(),
        img[:,w//3:2*w//3].mean(),
        img[:,2*w//3:].mean()
    ])

sectors=np.array(sectors)

# ---- IMU ----
imu = np.genfromtxt(DATASET/"imu_window.csv",delimiter=",",skip_header=1)
t=imu[:,0]
imu=imu[:,1:7]

# ---- features ----
fimu = np.concatenate([imu.mean(0),imu.std(0)])
fdepth = np.array([[m,s] for m,s in zip(means,stds)]).flatten()
features=np.concatenate([fimu,fdepth])

# ---- Plot ----
plt.figure(figsize=(12,8))

plt.subplot(221)
plt.plot(means,label="media")
plt.plot(stds,label="desvio")
plt.title("Depth por frame")
plt.legend()
plt.grid()

plt.subplot(222)
plt.imshow(sectors)
plt.xticks([0,1,2],["Esq","Centro","Dir"])
plt.ylabel("Frame")
plt.title("Depth por setores (heatmap)")

plt.subplot(223)
for k in range(6):
    plt.plot(t,imu[:,k])
plt.title("IMU (ax..gz)")
plt.grid()

plt.subplot(224)
plt.bar(range(len(features)),features)
plt.title("Vetor entrada MLP (28D)")

plt.tight_layout()
plt.savefig("entrada_exemplo.png",dpi=200)
plt.show()

print("Figura salva: entrada_exemplo.png")
