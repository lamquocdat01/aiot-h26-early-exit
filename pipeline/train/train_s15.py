"""H26 S1.5 confirmatory (Kaggle GPU): joint training of backbone + exit heads on CIFAR-100 (derived from train_s14b.py).
Recipes are FROZEN in prereg/PREREG_S15_H26.md (table RECIPE below). Changes vs S1.4b: n4 MobileNetV2, n5 DeiT-S added;
n2 recipe fixed (uniform lr, no x10 head lr, grad-norm clip 5, 10 epochs); grad clip 5 for every net.
Split fixed BEFORE training: 45,000 train / 5,000 calibration hold-out (stratified, 50 per class, seed); test 10,000 untouched.
Exits = exactly the positions measured in wave 1 (export_nets.py):
  n1 ResNet-56 (CIFAR 32x32, from scratch): after layer1[4], layer1, layer2, fc; heads = SDN mixed pool p=4
  n2 ResNet-50 (224, torchvision IMAGENET1K_V2 init): after layer1, layer2, layer3, fc; heads = mixed pool p=2
  n3 ViT-B/16 (224, torchvision IMAGENET1K_V1 init): after blocks 3, 6, 9, final; exit heads = LayerNorm + Linear (CLS)
Loss = sum_k w_k * CE(z_k, y), w = (0.25, 0.5, 0.75, 1.0)  (fixed before training).
Inputs: CIFAR uint8 -> (train aug: random crop 32 pad 4 + hflip) -> [n2,n3: bilinear upsample to 224, align_corners=False]
        -> normalize (n1: CIFAR mean/std; n2,n3: ImageNet mean/std)   == the Orin pipeline.
  n4 MobileNetV2 (224, torchvision IMAGENET1K_V2 init): after features[3], [6], [13], classifier; heads = mixed pool p=2
  n5 DeiT-S/16 (224, timm deit_small_patch16_224.fb_in1k): after blocks 3, 6, 9, final; heads = LayerNorm + Linear (CLS)
Usage: python train_s15.py --net n3 --seed 9201 --out /kaggle/working/n3_s9201
"""
import argparse, json, os, time, hashlib
import numpy as np, torch, torch.nn as nn, torch.nn.functional as F, torchvision as tv

ap = argparse.ArgumentParser(); ap.add_argument("--net", required=True); ap.add_argument("--seed", type=int, default=9101)
ap.add_argument("--epochs", type=int, default=None); ap.add_argument("--bs", type=int, default=None); ap.add_argument("--lr", type=float, default=None)
ap.add_argument("--out", required=True); ap.add_argument("--root", default="/kaggle/working/data"); ap.add_argument("--workers", type=int, default=2)
ap.add_argument("--smoke", type=int, default=0)
a = ap.parse_args()
#        opt, lr(bb), head lr mult, epochs, bs, wd
RECIPE = dict(n1=("sgd", 0.1, 1, 160, 128, 5e-4), n2=("sgd", 0.01, 1, 10, 128, 5e-4), n3=("adamw", 5e-5, 20, 5, 64, 0.05),
              n4=("sgd", 0.02, 1, 10, 128, 4e-5), n5=("adamw", 1e-4, 10, 6, 64, 0.05))[a.net]
a.epochs = a.epochs or RECIPE[3]; a.bs = a.bs or RECIPE[4]; os.makedirs(a.out, exist_ok=True); dev = "cuda" if torch.cuda.is_available() else "cpu"
torch.manual_seed(a.seed); np.random.seed(a.seed); torch.backends.cudnn.benchmark = True
T0 = time.time()

# ------------------------------------------------------------ data (uint8 tensors in memory)
tr = tv.datasets.CIFAR100(a.root, train=True, download=True); te = tv.datasets.CIFAR100(a.root, train=False, download=True)
Xtr = torch.tensor(tr.data).permute(0, 3, 1, 2).contiguous(); Ytr = torch.tensor(tr.targets); Xte = torch.tensor(te.data).permute(0, 3, 1, 2).contiguous(); Yte = torch.tensor(te.targets)
SPLIT_SEED = 9100                                     # same split for every net and training seed
rng = np.random.default_rng(SPLIT_SEED); y = Ytr.numpy()
hold = np.sort(np.concatenate([rng.permutation(np.where(y == c)[0])[:50] for c in range(100)])); train_idx = np.setdiff1d(np.arange(50000), hold)
np.save(f"{a.out}/holdout_idx.npy", hold); np.save(f"{a.out}/train_idx.npy", train_idx)
if a.smoke:
    train_idx, hold, Xte, Yte = train_idx[:a.smoke], hold[:a.smoke // 2], Xte[:a.smoke // 2], Yte[:a.smoke // 2]
if a.net == "n1":
    MEAN, STD, RES = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761), 32
else:
    MEAN, STD, RES = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225), 224
M = torch.tensor(MEAN, device=dev).view(1, 3, 1, 1); S = torch.tensor(STD, device=dev).view(1, 3, 1, 1)


def prep(xb, train):
    x = xb.to(dev, non_blocking=True).float() / 255
    if train:
        n = x.shape[0]; x = F.pad(x, (4, 4, 4, 4)); i, j = np.random.randint(0, 9, 2)
        x = x[:, :, i:i + 32, j:j + 32]
        flip = torch.rand(n, device=dev) < 0.5; x = torch.where(flip.view(-1, 1, 1, 1), x.flip(3), x)
    if RES != 32:
        x = F.interpolate(x, size=RES, mode="bilinear", align_corners=False)
    return (x - M) / S


# ------------------------------------------------------------ models (module names match export_nets.py / s13/model.py)
class PoolHead(nn.Module):
    def __init__(self, ch, nc, p):
        super().__init__(); self.p = p; self.fc = nn.Linear(2 * ch * p * p, nc)

    def forward(self, x):
        k = x.shape[-1] // self.p
        return self.fc(torch.cat([F.avg_pool2d(x, k).flatten(1), F.max_pool2d(x, k).flatten(1)], 1))


class BasicBlock(nn.Module):
    def __init__(self, inp, out, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(inp, out, 3, stride, 1, bias=False); self.bn1 = nn.BatchNorm2d(out)
        self.conv2 = nn.Conv2d(out, out, 3, 1, 1, bias=False); self.bn2 = nn.BatchNorm2d(out); self.relu = nn.ReLU(inplace=True)
        self.downsample = None if (stride == 1 and inp == out) else nn.Sequential(nn.Conv2d(inp, out, 1, stride, bias=False), nn.BatchNorm2d(out))

    def forward(self, x):
        idt = x if self.downsample is None else self.downsample(x)
        return self.relu(self.bn2(self.conv2(self.relu(self.bn1(self.conv1(x))))) + idt)


class ResNet56(nn.Module):           # == s13/model.py ResNet56 (chenyaofo layout)
    def __init__(self, nc=100):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, 1, 1, bias=False); self.bn1 = nn.BatchNorm2d(16); self.relu = nn.ReLU(inplace=True)
        self.layer1 = nn.Sequential(*[BasicBlock(16, 16) for _ in range(9)])
        self.layer2 = nn.Sequential(BasicBlock(16, 32, 2), *[BasicBlock(32, 32) for _ in range(8)])
        self.layer3 = nn.Sequential(BasicBlock(32, 64, 2), *[BasicBlock(64, 64) for _ in range(8)])
        self.avgpool = nn.AdaptiveAvgPool2d(1); self.fc = nn.Linear(64, nc)


class N1(nn.Module):
    def __init__(self):
        super().__init__(); self.bb = ResNet56(); self.heads = nn.ModuleList([PoolHead(16, 100, 4), PoolHead(16, 100, 4), PoolHead(32, 100, 4)])

    def forward(self, x):
        b = self.bb; x = b.relu(b.bn1(b.conv1(x))); x = b.layer1[:5](x); z1 = self.heads[0](x); x = b.layer1[5:](x); z2 = self.heads[1](x)
        x = b.layer2(x); z3 = self.heads[2](x); x = b.layer3(x); return [z1, z2, z3, b.fc(torch.flatten(b.avgpool(x), 1))]


class N2(nn.Module):
    def __init__(self):
        super().__init__(); self.bb = tv.models.resnet50(weights=tv.models.ResNet50_Weights.IMAGENET1K_V2); self.bb.fc = nn.Linear(2048, 100)
        self.heads = nn.ModuleList([PoolHead(256, 100, 2), PoolHead(512, 100, 2), PoolHead(1024, 100, 2)])

    def forward(self, x):
        m = self.bb; x = m.maxpool(m.relu(m.bn1(m.conv1(x)))); zs = []
        for i, l in enumerate([m.layer1, m.layer2, m.layer3]):
            x = l(x); zs.append(self.heads[i](x))
        x = m.layer4(x); zs.append(m.fc(torch.flatten(m.avgpool(x), 1))); return zs


class ViTHead(nn.Module):
    def __init__(self, nc, ln=None, fc=None):
        super().__init__(); self.ln = ln or nn.LayerNorm(768); self.fc = fc or nn.Linear(768, nc)

    def forward(self, t):
        return self.fc(self.ln(t)[:, 0])


class N3(nn.Module):
    def __init__(self):
        super().__init__(); self.bb = tv.models.vit_b_16(weights=tv.models.ViT_B_16_Weights.IMAGENET1K_V1); self.bb.heads.head = nn.Linear(768, 100)
        self.heads = nn.ModuleList([ViTHead(100) for _ in range(3)])

    def forward(self, x):
        m = self.bb; x = m._process_input(x); n = x.shape[0]
        x = torch.cat([m.class_token.expand(n, -1, -1), x], 1) + m.encoder.pos_embedding; zs = []
        for i, blk in enumerate(m.encoder.layers, 1):
            x = blk(x)
            if i in (3, 6, 9):
                zs.append(self.heads[(3, 6, 9).index(i)](x))
        zs.append(m.heads.head(m.encoder.ln(x)[:, 0])); return zs


class N4(nn.Module):
    def __init__(self):
        super().__init__(); self.bb = tv.models.mobilenet_v2(weights=tv.models.MobileNet_V2_Weights.IMAGENET1K_V2); self.bb.classifier[1] = nn.Linear(1280, 100)
        self.heads = nn.ModuleList([PoolHead(24, 100, 2), PoolHead(32, 100, 2), PoolHead(96, 100, 2)])

    def forward(self, x):
        f = self.bb.features; zs = []
        for i, (lo, hi) in enumerate(((0, 4), (4, 7), (7, 14))):
            x = f[lo:hi](x); zs.append(self.heads[i](x))
        x = f[14:](x); zs.append(self.bb.classifier(torch.flatten(F.adaptive_avg_pool2d(x, 1), 1))); return zs


class N5(nn.Module):
    def __init__(self):
        super().__init__(); import timm
        self.bb = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=True); self.bb.head = nn.Linear(384, 100)
        self.heads = nn.ModuleList([nn.ModuleDict(dict(ln=nn.LayerNorm(384), fc=nn.Linear(384, 100))) for _ in range(3)])

    def forward(self, x):
        m = self.bb; x = m.patch_embed(x); x = m.pos_drop(torch.cat([m.cls_token.expand(x.shape[0], -1, -1), x], 1) + m.pos_embed); zs = []
        for i, blk in enumerate(m.blocks, 1):
            x = blk(x)
            if i in (3, 6, 9):
                h = self.heads[(3, 6, 9).index(i)]; zs.append(h["fc"](h["ln"](x)[:, 0]))
        zs.append(m.head(m.norm(x)[:, 0])); return zs


net = {"n1": N1, "n2": N2, "n3": N3, "n4": N4, "n5": N5}[a.net]().to(dev)
W = torch.tensor([0.25, 0.5, 0.75, 1.0], device=dev)
HEADFIN = {"n1": None, "n2": lambda: net.bb.fc, "n3": lambda: net.bb.heads.head, "n4": lambda: net.bb.classifier[1], "n5": lambda: net.bb.head}[a.net]
head_params = list(net.heads.parameters()) + ([] if HEADFIN is None else list(HEADFIN().parameters()))
hp = {id(p) for p in head_params}; bb_params = [p for p in net.parameters() if id(p) not in hp]
kind, lr0, hmult, _, _, wd = RECIPE; lr = a.lr or lr0
groups = [{"params": bb_params, "lr": lr}, {"params": head_params, "lr": hmult * lr}]
opt = torch.optim.SGD(groups, lr=lr, momentum=0.9, weight_decay=wd, nesterov=True) if kind == "sgd" else torch.optim.AdamW(groups, lr=lr, weight_decay=wd)
steps = a.epochs * (len(train_idx) // a.bs); sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=[g["lr"] for g in opt.param_groups], total_steps=steps, pct_start=0.1)
scaler = torch.cuda.amp.GradScaler(enabled=(dev == "cuda")); log = []
for ep in range(a.epochs):
    net.train(); perm = torch.tensor(np.random.permutation(train_idx)); tot = 0.0; nb = 0; t0 = time.time()
    for i in range(0, len(perm) - a.bs + 1, a.bs):
        b = perm[i:i + a.bs]; x = prep(Xtr[b], True); yb = Ytr[b].to(dev)
        with torch.autocast("cuda", dtype=torch.float16, enabled=(dev == "cuda")):
            zs = net(x); loss = sum(W[k] * F.cross_entropy(z.float(), yb) for k, z in enumerate(zs))
        opt.zero_grad(set_to_none=True); scaler.scale(loss).backward(); scaler.unscale_(opt); gn = float(torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)); scaler.step(opt); scaler.update(); sched.step(); tot += float(loss); nb += 1
    log.append(dict(epoch=ep, loss=tot / nb, last_gradnorm=gn, sec=time.time() - t0)); print(log[-1], flush=True)


@torch.no_grad()
def dump(X, Y, tag):
    net.eval(); Z = [[] for _ in range(4)]
    for i in range(0, len(X), 250):
        zs = net(prep(X[i:i + 250], False))           # FP32 inference
        for k, z in enumerate(zs):
            Z[k].append(z.float().cpu().numpy())
    acc = []
    for k in range(4):
        zk = np.concatenate(Z[k]); np.save(f"{a.out}/logits_{tag}_e{k+1}.npy", zk); acc.append(float((zk.argmax(1) == Y.numpy()).mean()))
    np.save(f"{a.out}/y_{tag}.npy", Y.numpy()); return acc


rep = dict(net=a.net, seed=a.seed, recipe=RECIPE, clip=5.0, split_seed=SPLIT_SEED, epochs=a.epochs, bs=a.bs, lr=lr, loss="sum_k w_k CE(z_k,y), w=(0.25,0.5,0.75,1.0)",
           opt=type(opt).__name__, torch=torch.__version__, n_gpu=torch.cuda.device_count(), torchvision=tv.__version__, gpu=torch.cuda.get_device_name(0) if dev == "cuda" else "cpu", log=log,
           acc_holdout=dump(Xtr[hold], Ytr[hold], "holdout"), acc_test=dump(Xte, Yte, "test"))
sd = {k: v.cpu() for k, v in net.state_dict().items()}; torch.save(sd, f"{a.out}/weights.pt")
rep["gpu_sec"] = time.time() - T0
rep["sha256"] = {f: hashlib.sha256(open(f"{a.out}/{f}", "rb").read()).hexdigest() for f in sorted(os.listdir(a.out)) if not f.endswith(".json")}
json.dump(rep, open(f"{a.out}/report.json", "w"), indent=1); print("DONE", a.net, rep["acc_holdout"], rep["acc_test"], f"{rep['gpu_sec']/3600:.2f} h")
