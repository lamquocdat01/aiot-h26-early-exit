"""S1.5 (PC; from s14b/export_trained.py, + n4/n5 + ORT-vs-Kaggle check on 200 test images, prereg sec. 2): build the 4 segment modules from Kaggle-trained weights (train_s14b.py layout) and export
seg1..4 ONNX (FP32, batch 1) + FLOPs report, same exit positions and seg boundaries as wave 1 (export_nets.py).
Usage: python export_trained_s15.py <net n1..n5> <kaggle_run_dir (weights.pt, logits_test_e*.npy)> <out_dir>
"""
import sys, os, json, hashlib
import numpy as np
import torch, torch.nn as nn, torch.nn.functional as F, torchvision as tv
net, rd, od = sys.argv[1:4]; wp = f"{rd}/weights.pt"; os.makedirs(od, exist_ok=True); torch.backends.mha.set_fastpath_enabled(False)
sys.path.insert(0, "D:/repos/aiot-h26/s13"); sys.path.insert(0, "D:/repos/aiot-h26/s14b")
SD = torch.load(wp, map_location="cpu")


class PoolHead(nn.Module):
    def __init__(self, ch, nc, p):
        super().__init__(); self.p = p; self.fc = nn.Linear(2 * ch * p * p, nc)

    def forward(self, x):
        k = x.shape[-1] // self.p
        return self.fc(torch.cat([F.avg_pool2d(x, k).flatten(1), F.max_pool2d(x, k).flatten(1)], 1))


class Seg(nn.Module):
    def __init__(self, body, head=None):
        super().__init__(); self.body = body; self.head = head

    def forward(self, x):
        f = self.body(x)
        return (f, self.head(f)) if self.head is not None else f


class ViTEmbed(nn.Module):
    def __init__(self, m):
        super().__init__(); self.m = m

    def forward(self, x):
        m = self.m; x = m._process_input(x); n = x.shape[0]
        return torch.cat([m.class_token.expand(n, -1, -1), x], 1) + m.encoder.pos_embedding


class ViTHead(nn.Module):
    def __init__(self, nc, ln=None, fc=None):
        super().__init__(); self.ln = ln or nn.LayerNorm(768); self.fc = fc or nn.Linear(768, nc)

    def forward(self, t):
        return self.fc(self.ln(t)[:, 0])


def sub(prefix):
    return {k[len(prefix):]: v for k, v in SD.items() if k.startswith(prefix)}


if net == "n1":
    from model import ResNet56
    bb = ResNet56(100); bb.load_state_dict(sub("bb.")); hs = [PoolHead(c, 100, 4) for c in (16, 16, 32)]
    for i, h in enumerate(hs):
        h.load_state_dict(sub(f"heads.{i}."))
    segs = [Seg(nn.Sequential(bb.conv1, bb.bn1, bb.relu, *list(bb.layer1)[:5]), hs[0]), Seg(nn.Sequential(*list(bb.layer1)[5:]), hs[1]),
            Seg(bb.layer2, hs[2]), nn.Sequential(bb.layer3, bb.avgpool, nn.Flatten(1), bb.fc)]; shp = (3, 32, 32)
elif net == "n2":
    m = tv.models.resnet50(); m.fc = nn.Linear(2048, 100); m.load_state_dict(sub("bb.")); hs = [PoolHead(c, 100, 2) for c in (256, 512, 1024)]
    for i, h in enumerate(hs):
        h.load_state_dict(sub(f"heads.{i}."))
    segs = [Seg(nn.Sequential(m.conv1, m.bn1, m.relu, m.maxpool, m.layer1), hs[0]), Seg(m.layer2, hs[1]), Seg(m.layer3, hs[2]),
            nn.Sequential(m.layer4, m.avgpool, nn.Flatten(1), m.fc)]; shp = (3, 224, 224)
elif net == "n4":
    m = tv.models.mobilenet_v2(); m.classifier[1] = nn.Linear(1280, 100); m.load_state_dict(sub("bb.")); hs = [PoolHead(c, 100, 2) for c in (24, 32, 96)]
    for i, h in enumerate(hs):
        h.load_state_dict(sub(f"heads.{i}."))
    f = m.features
    segs = [Seg(f[0:4], hs[0]), Seg(f[4:7], hs[1]), Seg(f[7:14], hs[2]), nn.Sequential(f[14:], nn.AdaptiveAvgPool2d(1), nn.Flatten(1), m.classifier)]; shp = (3, 224, 224)
elif net == "n5":
    import timm
    m = timm.create_model("deit_small_patch16_224.fb_in1k", pretrained=False); m.head = nn.Linear(384, 100); m.load_state_dict(sub("bb.")); B = list(m.blocks)

    class Embed(nn.Module):
        def __init__(self, m):
            super().__init__(); self.m = m

        def forward(self, x):
            m = self.m; x = m.patch_embed(x)
            return m.pos_drop(torch.cat([m.cls_token.expand(x.shape[0], -1, -1), x], 1) + m.pos_embed)

    class DHead(nn.Module):
        def __init__(self, ln, fc):
            super().__init__(); self.ln, self.fc = ln, fc

        def forward(self, t):
            return self.fc(self.ln(t)[:, 0])
    hs = []
    for i in range(3):
        h = DHead(nn.LayerNorm(384), nn.Linear(384, 100)); h.ln.load_state_dict(sub(f"heads.{i}.ln.")); h.fc.load_state_dict(sub(f"heads.{i}.fc.")); hs.append(h)
    segs = [Seg(nn.Sequential(Embed(m), *B[0:3]), hs[0]), Seg(nn.Sequential(*B[3:6]), hs[1]), Seg(nn.Sequential(*B[6:9]), hs[2]),
            nn.Sequential(*B[9:12], DHead(m.norm, m.head))]; shp = (3, 224, 224)
else:
    m = tv.models.vit_b_16(); m.heads.head = nn.Linear(768, 100); m.load_state_dict(sub("bb.")); hs = [ViTHead(100) for _ in range(3)]
    for i, h in enumerate(hs):
        h.load_state_dict(sub(f"heads.{i}."))
    L = list(m.encoder.layers)
    segs = [Seg(nn.Sequential(ViTEmbed(m), *L[0:3]), hs[0]), Seg(nn.Sequential(*L[3:6]), hs[1]), Seg(nn.Sequential(*L[6:9]), hs[2]),
            nn.Sequential(*L[9:12], ViTHead(100, m.encoder.ln, m.heads.head))]; shp = (3, 224, 224)
from torch.utils.flop_counter import FlopCounterMode


def fl(mod, x):
    with FlopCounterMode(display=False) as fc:
        mod(x)
    return int(fc.get_total_flops())


rep = dict(net=net, weights=os.path.basename(wp), weights_sha256=hashlib.sha256(open(wp, "rb").read()).hexdigest(), in_shape=shp, seg_input_shapes=[])
with torch.no_grad():
    x = torch.zeros(1, *shp); sf, hf = [], []
    for k, s in enumerate(segs):
        s.eval(); rep["seg_input_shapes"].append(list(x.shape)); o = s(x); sf.append(fl(s, x))
        torch.onnx.export(s, (x,), f"{od}/seg{k+1}_fp32.onnx", input_names=["x"], output_names=["f", "z"] if k < 3 else ["z"], opset_version=17, dynamo=False)
        if k < 3:
            hf.append(fl(s.head, o[0])); x = o[0]
    xx = torch.zeros(1, *shp); full = 0; tr = []; cur = xx
    for k, s in enumerate(segs):
        b = s.body if k < 3 else s
        full += fl(b, cur); tr.append(full + (hf[k] if k < 3 else 0)); cur = b(cur)
    rep["flops"] = dict(seg=sf, head=hf, full=full, trunc=tr[:3], tool=f"torch.utils.flop_counter.FlopCounterMode (torch {torch.__version__})")
rep["sha256"] = {f: hashlib.sha256(open(f"{od}/{f}", "rb").read()).hexdigest() for f in sorted(os.listdir(od)) if f.endswith(".onnx")}
# ---- prereg sec. 2: ORT FP32 of the exported segment chain vs Kaggle PyTorch FP32 logits, first 200 test images (tol 1e-3)
import onnxruntime as ort, pickle
d = pickle.load(open("D:/data/aiot-h26/cifar100/cifar-100-python/test", "rb"), encoding="latin1"); XU = d["data"][:200].reshape(-1, 3, 32, 32)
x = torch.tensor(XU).float() / 255
if net == "n1":
    M, S = (0.5070, 0.4865, 0.4409), (0.2673, 0.2564, 0.2761)
else:
    M, S = (0.485, 0.456, 0.406), (0.229, 0.224, 0.225); x = F.interpolate(x, size=224, mode="bilinear", align_corners=False)
x = ((x - torch.tensor(M).view(1, 3, 1, 1)) / torch.tensor(S).view(1, 3, 1, 1)).numpy().astype("float32")
sess = [ort.InferenceSession(f"{od}/seg{k}_fp32.onnx", providers=["CPUExecutionProvider"]) for k in range(1, 5)]; dev = []
for k in range(4):
    out = [sess[k].run(None, {"x": x[i:i + 1]}) for i in range(len(x))]
    z = np.concatenate([o[1] if k < 3 else o[0] for o in out]); zk = np.load(f"{rd}/logits_test_e{k+1}.npy")[:200]; dev.append(float(np.abs(z - zk).max()))
    if k < 3:
        x = np.concatenate([o[0] for o in out])
rep["ort_vs_kaggle_maxabs_e1..4"] = dev; rep["ort_vs_kaggle_ok_1e-3"] = max(dev) <= 1e-3
json.dump(rep, open(f"{od}/export_report.json", "w"), indent=1); print(json.dumps(rep["flops"]), "ORTvsKaggle", dev)
