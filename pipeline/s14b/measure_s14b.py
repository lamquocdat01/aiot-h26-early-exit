"""S1.4b night 1 (Orin): joule + latency per (precision, form, path), batch 1, INA3221 sysfs sampler thread.
Per net: idle 60 s before, after rep (reps//2 - 1) and after the last rep; each rep runs every config once in a random
order (seed); each batch = warm-up (2 s) + >= n_min inferences AND >= t_min s; GPU/CPU freq + temps logged per batch;
a batch whose GPU freq differs from the locked value at start or end, or with tj >= 95 C, is flagged and re-measured
once after the last rep. Resumable: state in <out>/measure_raw.json (done batches are skipped on restart).
Usage: python measure_s14b.py <net> <eng_dir> <x.npy> <out_dir> --precs fp16,int8 [--reps 10] [--n_min 5000] [--t_min 10]
       [--seed 9401] [--extra n1]   (extra: N1 assumption tests: full with random input, trunc1 with random head)
"""
import argparse, json, os, threading, time, glob
import numpy as np
from cuda import cudart
from trt_s14b import Engine, chain, ck, Form

ap = argparse.ArgumentParser(); ap.add_argument("net"); ap.add_argument("ed"); ap.add_argument("x"); ap.add_argument("out")
ap.add_argument("--precs", default="fp16"); ap.add_argument("--reps", type=int, default=10); ap.add_argument("--n_min", type=int, default=5000)
ap.add_argument("--t_min", type=float, default=10.0); ap.add_argument("--seed", type=int, default=9401); ap.add_argument("--idle", type=float, default=60)
ap.add_argument("--extra", default=""); ap.add_argument("--only", default=""); ap.add_argument("--idle_each", type=float, default=0.0)
ap.add_argument("--tag_out", default=""); ap.add_argument("--gpu_lock", type=int, default=918000000)
a = ap.parse_args(); os.makedirs(a.out, exist_ok=True); RAWP = f"{a.out}/measure_raw.json"
X = np.load(a.x)
if X.dtype == np.uint8:          # N1: CIFAR uint8 -> normalized float32 (2000 images, as in the pilot)
    m = np.array([0.5070, 0.4865, 0.4409], np.float32).reshape(1, 3, 1, 1); s = np.array([0.2673, 0.2564, 0.2761], np.float32).reshape(1, 3, 1, 1)
    X = ((X[:2000].astype(np.float32) / 255 - m) / s).astype(np.float32)
HW = glob.glob("/sys/bus/i2c/drivers/ina3221/*/hwmon/hwmon*")[0]; CH = {}
for i in (1, 2, 3):
    try:
        CH[open(f"{HW}/in{i}_label").read().strip()] = i
    except OSError:
        pass
FH = {n: (open(f"{HW}/in{i}_input"), open(f"{HW}/curr{i}_input")) for n, i in CH.items() if n in ("VDD_IN", "VDD_CPU_GPU_CV")}
GPUF = "/sys/class/devfreq/17000000.gpu/cur_freq"; CPUF = "/sys/devices/system/cpu/cpu0/cpufreq/scaling_cur_freq"
TZ = []
for _p in sorted(glob.glob("/sys/devices/virtual/thermal/thermal_zone*/temp")):
    try:
        int(open(_p).read()); TZ.append(_p)
    except Exception:
        pass


def rd(f):
    f.seek(0); return int(f.read())


def state():
    t = []
    for p in TZ:
        try:
            t.append(int(open(p).read()) / 1000)
        except Exception:            # some zones are not readable as ints on this board
            pass
    return dict(gpu_hz=int(open(GPUF).read()), cpu_khz=int(open(CPUF).read()), tmax=max(t), fan=int(open(glob.glob("/sys/devices/platform/pwm-fan/hwmon/hwmon*/pwm1")[0]).read()))


class Sampler(threading.Thread):
    def __init__(self):
        super().__init__(daemon=True); self.buf = []; self.on = True

    def run(self):
        while self.on:
            row = [time.perf_counter()]
            for n in ("VDD_IN", "VDD_CPU_GPU_CV"):
                v, c = FH[n]; row.append(rd(v) * rd(c) / 1000.0)
            self.buf.append(row); time.sleep(0.0005)


def energy(buf, t0, t1):
    s = np.array([r for r in buf if t0 <= r[0] <= t1]); dur = s[-1, 0] - s[0, 0]
    return [float(np.trapz(s[:, c], s[:, 0]) / 1000 * (t1 - t0) / dur) for c in (1, 2)], s[:, 1:].mean(0).tolist(), len(s), int((np.diff(s[:, 1]) != 0).sum()) / dur


def idle(sec, smp, tag):
    s0 = state(); t0 = time.perf_counter(); time.sleep(sec); t1 = time.perf_counter()
    e, pm, ns, chg = energy(smp.buf, t0, t1)
    return dict(tag=tag, sec=t1 - t0, mean_mW=pm, n_samples=ns, value_change_hz=chg, state0=s0, state1=state(), wall=time.strftime("%F %T"))


ck(cudart.cudaSetDevice(0)); st = ck(cudart.cudaStreamCreate())
cfgs = {}
for P in a.precs.split(","):
    def E(n):
        p = f"{a.ed}/{n}_{P}.plan"
        return Engine(p) if os.path.exists(p) else None
    full = E("full")
    if full: cfgs[f"{P}_full_p4"] = Form("full", 4, [full], st)
    for k in (1, 2, 3):
        t = E(f"trunc{k}")
        if t: cfgs[f"{P}_trunc_p{k}"] = Form("trunc", k, [t], st)
    segs = [E(f"seg{k}") for k in range(1, 5)]
    if all(segs):
        chain(segs)
        for k in range(1, 5):
            cfgs[f"{P}_chain_p{k}"] = Form("chain", k, segs, st)
    c = E("cond")
    if c:
        for k in range(1, 5):
            cfgs[f"{P}_cond_p{k}"] = Form("cond", k, [Engine(c.path)], st)     # separate context per forced lam
if a.extra == "n1":
    cfgs["fp16_full_p4_randin"] = cfgs["fp16_full_p4"]
    if os.path.exists(f"{a.ed}/trunc1rh_fp16.plan"):
        cfgs["fp16_trunc_p1_randhead"] = Form("trunc", 1, [Engine(f"{a.ed}/trunc1rh_fp16.plan")], st)
XR = np.random.default_rng(1).standard_normal(X[:500].shape).astype(np.float32)
names = sorted(cfgs)
if a.only:
    names = [n for n in names if n in a.only.split(",")]
rec = json.load(open(RAWP)) if os.path.exists(RAWP) else dict(net=a.net, args=vars(a), configs=names, runs=[], idle=[])
done = {(r["cfg"], r["rep"]) for r in rec["runs"] if not r.get("flag")}
smp = Sampler(); smp.start(); time.sleep(1)
if not any(i["tag"] == "before" for i in rec["idle"]):
    rec["idle"].append(idle(a.idle, smp, "before")); json.dump(rec, open(RAWP, "w"), indent=1)


def batch(name, rep, tag=""):
    f = cfgs[name]; Xs = XR if name.endswith("randin") else X
    pre = None
    if a.idle_each > 0:                       # idle measured right before this batch (last idle_each s of idle_each+1 s)
        time.sleep(1.0); ta = time.perf_counter(); time.sleep(a.idle_each); tb = time.perf_counter()
        ei, pmi, nsi, _ = energy(smp.buf, ta, tb); pre = dict(sec=tb - ta, mean_mW=pmi, n_samples=nsi)
    t0 = time.perf_counter(); i = 0
    while time.perf_counter() - t0 < 2.0:
        f.once(Xs[i % len(Xs)][None]); i += 1
    s0 = state(); lat = []; t0 = time.perf_counter(); i = 0
    while i < a.n_min or time.perf_counter() - t0 < a.t_min:
        q = time.perf_counter(); f.once(Xs[i % len(Xs)][None]); lat.append(time.perf_counter() - q); i += 1
    t1 = time.perf_counter(); s1 = state(); e, pm, ns, chg = energy(smp.buf, t0, t1)
    flag = bool(s0["gpu_hz"] != a.gpu_lock or s1["gpu_hz"] != a.gpu_lock or max(s0["tmax"], s1["tmax"]) >= 95)
    r = dict(cfg=name, rep=rep, n=i, sec=t1 - t0, E_J=e, mean_mW=pm, n_samples=ns, value_change_hz=chg,
             lat_p50_ms=float(np.percentile(lat, 50) * 1e3), lat_p99_ms=float(np.percentile(lat, 99) * 1e3), state0=s0, state1=s1,
             flag=flag, tag=tag, wall=time.strftime("%F %T"), idle_pre=pre)
    rec["runs"].append(r); smp.buf = [b for b in smp.buf if b[0] > t1 - 1]; json.dump(rec, open(RAWP, "w"), indent=1)
    print(rep, name, i, f"{t1-t0:.1f}s", [round(v) for v in pm], s1["gpu_hz"], s1["tmax"], "FLAG" if flag else "", flush=True)


rng = np.random.default_rng(a.seed)
for rep in range(a.reps):
    order = rng.permutation(len(names))
    for ci in order:
        if (names[ci], rep) not in done:
            batch(names[ci], rep)
    if rep == a.reps // 2 - 1 and not any(i["tag"] == "middle" for i in rec["idle"]):
        rec["idle"].append(idle(a.idle, smp, "middle")); json.dump(rec, open(RAWP, "w"), indent=1)
for r in [r for r in rec["runs"] if r["flag"] and not r["tag"]]:
    if not any(q["cfg"] == r["cfg"] and q["rep"] == r["rep"] and q["tag"] == "remeasure" for q in rec["runs"]):
        batch(r["cfg"], r["rep"], "remeasure")
if not any(i["tag"] == "after" for i in rec["idle"]):
    rec["idle"].append(idle(a.idle, smp, "after"))
rec["finished"] = time.strftime("%F %T"); json.dump(rec, open(RAWP, "w"), indent=1); smp.on = False; print("MEASURE_DONE", a.net)
