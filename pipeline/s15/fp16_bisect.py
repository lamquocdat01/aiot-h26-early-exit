"""S1.5 step 4 (Orin): locate FP16 deviation sources by building the same ONNX with different sets of layers pinned to
FP32 (layer.precision + output type, OBEY_PRECISION_CONSTRAINTS) and comparing to the ORT FP32 reference.
Rules: none | all | te:<TYPE>:<regex> | type:<T1>+<T2> (trt.LayerType names) | re:<regex on layer name> | combos joined by ','.
Reports max/median |dev|, argmax flip rate vs reference, #pinned layers, median GPU latency (ms, 200 runs).
Usage: python fp16_bisect.py <onnx> <ref.npz> <ref_key> <x.npy> <out.json> <rule> [<rule> ...] [--list]
"""
import sys, re, json, time
import numpy as np
import tensorrt as trt
from cuda import cudart
sys.path.insert(0, "/home/user/h26/s14b/code")
from trt_s14b import ck

onnx, refp, key, xp, outp = sys.argv[1:6]; rules = [r for r in sys.argv[6:] if not r.startswith("--")]
LOG = trt.Logger(trt.Logger.ERROR); X = np.load(xp).astype(np.float32); Z = np.load(refp)[key]
res = json.load(open(outp)) if __import__("os").path.exists(outp) else {}


def parse():
    b = trt.Builder(LOG); net = b.create_network(0); p = trt.OnnxParser(net, LOG)
    assert p.parse_from_file(onnx), [str(p.get_error(i)) for i in range(p.num_errors)]
    return b, net, p


if "--list" in sys.argv:
    b, net, p = parse(); from collections import Counter
    print(Counter(str(net.get_layer(i).type) for i in range(net.num_layers)))
    for i in range(min(net.num_layers, 400)):
        L = net.get_layer(i); print(i, L.type, L.name)
    sys.exit()


def match(L, rule):
    if rule == "all":
        return True
    for part in rule.split(","):
        if part.startswith("type:") and str(L.type).split(".")[-1] in part[5:].split("+"):
            return True
        if part.startswith("re:") and re.search(part[3:], L.name):
            return True
        if part.startswith("te:"):          # te:<TYPE>:<regex>  (type AND name)
            _, T, rx = part.split(":", 2)
            if str(L.type).split(".")[-1] == T and re.search(rx, L.name):
                return True
    return False


for rule in rules:
    b, net, p = parse(); c = b.create_builder_config(); c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
    c.set_flag(trt.BuilderFlag.FP16); pinned = 0
    if rule != "none":
        c.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
        for i in range(net.num_layers):
            L = net.get_layer(i)
            if L.type in (trt.LayerType.CONSTANT, trt.LayerType.SHAPE) or not match(L, rule):
                continue
            if not any(L.get_output(j).dtype in (trt.float32, trt.float16) for j in range(L.num_outputs)):
                continue
            L.precision = trt.float32; pinned += 1
            for j in range(L.num_outputs):
                if L.get_output(j).dtype in (trt.float32, trt.float16):
                    L.set_output_type(j, trt.float32)
    t0 = time.time(); s = b.build_serialized_network(net, c)
    if s is None:
        res[rule] = dict(ok=False); print(rule, "BUILD FAIL", flush=True); continue
    e = trt.Runtime(LOG).deserialize_cuda_engine(s); ctx = e.create_execution_context(); st = ck(cudart.cudaStreamCreate())
    names = [e.get_tensor_name(i) for i in range(e.num_io_tensors)]; inp = [n for n in names if e.get_tensor_mode(n) == trt.TensorIOMode.INPUT][0]
    outs = [n for n in names if n != inp]; on = "z" if "z" in outs else outs[-1]
    buf = {}
    for n in names:
        shp = tuple(e.get_tensor_shape(n)); h = np.zeros(shp, np.float32 if e.get_tensor_dtype(n) == trt.float32 else np.float16)
        d = ck(cudart.cudaMalloc(h.nbytes)); ctx.set_tensor_address(n, d); buf[n] = (h, d)
    O = []
    for i in range(len(X)):
        h, d = buf[inp]; h[...] = X[i:i + 1]; ck(cudart.cudaMemcpyAsync(d, h.ctypes.data, h.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, st))
        ctx.execute_async_v3(st); h, d = buf[on]; ck(cudart.cudaMemcpyAsync(h.ctypes.data, d, h.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, st))
        ck(cudart.cudaStreamSynchronize(st)); O.append(h.reshape(-1).astype(np.float32).copy())
    O = np.stack(O); dv = np.abs(O - Z[: len(O)]).max(1)
    lat = []
    for _ in range(200):
        q = time.perf_counter(); ctx.execute_async_v3(st); ck(cudart.cudaStreamSynchronize(st)); lat.append(time.perf_counter() - q)
    res[rule] = dict(ok=True, pinned=pinned, n_layers=net.num_layers, max_dev=float(dv.max()), med_dev=float(np.median(dv)),
                     flip=float((O.argmax(1) != Z[: len(O)].argmax(1)).mean()), lat_ms=float(np.median(lat) * 1e3), build_s=round(time.time() - t0, 1))
    print(rule, res[rule], flush=True); json.dump(res, open(outp, "w"), indent=1)
    for n, (h, d) in buf.items():
        cudart.cudaFree(d)
    del ctx, e
