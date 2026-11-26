"""S1.5 (Orin; = s14b/build_s14b.py + env PIN="te:<TYPE>:<regex>[;...]": matching layers pinned to FP32, step-4 fix).
S1.4b night 1 (Orin): build TensorRT engines with the Python builder API (TRT 10.3), batch 1 static.
*_fp32.onnx -> <name>_fp16.plan (FP16 flag); *_int8.onnx -> <name>_int8.plan (INT8 + FP16 flags, explicit QDQ).
cond_*: every layer whose name contains 'thr_' (cast/softmax/reducemax/gather/less/squeeze of the exit test) is pinned
to FP32 (layer.precision + output type, OBEY_PRECISION_CONSTRAINTS). Failures are logged verbatim, not forced.
Usage: ~/h26/venv/bin/python build_s14b.py <onnx_dir> <eng_dir> [names...]
"""
import sys, os, json, time, glob, traceback
import tensorrt as trt, re
PIN = [p.split(':', 2) for p in os.environ.get('PIN', '').split(';') if p]          # TYPE may be '*' (any layer type)
PIN_OK = lambda nm: re.search(os.environ.get('PIN_ENG', ''), nm) is not None        # optional: only engines whose name matches

od, ed = sys.argv[1], sys.argv[2]; only = sys.argv[3:]
os.makedirs(ed, exist_ok=True)
LOG = trt.Logger(trt.Logger.WARNING)
rep_p = f"{ed}/build_report.json"
rep = json.load(open(rep_p)) if os.path.exists(rep_p) else {}
files = sorted(glob.glob(f"{od}/*_fp32.onnx") + glob.glob(f"{od}/*_int8.onnx"))
for f in files:
    base = os.path.basename(f)[:-5]; name, tag = base.rsplit("_", 1)
    out = f"{ed}/{name}_{'fp16' if tag == 'fp32' else 'int8'}.plan"
    key = os.path.basename(out)
    if only and not any(o in key for o in only):
        continue
    if os.path.exists(out) and rep.get(key, {}).get("ok"):
        continue
    t0 = time.time(); r = dict(onnx=os.path.basename(f))
    try:
        b = trt.Builder(LOG); net = b.create_network(0); p = trt.OnnxParser(net, LOG)
        if not p.parse_from_file(f):
            raise RuntimeError("parse: " + " | ".join(str(p.get_error(i)) for i in range(p.num_errors)))
        c = b.create_builder_config(); c.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, 1 << 30)
        c.set_flag(trt.BuilderFlag.FP16)
        if tag == "int8":
            c.set_flag(trt.BuilderFlag.INT8)
        pinned = []
        if name.startswith("cond"):
            c.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for i in range(net.num_layers):
                L = net.get_layer(i)
                if "thr_" in L.name and L.type != trt.LayerType.CONSTANT and any(L.get_output(j).dtype in (trt.float32, trt.float16) for j in range(L.num_outputs)):
                    L.precision = trt.float32
                    for j in range(L.num_outputs):
                        if L.get_output(j).dtype in (trt.float32, trt.float16):
                            L.set_output_type(j, trt.float32)
                    pinned.append(L.name)
        if PIN:
            c.set_flag(trt.BuilderFlag.OBEY_PRECISION_CONSTRAINTS)
            for i in range(net.num_layers):
                L = net.get_layer(i)
                if PIN_OK(name) and any(T in ("*", str(L.type).split(".")[-1]) and re.search(rx, L.name) for _, T, rx in PIN) and                         any(L.get_output(j).dtype in (trt.float32, trt.float16) for j in range(L.num_outputs)):
                    L.precision = trt.float32
                    for j in range(L.num_outputs):
                        if L.get_output(j).dtype in (trt.float32, trt.float16):
                            L.set_output_type(j, trt.float32)
                    pinned.append(L.name)
        r["pin_rule"] = os.environ.get("PIN", ""); r["pin_eng"] = os.environ.get("PIN_ENG", ""); r["pinned_fp32_layers"] = pinned
        s = b.build_serialized_network(net, c)
        if s is None:
            raise RuntimeError("build_serialized_network returned None (see TensorRT log)")
        open(out, "wb").write(bytes(s)); r.update(ok=True, bytes=os.path.getsize(out))
    except Exception as ex:
        r.update(ok=False, error=f"{type(ex).__name__}: {ex}"[:2000])
    r["sec"] = round(time.time() - t0, 1); rep[key] = r; print(key, {k: v for k, v in r.items() if k != "pinned_fp32_layers"}, len(r.get("pinned_fp32_layers", [])), flush=True)
    json.dump(rep, open(rep_p, "w"), indent=1)
