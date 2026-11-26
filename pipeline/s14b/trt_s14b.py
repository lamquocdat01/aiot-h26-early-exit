"""TensorRT 10 + cuda-python helpers for S1.4b (multi-input engines, device chaining, the 4 deployment forms)."""
import numpy as np
import tensorrt as trt
from cuda import cudart

LOG = trt.Logger(trt.Logger.WARNING)


def ck(r):
    err = r[0] if isinstance(r, tuple) else r
    if int(err) != 0:
        raise RuntimeError(f"CUDA error {err}")
    return r[1] if isinstance(r, tuple) and len(r) == 2 else None


class Engine:
    def __init__(self, path):
        self.path = path; self.rt = trt.Runtime(LOG)
        self.eng = self.rt.deserialize_cuda_engine(open(path, "rb").read())
        if self.eng is None:
            raise RuntimeError(f"cannot deserialize {path}")
        self.ctx = self.eng.create_execution_context()
        self.names = [self.eng.get_tensor_name(i) for i in range(self.eng.num_io_tensors)]
        self.inp = [n for n in self.names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
        self.out = [n for n in self.names if self.eng.get_tensor_mode(n) == trt.TensorIOMode.OUTPUT]
        self.host, self.dev = {}, {}
        for n in self.names:
            shp = tuple(self.ctx.get_tensor_shape(n))
            if any(s < 0 for s in shp):          # data-dependent output shape (If) -> allocate generously
                shp = (1, 1000)
            h = np.zeros(shp, dtype=trt.nptype(self.eng.get_tensor_dtype(n)))
            self.host[n] = h; self.dev[n] = ck(cudart.cudaMalloc(max(h.nbytes, 4096)))
            self.ctx.set_tensor_address(n, self.dev[n])

    def bind_input(self, name, ptr):
        self.ctx.set_tensor_address(name, ptr)

    def h2d(self, name, arr, st):
        a = np.ascontiguousarray(arr, dtype=self.host[name].dtype).reshape(self.host[name].shape)
        ck(cudart.cudaMemcpyAsync(self.dev[name], a.ctypes.data, a.nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, st))
        return a

    def d2h(self, name, st):
        h = self.host[name]
        ck(cudart.cudaMemcpyAsync(h.ctypes.data, self.dev[name], h.nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, st))
        return h

    def run(self, st):
        if not self.ctx.execute_async_v3(st):
            raise RuntimeError(f"execute failed {self.path}")


def chain(segs):
    for e0, e1 in zip(segs[:-1], segs[1:]):
        e1.bind_input("x", e0.dev["f"])


def softmax_max(z):
    z = z.reshape(-1).astype(np.float64); z = z - z.max(); p = np.exp(z); return float(p.max() / p.sum())


class Form:
    """One measurable configuration. kind in {full, trunc, chain, cond}; path k (1..4)."""
    def __init__(self, kind, k, engines, st):
        self.kind, self.k, self.E, self.st = kind, k, engines, st
        if kind == "cond":
            lam = np.full(3, 2.0, np.float32)
            if k < 4:
                lam[k - 1] = -1.0           # force exit at k (max softmax >= -1 always true); 2.0 never fires
            self.lam = self.E[0].h2d("lam", lam, st); ck(cudart.cudaStreamSynchronize(st))

    def once(self, x):
        st = self.st
        if self.kind in ("full", "trunc", "cond"):
            e = self.E[0]; e.h2d("x", x, st); e.run(st); e.d2h(e.out[0], st); ck(cudart.cudaStreamSynchronize(st))
            return e.host[e.out[0]]
        segs = self.E[: self.k]; segs[0].h2d("x", x, st); z = None
        for e in segs:                                   # host threshold check after every segment (as in the pilot)
            e.run(st); z = e.d2h("z", st); ck(cudart.cudaStreamSynchronize(st)); softmax_max(z)
        return z
