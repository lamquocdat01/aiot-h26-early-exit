# aiot-h26-early-exit

Code, result files and preregistration for the article

> **Certifying Measured Energy Savings of Early-Exit Networks on an Embedded GPU, with the Right to Abstain**
> Dat Lam Quoc, Truong Nguyen Xuan, Quang Le Nhut Thanh, Trung Lam Van, Dung Le Trung, Loi Truong Nguyen Thang
> FPT School of Business and Technology (FSB), FPT University, Ho Chi Minh City, Vietnam
> Manuscript submitted to *The Journal of Supercomputing* (2026). Snapshot for that submission: tag `submission-jsc-v1`.

The article presents a certification procedure that, from labelled calibration data and batch-level measurements of net energy on a Jetson Orin Nano, returns an early-exit policy with a finite-sample risk guarantee and a finite-sample lower bound on its expected net energy saving, or abstains and keeps the uncut network. It reports a preregistered study on five networks (ResNet-56, ResNet-50, ViT-B/16, MobileNetV2, DeiT-S) and five training seeds, including the three preregistered hypotheses that failed.

## Reproduce every number, table and figure (CPU only, about one minute)

Requires Python 3.11.

```
python -m venv .venv
.venv/bin/pip install -r requirements.txt        # Windows: .venv\Scripts\pip install -r requirements.txt
.venv/bin/python reproduce/run_all.py            # Windows: .venv\Scripts\python reproduce\run_all.py
```

`run_all.py` regenerates from `results/`:

- `out/numbers.tex` and `out/numbers.json`: every value printed in the article and its supplement is a LaTeX macro `\nv{key}` defined here; `numbers.json` records the source file of each value;
- the tables `out/tab_paths.tex` (Table 4), `out/tab_cap.tex` (Table 9), and `out/tab_h4.tex`, `out/tab_seed.tex`, `out/tab_cost.tex` (supplement);
- Figures 2 and 3 (`out/fig_ratios.pdf`, `out/fig_cert.pdf`);

and compares them with `reference/`. It prints `REPRODUCED` when every value matches.

## Repository layout

| Folder | Content |
|---|---|
| `reproduce/` | `make_numbers.py`, `make_fig.py`, `make_fig2.py`, `run_all.py` |
| `results/` | Result files of the preregistered study: energy and latency per deployment form and exit path (`s15/cost`), Procedure A outputs per network and seed (`s15/procA_b`), sensitivity analysis (`s15/sens`), FP16-versus-FP32 label-flip checks with and without FP32 pinning (`s15/flip`), FP16 bisection (`s15/fp16_fix`), timing decomposition (`s15/mech`), post-hoc capacity simulation (`a8`), and the two exploratory cost models scored out of sample (`s14b_n1`, `s14b_n2`) |
| `pipeline/` | Scripts that produced `results/`: training on a cloud GPU (`train/train_s15.py`), ONNX export, TensorRT engine build, measurement with power-mode and clock guards, Procedure A (`s15/proc_A_s15.py`), pooling, sensitivity, timing (`s15/`, `s14b/`), cost models (`s14b/fit_predict.py`, `s14b/cost_v2.py`), capacity simulation (`a8/`) |
| `prereg/` | Preregistration of the confirmatory study and of Procedure A, the two cost-model preregistrations and their hashed out-of-sample predictions, with `SHA256SUMS.txt` (files are in the authors' working language, Vietnamese; Table 3 of the article states the hypotheses in English) |
| `reference/` | Expected outputs of `run_all.py` |
| `docs/` | `DEVIATIONS_S15_LOG.md`, the time-stamped log of every deviation from the preregistration (Vietnamese; summarised in Table 7 of the article) |

## Environment used for the measurements

NVIDIA Jetson Orin Nano developer kit, JetPack 6.2.2 (L4T R36.5.0), TensorRT 10.3.0, 25 W power mode, GPU clock fixed at 918 MHz, FP16, batch size one; energy from the on-board INA3221 monitor (board input rail), idle power subtracted per batch. Training ran on cloud GPUs with PyTorch.

The `pipeline/` scripts need trained weights, ONNX exports and a Jetson board, which are not included (size). Host names and paths in them are placeholders; `pipeline/s15/ingest.sh` reads the board address from the environment variable `JETSON_HOST`. Raw per-batch power traces and on-device logits (several GB) are not included.

## Data

CIFAR-100 (Krizhevsky, 2009) is public and is not redistributed here. It is available from the University of Toronto (https://www.cs.toronto.edu/~kriz/cifar.html) or through `torchvision.datasets.CIFAR100`.

## Limitations

The reproduction script regenerates every reported number from the result files. Re-measuring the energy numbers requires a Jetson Orin Nano with the configuration above, and results on other boards, power modes, software versions or batch sizes will differ: the article's point is that the cost of an exit depends on the deployed engine.

## License and citation

Code is released under the MIT License (`LICENSE`). Please cite the article; `CITATION.cff` gives the metadata (a DOI will be added on publication).
