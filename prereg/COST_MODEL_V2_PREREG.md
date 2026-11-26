# PREREG — mô hình chi phí joule bản 2 (H26, S1.4b đợt 2) · 23-11-2026 · Claude Code (Opus 5.5 · medium)

Đóng băng **trước mọi phép đo trên N4 và N5**. SHA-256 và giờ ghi trong `prereg/SHA256SUMS.txt`; commit trong repo `D:\repos\aiot-h26`. Code: `s14b/cost_v2.py`. Tham số đã khớp: `results/s14b_n2/cost_v2_fit.json`.

## 1. Dạng mô hình (FP16, batch 1, 25 W, clock khoá; mJ/suy luận, đã trừ nền)
c_path = e_net·F_path + κ_d·calls + β·B_path + h·checks

- **e_net** = c_full,đo(net) / F_full(net). Đo engine đầy đủ của **chính mạng đó**.
- **F** là GFLOP (cùng công cụ S1.3). **B_path** là tổng MB tensor FP16 đi qua các ranh giới đoạn (k − 1 ranh giới cho đường k).
- **Hệ số tính toán theo cách triển khai** (đây là nơi ghi đủ "theo cách triển khai d"; κ, β, h dùng chung mọi mạng):

| Cách d | Công thức | Ghi chú |
|---|---|---|
| trunc (engine liền tới exit k) | c = e·F_trunc,k + κ_trunc | 1 lời gọi, 0 byte, 0 lần kiểm |
| chain (đoạn + kiểm ở host) | c = e·F_chain,k + θ_chain·k + β·B_k | θ_chain = κ_chain + h. Số lời gọi = số lần kiểm = k nên **không tách được κ_chain và h** |
| cond (engine `If`) | c = e·F_tất-cả-đoạn + κ_cond, **không phụ thuộc đường** | Theo kết luận Bước 3: TensorRT 10.3 chạy mọi nhánh. Vì vậy ρ\*_cond = "không bao giờ có lãi" |
| full | c = c_full,đo | định nghĩa của e_net |

## 2. Tham số đã khớp (trên số FP16 đợt 1 của N1, N2, N3)
- 25 điểm: trunc 9, chain 12, cond 3 (trung bình 4 đường). Bình phương tối thiểu trên sai số tương đối (`scipy.optimize.least_squares`).
- **Giá trị:** κ_trunc = 0,1586 mJ · θ_chain = 0,1879 mJ/đoạn · β = 2,452 mJ/MB · κ_cond = 0,546 mJ.
- **Phần dư trong mẫu (trunc + chain, 21 điểm):** trung vị **4,3 %**, lớn nhất **42 %** (N2 chain_1 +41,9 %, N2 trunc_1 +27,5 %). Cond: −1,4 / +19,9 / +2,1 % (N1/N2/N3).
- Ghi chú: phần dư lớn nhất tập trung ở exit 1 của ResNet-50 (stem 224×224, nhiều truy cập bộ nhớ, ít FLOP). Dạng mô hình **không sửa** sau khi nhìn N4/N5.

## 3. Mạng kiểm (ngoài mẫu, đầu exit ngẫu nhiên, FP16)
- **N4:** MobileNetV2 (torchvision IMAGENET1K_V2), 224×224. Exit sau features[3], features[6], features[13], cuối.
- **N5:** DeiT-S/16 (timm `deit_small_patch16_224.fb_in1k`), 224×224. Exit sau block 3, 6, 9, cuối (LN + Linear trên CLS).

## 4. Quy trình
1. Build engine 4 cách (softmax/so ngưỡng của `If` ghim FP32). Kiểm đúng/sai so với ORT: **lệch logit FP16 ≤ 0,15** (ghi trước).
2. **Phiên 1:** đo **chỉ engine đầy đủ** của N4 và N5, ≥ 10 lô, nền idle đo ngay trước mỗi lô. Ra e_net. Dự báo mọi đường và ρ\* vào `prereg/N4N5_PREDICTION.json` (SHA-256 + giờ), **commit trước khi đo đường nào khác**.
3. **Phiên 2:** mọi cấu hình (full, trunc 1–3, chain 1–4, cond 1–4), ≥ 10 lô mỗi cấu hình, thứ tự xáo (seed 9501 cho N4, 9502 cho N5). Nền idle 5 s **ngay trước mỗi lô**, trừ theo từng lô.
   - ρ\*_đo dùng c_full của phiên 2 (cùng phiên với các đường).
   - Sai số chi phí so với số đo phiên 2.

## 5. Định nghĩa và cổng G-C1′
- **Sai số chi phí:** |c_dự − c_đo|/c_đo, tính trên 7 cấu hình trunc 1–3 + chain 1–4 của mỗi mạng.
- **ρ\*_j** = (c_chain,4 − c_full)/(c_chain,4 − c_chain,j), với j = 1, 2, 3.
- **G-C1′ ĐẠT** khi cả hai điều kiện đúng:
  1. Trung vị sai số chi phí ≤ 10 % **trên N4** và ≤ 10 % **trên N5**;
  2. Với **mọi** exit j ∈ {1, 2, 3} của N4 và N5: |ρ\*_dự − ρ\*_đo| ≤ 0,05 **hoặc** ≤ 20 % tương đối.
- Thêm (không phải cổng): cond có thật sự không phụ thuộc đường trên N4/N5 không (độ chênh giữa các đường so với trung bình).
- **INT8:** ngoài phạm vi cổng.
- **Lệch so với đợt 1:** cách trừ nền đổi từ "trung bình 3 lần idle" sang "idle ngay trước từng lô". Mô hình được khớp trên số đợt 1 (trừ nền kiểu cũ).
