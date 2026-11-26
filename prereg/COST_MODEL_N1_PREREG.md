# PREREG — mô hình chi phí joule, S1.4b đêm 1 (H26) · 23-11-2026 · Claude Code (Opus 5.5 · medium)

Đóng băng **trước mọi phép đo của đêm 1**. Sửa sau thời điểm này phải ghi amendment có giờ ở cuối file. SHA-256 của file (bản không có dòng SHA) ghi trong `prereg/SHA256SUMS.txt`, và được commit trong repo `D:\repos\aiot-h26` (thư mục `prereg/`).

## 1. Đối tượng
- **Mạng:**
  - **N1** ResNet-56 / CIFAR-100, đầu exit đã huấn luyện ở pilot;
  - **N2** ResNet-50 @224 (torchvision IMAGENET1K_V2), 3 đầu exit ngẫu nhiên;
  - **N3** ViT-B/16 @224 (torchvision IMAGENET1K_V1), 3 đầu exit ngẫu nhiên.
  - Mỗi mạng có 4 exit; vị trí theo `S1/S14B_DESIGN_H26.md` và `export_nets.py`.
- **Máy:** Orin Nano 8GB, nvpmodel 25W (id 1), `jetson_clocks` (GPU 918 MHz, CPU 1344 MHz), batch 1, TensorRT 10.3.
- **Ảnh vào:** CIFAR-100 test. N2/N3 phóng lên 224 bằng bilinear.
- **Cách triển khai d và đường k:**

| d | Đường k | Tính toán G (GFLOP, cùng công cụ S1.3) | Số lời gọi | Số lần kiểm ngưỡng |
|---|---|---|---|---|
| full (engine đầy đủ, không đầu) | 4 | G_full | 1 | 0 |
| trunc (engine liền tới exit k, gồm đầu k) | 1, 2, 3 | G_trunc,k | 1 | 0 |
| chain (đoạn + kiểm ở host: D2H z_k + softmax mỗi đoạn) | 1–4 | Σ_{j≤k} G_seg,j (gồm đầu của đoạn) | k | k (host) |
| cond (một engine `If`, softmax/so ngưỡng ghim FP32, ép đường bằng λ đầu vào) | 1–4 | như chain | 1 | min(k, 3) (trên thiết bị) |

## 2. Mô hình (dạng đóng băng)
c_{n,P,d,k} = e_P·(1 + φ_d)·G_{n,d,k} + τ_d·calls + h_d·checks [mJ/suy luận, trừ nền idle]

**Ràng buộc**
- φ_full ≡ 0: engine đầy đủ là chuẩn hiệu quả mỗi FLOP.
- τ_full = τ_trunc = τ_cond ≡ τ₁: chi phí một lời gọi engine đơn.
- Với chain, số lời gọi = số lần kiểm = k, nên chỉ khớp tổng θ_chain := τ_chain + h_chain. **Không tách được** (ghi trước).
- h_cond là chi phí mỗi nút `If` thực thi.

**Tham số tự do cho mỗi precision P:** e_P, φ_trunc, φ_chain, φ_cond, τ₁, θ_chain, h_cond (7 tham số).

**Dữ liệu khớp:** N1 + N2 (2 × 12 cấu hình; mỗi cấu hình lấy trung bình của ≥ 10 lô).

**Cách khớp**
- Bình phương tối thiểu trên **sai số tương đối** Σ((c_đo − c_dự)/c_đo)², bằng `scipy.optimize.least_squares`, tham số không ràng buộc dấu.
- CI 90 %: bootstrap theo chỉ số lần lặp (lấy lại 10 lần lặp có hoàn lại, ghép cặp trên mọi cấu hình), 2.000 lần, phân vị 5–95.
- Báo: phần dư từng đường, R² (trên c, không trọng số).

**INT8:** khớp riêng trên N1 + N2 INT8. Nếu trung vị |phần dư tương đối| > 10 % thì ghi **"INT8 không khớp mô hình"**, không ép.

## 3. Dự báo ngoài mẫu (N3)
- **Chính (dùng cho cổng):** từ (e, φ, τ, h) FP16 khớp trên N1 + N2 và G, calls, checks của N3, dự báo c cho 12 cấu hình FP16 của N3 và ρ\* ở mục 4.
- Ghi vào `prereg/N3_PREDICTION.json` (giờ + SHA-256) và **commit trước mọi phép đo trên N3**.
- **Phụ (không dùng cho cổng; khai trước):** "neo" bằng engine đầy đủ đo của N3: e_N3 := c_full,đo(N3)/G_full(N3), giữ φ, τ, h từ N1 + N2.

## 4. Định nghĩa
- **Sai số chi phí:** |c_dự − c_đo|/c_đo cho từng cấu hình FP16 của N3. Báo trung vị và lớn nhất.
- **Điểm hoà vốn:** với d ∈ {chain, cond} và exit j ∈ {1, 2, 3}, trong triển khai đủ 3 exit,
  **ρ\*_{d,j} = (c_{d,4} − c_full)/(c_{d,4} − c_{d,j})**.
  - Đây là tỉ lệ mẫu phải dừng ở j (phần còn lại đi hết) để chi phí kỳ vọng bằng engine đầy đủ.
  - ρ\* ≤ 0 nghĩa là luôn có lãi.
  - Không xác định nếu c_{d,j} ≥ c_{d,4}.
- **Sai số ρ\*:** |ρ\*_dự − ρ\*_đo|/|ρ\*_đo|. Nếu |ρ\*_đo| < 0,05 thì dùng sai số tuyệt đối, và coi đạt khi ≤ 0,05.
- **Cổng G-C1 (N3, FP16):** trung vị sai số chi phí ≤ 10 % (12 cấu hình) **và** trung vị sai số ρ\* ≤ 10 % (6 giá trị).
- **Cổng G-C2:** tồn tại ≥ 1 (mạng ∈ {N1, N2, N3}, P, d ∈ {chain, cond}) có ρ\*_đo < 0,5 ở exit 1 hoặc exit 2.

## 5. Quy trình đo (đóng băng)
- **Thứ tự lô:** mỗi mạng ≥ 10 lần lặp. Trong mỗi lần lặp, mọi cấu hình chạy một lần theo thứ tự ngẫu nhiên, seed 9401 / 9402 / 9403 cho N1 / N2 / N3.
- **Một lô:** warm-up 2 s, rồi chạy ≥ n_min suy luận **và** ≥ 10 s. n_min = 5.000 (N1), 2.000 (N2), 1.000 (N3).
- **Nền idle:** 60 s trước, giữa (sau lần lặp thứ 5) và sau mỗi mạng. Nền dùng để trừ = trung bình ba lần đo idle.
- **Lô bị đánh dấu:** tần số GPU ≠ 918 MHz ở đầu hoặc cuối lô, hoặc tmax ≥ 95 °C. Lô đánh dấu được đo lại một lần ở cuối. Báo kèm số lô bị đánh dấu.
- **Kiểm giả định** (chỉ N1, FP16):
  - full với ảnh thật so với đầu vào ngẫu nhiên N(0, 1);
  - trunc1 với đầu exit đã huấn luyện so với đầu ngẫu nhiên.
  - Báo chênh ± CI 90 % (t ghép cặp theo lần lặp).
- **Kiểm đúng/sai trước khi đo:** 200 ảnh so với ORT. FP16: max|lệch logit| ≤ 0,05. INT8: so với ORT-INT8, chỉ báo, không có ngưỡng. cond: báo số ảnh đổi nhánh ở λ = 0,9.
