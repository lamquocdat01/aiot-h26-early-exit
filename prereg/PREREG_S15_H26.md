# PREREG — S1.5 đợt xác nhận · 24-11-2026 · Claude Code (Opus 5.5 · medium)

**Đóng băng trước khi huấn luyện và đo.** SHA-256 và giờ ghi trong `prereg/SHA256SUMS.txt`; file cũng được commit trong repo `D:\repos\aiot-h26` (`prereg/`). Mọi sửa đổi về sau đều là **amendment**: ghi giờ, lý do và mục bị thay, đặt ở cuối file này.

- **Cơ sở:** PLAN mục 0h (A7); `PROC_A_PREREG.md` (thủ tục A, giữ nguyên trừ mục 5 dưới đây); `S1/NOVELTY_CHALLENGE_K_H26.md`.
- **Số của đợt này** ghi riêng ở `results/s15/`. Pilot, đợt 1 và đợt 2 là thăm dò, không gộp vào.
- **Seed:** huấn luyện 9201–9205; rút mẫu / chia tập từ 9301 trở đi; xáo thứ tự đo từ 9501.

## 1. Mạng, dữ liệu, huấn luyện
- **Dữ liệu:** CIFAR-100. Tách 45.000 train / 5.000 hold-out (50 ảnh/lớp, split seed 9100, cùng danh sách chỉ số cho mọi mạng và seed); test 10.000.
- **Đầu vào:** ảnh 224 = phóng bilinear từ 32 (align_corners = False); chuẩn hoá ImageNet (N1: mean/std CIFAR).
- **Loss:** Σ_k w_k·CE, w = (0,25; 0,5; 0,75; 1). Huấn luyện đồng thời backbone và đầu exit; AMP; clip grad-norm 5; OneCycle (pct_start 0,1).
- **Code:** `train_multiexit/kaggle_s15/train_s15.py` (SHA-256 bắt đầu `0b9b382c638f3ad0`); runner `h26-s15-a/run.py` (`506a10e2…`), `h26-s15-b/run.py` (`9d9d855b…`).

| Mạng | Kiến trúc · init | Vị trí exit (như đã đo) | Đầu exit | Công thức (opt, lr, lr đầu ×, epoch, batch, wd) | Ngưỡng exit cuối (test) |
|---|---|---|---|---|---|
| N1 | ResNet-56, 32×32, từ đầu | sau layer1[4], layer1, layer2, fc | pool hỗn hợp p = 4 | SGD-nesterov, 0,1, ×1, 160, 128, 5e-4 | ≥ 70 % |
| N2 | ResNet-50, IMAGENET1K_V2 | sau layer1, layer2, layer3, fc | pool hỗn hợp p = 2 | SGD-nesterov, 0,01, **×1**, **10**, 128, 5e-4 | ≥ 75 % |
| N3 | ViT-B/16, IMAGENET1K_V1 | sau block 3, 6, 9, cuối | LN + Linear (CLS) | AdamW, 5e-5, ×20, 5, 64, 0,05 | ≥ 88 % |
| N4 | MobileNetV2, IMAGENET1K_V2 | sau features[3], [6], [13], classifier | pool hỗn hợp p = 2 | SGD-nesterov, 0,02, ×1, 10, 128, 4e-5 | ≥ 70 % |
| N5 | DeiT-S/16, timm `deit_small_patch16_224.fb_in1k` | sau block 3, 6, 9, cuối | LN + Linear (CLS) | AdamW, 1e-4, ×10, 6, 64, 0,05 | ≥ 85 % |

- **Sửa N2** (nguyên nhân nghi ở đợt 2: lr đầu exit ×10 = 0,1 trên đặc trưng 8.192 chiều, loss epoch đầu = 37):
  - lr đồng nhất;
  - clip 5;
  - 10 epoch.
- **Seed:** 5 seed/mạng. Nếu hạn mức Kaggle không đủ thì ≥ 3 seed cho mọi mạng, ghi là lệch prereg.
- **Cổng trong runner:** seed 9201 của một mạng chạy trước. Nếu không đạt ngưỡng thì các seed còn lại **không chạy**. Khi đó được chỉnh siêu tham số tối đa 2 lần, mỗi lần ghi amendment.
- **Mạng không đạt sau 2 lần chỉnh:** dùng số thật, ghi giới hạn, và **mạng đó bị loại khỏi H3** (vẫn báo H1, H2, H4).
- **Lưu từ Kaggle:** trọng số, logits FP32 từng exit (hold-out + test), accuracy, cấu hình, giờ GPU, SHA-256.

## 2. Engine và kiểm đúng/sai
- **Thiết bị:** Jetson Orin Nano, TensorRT 10.3, FP16, batch 1. Engine dựng từ ONNX xuất từ trọng số đã huấn luyện, gồm:
  - full;
  - trunc 1–3;
  - đoạn 1–4 (chuỗi, kiểm ngưỡng ở host);
  - `If` (λ ép đường).
- **Kiểm đúng/sai FP16 (thay ngưỡng lệch logit cũ):** tỉ lệ đổi nhãn argmax so với tham chiếu FP32 **≤ 1 % ở mọi exit**, trên toàn bộ test 10.000.
  - Engine được kiểm: đoạn 1–4 (nguồn logits cho H3/H4). Engine full, trunc và `If` kiểm trên 1.000 ảnh test đầu.
  - Tham chiếu FP32 là logits PyTorch trên Kaggle. Trước đó phải kiểm khớp ORT FP32 của ONNX đã xuất trên 200 ảnh: lệch tuyệt đối ≤ 1e-3.
  - Engine không đạt → **không dùng cho H3/H4** cho tới khi sửa (Bước 4: ghim lớp FP32). Ghi số lớp bị ghim và chi phí trước/sau.
- **INT8:** chỉ phụ và mô tả, ở nơi build được. Không thuộc H nào.

## 3. Đo chi phí
- 25 W (nvpmodel 1), `jetson_clocks` (GPU 918 MHz). AAP dừng. VDD_IN qua INA3221.
- **Nền idle đo ngay trước mỗi lô và trừ theo từng lô** (`--idle_each`; một cách duy nhất cho cả đợt).
- ≥ 10 lô cho mỗi cấu hình, thứ tự xáo (seed 9501 + chỉ số mạng); mỗi lô ≥ 10 s. Đánh dấu lô có nhiệt > 95 °C.
- Đơn vị: mJ/suy luận net; CI 90 % lấy theo bootstrap lô (10.000 lần, seed 9601).
- **Chi phí đo trên seed 9201.** Kiểm trên seed 9202: full và chain 1–4 lệch ≤ 2 % so với 9201. Không đạt → báo và đo cả hai.

## 4. Giả thuyết (PASS/FAIL nguyên văn; không diễn giải lại cổng trượt sát)
- **H1 (K1):** R = c̄_chain4 / c̄_full.
  - **PASS** khi:
    - N3 và N5 đều có CI 90 % của R nằm trọn ≤ 1,10; **và**
    - N1, N2, N4 đều có CI 90 % nằm trọn ≥ 1,20.
  - CI: bootstrap tỉ số trung bình theo lô (lô của hai cấu hình rút độc lập).
- **H2 (K2):** với engine `If` ép đường 1–4, D = (max_k c̄_k − min_k c̄_k) / mean_k c̄_k.
  - **PASS** khi D ≤ 5 % ở **mọi** mạng build được (ước lượng điểm; báo kèm CI 90 %).
  - Mạng không build được thì ghi rõ và không tính.
- **H3 (K3), tại n = 2.000, η = 5 %, giao thức chia tập ở mục 5:**
  - **(a) Transformer có đầu ra.** Với mỗi transformer: "A trả s_LB > 0" ở một mức α nghĩa là ≥ 50 % lần chia có đầu ra với s_LB > 0. Yêu cầu điều này ở ≥ 2/3 mức α, trên **cả N3 và N5**.
  - **(b) CNN từ chối.** Với mỗi CNN: "A trả 'không có'" ở một mức α nghĩa là ≥ 50 % lần chia không có đầu ra. Yêu cầu điều này ở **cả 3** mức α, trên **cả N1, N2 và N4**.
  - **(c) FLOPs gây lỗ.** Trên ≥ 2 trong 3 CNN, bộ chọn theo FLOPs có tiết kiệm thật trên tập đánh giá < 0 ở ≥ 50 % lần chia, tại ≥ 2/3 mức α.
  - **H3 PASS** khi (a), (b) và (c) cùng đúng. Mạng bị loại theo mục 1 thì không tính, và yêu cầu "cả" áp lên các mạng còn lại.
- **H4 (hợp lệ):** trên **mọi** ô (mạng, α ∈ {1, 2, 5} %, n ∈ {500, 2.000, 5.000}), η = 5 %:
  - tỉ lệ vi phạm rủi ro ≤ δ = 0,1; **và**
  - tỉ lệ vi phạm cận tiết kiệm ≤ δ.
  - Mỗi tỉ lệ tính trên 500 lần chia (mục 5) và báo kèm CI Clopper–Pearson 95 %.
  - **PASS** khi mọi ô đạt (theo ước lượng điểm). FAIL thì liệt kê từng ô.
- **Báo theo từng seed và gộp.**
  - H3, H4 tính cho từng seed.
  - Gộp: cộng các lần chia của mọi seed, nên mỗi ô có 2.500 lần chia.
  - Kết luận PASS/FAIL chính dựa trên **bản gộp**. Bản theo seed báo số seed đạt.
  - H1, H2 dựa trên chi phí seed 9201.

## 5. Thủ tục A — thay đổi duy nhất so với `PROC_A_PREREG.md`: giao thức dữ liệu
- **Gộp:** hold-out 5.000 + test 10.000 = 15.000 mẫu (logits FP16 từ Orin của từng seed).
- **Mỗi lần chia** (seed 9301 + i, i = 0…499; cùng seed cho mọi mạng): hoán vị ngẫu nhiên; n mẫu đầu là **tập hiệu chỉnh**, **phần còn lại là tập đánh giá**. Hai tập rời nhau, không rút có hoàn lại.
- **Trên tập hiệu chỉnh:** A chọn θ̂ (rủi ro: CP + fixed-sequence; π: Maurer–Pontil; chi phí: t-UCB + η). Mọi tham số khác giữ như PROC_A_PREREG: δ = 0,1 = 0,05 + 0,03 + 0,02; |Θ| = 143; lưới λ.
- **Trên tập đánh giá:**
  - r_eval(θ̂) = tỉ lệ loss trên tập đánh giá. **Vi phạm rủi ro:** r_eval > α.
  - s_eval(θ̂) = c̄_full − Σ_k π_k^eval·c̄_k, với c̄ là chi phí đo trung bình. **Vi phạm cận tiết kiệm:** s_eval < s_LB.
- **Bộ chọn theo FLOPs:** trên cùng tập θ đã chứng nhận rủi ro, chọn θ có FLOPs kỳ vọng nhỏ nhất trên tập hiệu chỉnh. Báo s_eval của nó.
- **Phân tích nhạy:** η ∈ {0, 10 %}. Chỉ báo, không thuộc H.
- **Giới hạn khai trước:**
  - Vi phạm cận tiết kiệm chỉ kiểm phần π. Phần chi phí dùng cùng các phép đo cho cả cận lẫn "chân lý", nên không kiểm độc lập được.
  - r_eval có nhiễu do tập đánh giá hữu hạn (10.000–14.500 mẫu).

## 6. Kiểm phụ (báo, không thuộc H)
- **Hold-out so với test:** với từng mạng và seed, kiểm định hai mẫu:
  - (i) đúng/sai ở từng exit: z hai tỉ lệ;
  - (ii) max-softmax từng exit: KS.
  - Hiệu chỉnh Holm trên 8 kiểm định cho mỗi (mạng, seed), mức 0,05.
- **Cơ chế (Bước 5):** profiler theo lớp của engine full so với chuỗi cho N1, N2, N4 và N3. Phân rã phần chênh thành: số lớp sau fusion, lớp reformat ở ranh giới, chép vào/ra, thời gian từng đoạn.
