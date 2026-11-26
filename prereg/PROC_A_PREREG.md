# PREREG — thủ tục A (chứng chỉ kép rủi ro + tiết kiệm joule đo), S1.4b đợt 2 · 23-11-2026 · Claude Code (Opus 5.5 · medium)

Đóng băng **trước khi chạy A trên dữ liệu thật**. SHA-256 và giờ ghi trong `prereg/SHA256SUMS.txt`; commit trong repo. Code: `s14b/proc_A.py`. Định lý: `S1/THEORY_A_H26.md` A1. Phần rủi ro trích Jang 2026 Thm 2 / Prop 1, không claim.

## 1. Dữ liệu
- **Mạng:** N3 (ưu tiên), N1, N2, huấn luyện đồng thời trên Kaggle (seed 9101).
- **Logits:** lấy **từ engine FP16 trên Orin**, trên hold-out 5.000 ảnh (tách từ train trước khi huấn luyện, split seed 9100) và test 10.000.
- **Hiệu chuẩn:** rút từ hold-out. **Chân lý:** đánh giá trên test (rủi ro thật, phân phối exit thật).
- **Chi phí:** đo trên Orin với trọng số đã huấn luyện (Bước 6.2): 25 W, clock khoá, batch 1, ≥ 10 lô cho mỗi cấu hình, nền idle trừ theo từng lô. Cấu hình gồm:
  - full;
  - chain đường 1–4 (engine đoạn, kiểm ở host tại cả 3 exit);
  - trunc 1–3.

## 2. Loss và rủi ro
ℓ_θ(x, y) = 1{ŷ_E(x) ≠ y **và** ŷ_4(x) = y}. Đây là performance gap so với exit cuối; ŷ_k là argmax logits FP16 Orin của exit k. r(θ) = E ℓ_θ.

## 3. Họ chính sách Θ (khai trước)
- **chain(E, λ):** dùng engine đoạn của cả 3 exit.
  - Tập cổng bật E ⊆ {1, 2, 3}, E ≠ ∅ (7 tập). Mẫu dừng ở exit đầu tiên j ∈ E có max softmax ≥ λ; không có thì chạy tới exit 4.
  - Cổng tắt vẫn trả phí đoạn và phí kiểm, như sentinel 1⁺ của Jang. Vì vậy **chi phí chỉ phụ thuộc đường thực thi k** và bằng c_chain,k đo được.
  - λ ∈ lưới 20 giá trị cách đều từ 0,99 xuống 0,50; fixed-sequence đi từ 0,99 (bảo thủ) xuống.
- **trunc(j):** engine cắt tĩnh, mọi mẫu dừng ở j (3 chính sách, mỗi cái là một chuỗi 1 điểm).
- **"Không exit":** engine full. Đây là đầu ra mặc định khi A trả "không có".
- **|Θ| = 7 × 20 + 3 = 143.** Số chuỗi J = 7 + 3 = 10.

## 4. Thủ tục (Định lý A1)
**Chia δ:** δ = 0,10 = δ_r 0,05 + δ_π 0,03 + δ_c 0,02.
1. **(R) Rủi ro.** Cận trên Clopper–Pearson ở mức δ_r/J cho mỗi chuỗi (Bonferroni giữa các chuỗi), fixed-sequence trong chuỗi.
2. **(C) Chi phí.** Với mỗi đại lượng q ∈ Q = {full, chain_1…4, trunc_1…3} (|Q| = 8):
   - U_q = trung bình + t_{1−δ_c/8, n_c−1}·sd/√n_c + η·c̄_q;
   - L_full = trung bình − t·sd/√n_c − η·c̄_full.
   - **η = 5 %** (sai lệch hệ thống khai trước). Phân tích nhạy: η ∈ {0, 10 %}.
3. **(Π) Phân phối đường.** Cận trên Maurer–Pontil của E[U_{E_θ(X)}] ở mức δ_π/|Θ|, miền [0, max U].
4. **Kết quả.** s_LB(θ) = L_full − Ū_θ. Trả argmax s_LB trên các θ đã chứng nhận rủi ro có s_LB > 0. Không có thì trả **"không có"**.

**Lưới thí nghiệm**
- α ∈ {1 %, 2 %, 5 %}; n ∈ {500, 2.000, 5.000}.
- 200 lần rút **có hoàn lại** từ hold-out, seed 9601 + chỉ số.
- n = 5.000 bằng cỡ hold-out, nên các lần rút là bootstrap của hold-out.

## 5. Báo cáo (mỗi mạng × α × n)
- Tỉ lệ có đầu ra; chính sách hay gặp nhất.
- s_LB trung bình (mJ và % so với c_full).
- **Tiết kiệm thật trên test** = c_full − Σ_k π_k^test(θ̂)·c_k.
- **Tỉ lệ vi phạm rủi ro:** r_test(θ̂) > α. **Tỉ lệ vi phạm tiết kiệm:** s_test(θ̂) < s_LB.
- **So sánh — chọn theo FLOPs (kiểu Jang/Jazbec):** cùng tập R̂, chọn θ có FLOPs kỳ vọng nhỏ nhất (theo π trên mẫu hiệu chuẩn). Báo tiết kiệm joule thật trên test của nó (có thể âm) và tỉ lệ lần nó tốn joule hơn engine full.

## 6. Cổng G-J′ (nguyên văn PLAN A6, định nghĩa thao tác ghi trước)
- "A trả cận dưới tiết kiệm joule > 0" ở một (mạng, α) nghĩa là: **≥ 50 %** trong 200 lần rút ở n = 2.000 có đầu ra với s_LB > 0.
- "A trả 'không có'" ở một mạng nghĩa là: **≥ 50 %** lần rút ở n = 2.000 trả "không có", ở **cả 3** mức α.
- **G-J′ ĐẠT** khi:
  1. Có ≥ 1 mạng mà A trả s_LB > 0 ở ≥ 2/3 mức α; **và**
  2. Có ≥ 1 mạng khác mà A trả "không có".
