#!/bin/bash

# Ép toàn bộ tiến trình chạy trên GPU 3 (hiện tại là card số 2)
export CUDA_VISIBLE_DEVICES=5

# Lấy luôn cả 2 thư mục nếu sếp muốn chạy Mistral bên data và 3 con bên data_1
JSON_FILES=(
    "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/CORE_NACIR_sub/data/beliefs_mistral_7b.json"
    "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/CORE_NACIR_sub/data_1/beliefs_phi4-mini_latest.json"
    "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/CORE_NACIR_sub/data_1/beliefs_qwen2_5_3b.json"
    "/mlcv1/WorkingSpace/Personal/core_baotg/thuy/CORE_NACIR_sub/data_1/beliefs_qwen2_5_7b.json"
)

PYTHON_SCRIPT="/mlcv1/WorkingSpace/Personal/core_baotg/thuy/CORE_NACIR/examples/plugir/plugir_run.py"

echo "=========================================================="
echo "🚀 Bắt đầu chạy Batch Evaluation trên GPU 2 (Rerank K=50)"
echo "✅ CHẾ ĐỘ MỚI: Truyền toàn bộ file JSON vào Python cùng lúc để giữ Cache RAM!"
echo "=========================================================="

python "$PYTHON_SCRIPT" \
    --beliefs_path "${JSON_FILES[@]}" \
    --output_dir "logs_batch" \
    --rerank_k 50

echo "🎉 TẤT CẢ CÁC MÔ HÌNH ĐÃ CHẠY XONG BẰNG CACHE!"
