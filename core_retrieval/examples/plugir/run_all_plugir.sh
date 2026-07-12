#!/bin/bash

# Thư mục chứa các file json từ các model LLM
DATA_DIR="/workingspace_aiclub/WorkingSpace/Personal/core_baotg/thuyntn/CORE_NACIR_sub/data"
# Script chính cần chạy (bản plug-and-play mới của PlugIR)
RUN_SCRIPT="/workingspace_aiclub/WorkingSpace/Personal/core_baotg/thuyntn/CORE_NACIR/examples/plugir/plugir_run.py"

cd /workingspace_aiclub/WorkingSpace/Personal/core_baotg/thuyntn/CORE_NACIR

for belief_file in "$DATA_DIR"/*.json; do
    filename=$(basename -- "$belief_file")
    model_name="${filename%.*}"
    
    output_dir="logs/plugir_$model_name"
    
    echo "=========================================================="
    echo "Đang chạy PlugIR (Plug-and-play) với: $model_name"
    echo "File beliefs: $belief_file"
    echo "=========================================================="
    
    CUDA_VISIBLE_DEVICES=3 python "$RUN_SCRIPT" \
        --beliefs_path "$belief_file" \
        --output_dir "$output_dir"
done

echo "Tất cả các mô hình đã chạy xong cho PlugIR!"
