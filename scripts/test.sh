nohup python multi_test.py \
    --time_flag "20251022_195506" \
    --method_type "RoME" \
    --test_problem_type "CA" \
    --training_problem_types "IS_IP_SC" \
    --test_num 100 \
    --gnn_type "moe" \
    --solver "gurobi" \
    --device "cuda:0" \
    --emb_size 64 \
    --num_shared_experts 1 \
    --num_dedicate_experts 5 \
    --top_k 2 \
    --max_time 1000 \
    --num_workers 16 \
    > ./logs/test.log &