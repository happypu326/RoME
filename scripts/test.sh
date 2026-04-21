nohup python test.py \
    --time_flag "20260123_105309" \
    --method_type "RoME" \
    --test_problem_type "JS" \
    --difficulty "hard" \
    --training_problem_types "BP_CA_CFLP_GISP_IP_LB_MIS_MVC_SC" \
    --test_num 100 \
    --gnn_type "moe" \
    --solver "gurobi" \
    --device "cpu" \
    --emb_size 64 \
    --num_shared_experts 2 \
    --num_dedicate_experts 16 \
    --top_k 8 \
    --max_time 1000 \
    --num_workers 20 \
    --instance_dir "/data/RoME/dataset/test" \
    > ./logs/test.log &

