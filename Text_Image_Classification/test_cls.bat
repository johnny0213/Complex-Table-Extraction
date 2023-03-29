python -u run_cls.py ^
    --model_name_or_path ./ernie-layoutx-base-uncased/models/rvl_cdip_sampled/checkpoint-16000 ^
    --output_dir ./ernie-layoutx-base-uncased/models/rvl_cdip_sampled ^
    --dataset_name rvl_cdip_sampled ^
    --do_predict ^
    --num_train_epochs 20 ^
    --lr_scheduler_type linear ^
    --max_seq_length 512 ^
    --warmup_ratio 0.05 ^
    --weight_decay 0 ^
    --eval_steps 500 ^
    --save_steps 500 ^
    --save_total_limit 1 ^
    --pattern "cls" ^
    --use_segment_box ^
    --return_entity_level_metrics false ^
    --overwrite_cache false ^
    --doc_stride 128 ^
    --target_size 1000 ^
    --per_device_train_batch_size 4 ^
    --per_device_eval_batch_size 8 ^
    --learning_rate 1e-5 ^
    --preprocessing_num_workers 1 ^
    --train_nshard 16 ^
    --seed 1000 ^
    --metric_for_best_model acc ^
    --greater_is_better true ^
    --overwrite_output_dir
