work_dir="exp/walker"
if [ ! -d $work_dir ]; then
    mkdir -p $work_dir
fi
python rcrl/train.py \
walker \
--work_dir $work_dir \
--num_train_steps 1000000 \
--log_freq 1000 \
--eval_freq 1000 \
--init_steps 10000 \
--save_model \
--save_buffer \
--save_tb \
1>$work_dir/training.log 2>&1 &
echo $! > $work_dir/pid.txt