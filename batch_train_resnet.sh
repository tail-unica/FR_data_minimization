export OMP_NUM_THREADS=8

#for synth_ds in "GC" "DC" "IDF"; do
#  for synth_id in $(seq 2000 2000 8000); do
#      for cmt in "descending" "ascending"; do
#      # Execute the train.py script with the current combination of parameters
#      CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --synth_ds $synth_ds --synth_id $synth_id --auth_id 0 --experiment "confidence" --cmt $cmt
#      ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
#      done
#  done
#done

#for auth_ds in "M2-S" "WF"; do
#    for auth_id in $(seq 2000 2000 8000); do
#        for cmt in "descending" "ascending"; do
#          # Execute the train.py script with the current combination of parameters
#          CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --auth_ds $auth_ds --synth_id 0 --auth_id $auth_id --experiment "confidence" --cmt $cmt
#          ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
#        done
#    done
#done


#for synth_ds in "GC" "DC" "IDF"; do
#  for synth_id in $(seq 2000 2000 8000); do
#      for cmt in "descending" "ascending"; do
#      # Execute the train.py script with the current combination of parameters
#      CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --synth_ds $synth_ds --synth_id $synth_id --auth_id 0 --experiment "gradient_magnitude" --cmt $cmt
#      ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
#      done
#  done
#done

#for auth_ds in "M2-S" "WF"; do
#    for auth_id in $(seq 2000 2000 8000); do
#        for cmt in "descending" "ascending"; do
#          # Execute the train.py script with the current combination of parameters
#          CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --auth_ds $auth_ds --synth_id 0 --auth_id $auth_id --experiment "gradient_magnitude" --cmt $cmt
#          ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
#        done
#    done
#done



for auth_ds in "M2-S" "WF"; do
  for auth_id in $(seq 2000 2000 8000); do
    for ((i=1; i<=5; i++)); do
      # Execute the train.py script with the current combination of parameters
      CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --synth_id 0 --auth_id $auth_id --experiment "baseline" --iter $i
      ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
    done
  done
done

for auth_ds in "WF"; do
    for auth_id in $(seq 8000 2000 8000); do
        for cmt in "descending" "ascending"; do
          # Execute the train.py script with the current combination of parameters
          CUDA_VISIBLE_DEVICES=0 python3 -m torch.distributed.launch --nproc_per_node=1 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --auth_ds $auth_ds --synth_id 0 --auth_id $auth_id --experiment "gradient_magnitude" --cmt $cmt
          ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
        done
    done
done