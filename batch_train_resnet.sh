export OMP_NUM_THREADS=32

for ((i=1; i<=4; i++)); do
  case $i in
    1)
      for auth_id in $(seq 4000 4000 8000); do

        # Execute the train.py script with the current combination of parameters
        CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --synth_id 10000 --auth_id $auth_id --mixstyle true
        ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
      done
      ;;
    2)
      for auth_id in $(seq 4000 4000 8000); do

        # Execute the train.py script with the current combination of parameters
        CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --synth_id 10000 --auth_id $auth_id --randaugment true
        ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
      done
      ;;
    3)
      for auth_id in $(seq 4000 4000 8000); do

        # Execute the train.py script with the current combination of parameters
        CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --synth_id 10000 --auth_id $auth_id --fiqa true
        ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
      done
      ;;
    4)
      for auth_id in $(seq 4000 4000 8000); do

        # Execute the train.py script with the current combination of parameters
        CUDA_VISIBLE_DEVICES=0,1 python3 -m torch.distributed.launch --nproc_per_node=2 --nnodes=1 --node_rank=0 --master_addr="127.0.0.1" --master_port=1236 train.py --synth_id 10000 --auth_id $auth_id --randaugall true
        ps -ef | grep "train" | grep -v grep | awk '{print "kill -9 "$2}' | sh
      done
  esac
done
