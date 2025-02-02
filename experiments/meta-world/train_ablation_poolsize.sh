# python run_experiments.py --algorithm fusenet --start-mode 0 --tag FuseHeads --fuse_heads
pool_size="18 14 10"
seed=0

for size in $pool_size;
do 
python run_experiments.py --algorithm fusenet_merge \
                        --start-mode $(($size + 1)) --tag "Merge_PoolSize${size}_FuseHeads_${seed}" \
                        --fuse_heads --pool_size $size --seed $seed
done
# python run_experiments.py --algorithm simple