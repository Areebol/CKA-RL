# TAG=Ablation/woMerge
# TAG=Baseline
# TAG=Ablation/Merge_PoolSize8_FuseHeads
TAG=Extended_task_2
# FUSE_TYPE=fusenet
FUSE_TYPE=fusenet_merge
python extract_results.py --tag $TAG
python process_results.py --tag $TAG --fuse-type $FUSE_TYPE