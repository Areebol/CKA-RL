TAG=main
FUSE_TYPE=fusenet_merge
python extract_results.py --tag $TAG
python process_results.py --tag $TAG --fuse-type $FUSE_TYPE