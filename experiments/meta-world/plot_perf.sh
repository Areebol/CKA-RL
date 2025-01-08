TAG=$1
FUSE_TYPE=$2
python extract_results.py --tag $TAG
python process_results.py --tag $TAG --fuse-type $FUSE_TYPE