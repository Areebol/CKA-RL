TAG=$1
python extract_results.py --tag $TAG
python process_results.py --tag $TAG