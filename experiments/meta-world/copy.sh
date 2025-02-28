#!/bin/bash

# 源目录，即要搜索的目录
source_dir="./runs/Baseline"
# 目标目录，即要将文件夹复制到的目录
destination_dir="./runs/FinalResults"

# 确保目标目录存在，如果不存在则创建
mkdir -p "$destination_dir"

# 查找所有末尾带有42的文件夹并复制到目标目录
find "$source_dir" -type d -name '*42' -exec cp -r {} "$destination_dir" \;

echo "复制完成"