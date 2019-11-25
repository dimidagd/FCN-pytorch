if [ "$1" != "" ]; then
    bsub -q gpuv100 -gpu "num=1:mode=exclusive_process" -J "camvid_util" -W 12:00 -R "rusage[mem=8GB]" -o log_train.txt -e log_err_train.txt $1
else
    echo "Give the script in the argument that you want to run in cluster"
fi
