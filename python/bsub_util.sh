if [ "$1" != "" ]; then
    bsub -q gpum2050 -gpu "num=2" -J "camvid_util" -W 12:00 -R "rusage[mem=16GB]" -o log_train.txt -e log_err_train.txt $1
else
    echo "Give the script in the argument that you want to run in cluster"
fi