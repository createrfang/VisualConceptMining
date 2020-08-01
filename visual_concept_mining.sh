# Configuration
TRAINSETPATH=/share/home/fangzhengqing/Data/CUB_200_2011/train
TESTSETPATH=/share/home/fangzhengqing/Data/CUB_200_2011/test
ROOTDIR=`pwd`



EXP='cub_densenet201_2'
RESOURCEPATH="$ROOTDIR/resource/$EXP"
LOGDIRPATH="$ROOTDIR/log/$EXP"

mkdir -p $RESOURCEPATH $LOGDIRPATH

# Set 0 if you don't need data-augment
NEEDAUGMENT=0
if test  $NEEDAUGMENT == 1
then
python utils/augment.py $TRAINSETPATH --out_num 20000
fi

# Pretrain
NEEDPRETRAIN=1
PRETRAINPATH=$TRAINSETPATH

PRETRAINDIR=$RESOURCEPATH'/pretrain_res'
LOGDIR=$LOGDIRPATH'/pretrain'
if test $NEEDPRETRAIN == 1
then
#echo $PRETRAINPATH $TESTSETPATH $PRETRAINDIR
CUDA_VISIBLE_DEVICES=3 python pretrain.py $PRETRAINPATH $TESTSETPATH $PRETRAINDIR $LOGDIR --epoch 120 --classes 200
fi





