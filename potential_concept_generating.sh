#potential concept generating
PRETRAINMODEL=resource/keratitis_densenet201/pretrain_res/checkpoints/densenet121_33.pth
UNETMODEL=resource/Unet_cpu.model
PCPATH=/share/home/fangzhengqing/Data/Keratitisbaseline_pcg_no_unet2/train


python -m potential_concept_generate --outputdir $PCPATH