#

sh_path=/data/hyxu/Pruner/task_train.sh

echo "当前脚本进程id is $$"

bash $sh_path "translate,enc" 2e-5 5 enc5

echo "enc5训练完毕"

bash $sh_path "translate,enc" 2e-5 1 enc1

echo "enc1 训练完成"

bash $sh_path "translate,enc" 4e-6 5 lr-enc5

echo "lr 4e-6 enc5 训练完成"

bash $sh_path "translate,enc" 4e-6 1 lr-enc1

echo "lr 4e-6 enc1 训练完成"

bash $sh_path "translate,enc,dec" 2e-5 1 lr-enc1-dec

echo "lr 2e-5 enc1 dec 训练完成"
bash $sh_path "translate,enc,dec" 2e-5 5 lr-enc5-dec

echo "lr 2e-5 enc1 dec 训练完成"