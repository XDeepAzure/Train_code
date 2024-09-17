#

sh_path=/data/hyxu/Pruner/task_train.sh

echo "当前脚本进程id is $$"

bash $sh_path "translate,enc" 2e-5 5 enc5

echo "enc5训练完毕"
