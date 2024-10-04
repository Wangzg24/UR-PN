LR=1e-5
#ModelType=proto_yuanwen
ModelType=matpn_tri
#ModelType=proto
N=5
K=5

python train_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --model $ModelType --encoder bert --hidden_size 768 --val_step 2000 --lr $LR \
    --pretrain_ckpt pretrain \
    --batch_size 2 --save_ckpt checkpoint/$ModelType/$N-$K-$LR.pth.tar \
    --cat_entity_rep \
    --backend_model bert
