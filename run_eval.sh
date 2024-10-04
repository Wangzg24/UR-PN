LR=1e-5
#ModelType=proto_yuanwen
ModelType=matpn_tri
#ModelType=proto
N=5
K=5
#Dataset=val_fewrel
Dataset=test_fewrel
#Dataset=val_pubmed_new

python test_demo.py \
    --trainN $N --N $N --K $K --Q 1 --dot \
    --model $ModelType --encoder bert --hidden_size 768 --val_step 2000 --test $Dataset \
    --batch_size 2 --only_test \
    --load_ckpt checkpoint/$ModelType/$N-$K-$LR.pth.tar \
    --pretrain_ckpt pretrain \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model bert