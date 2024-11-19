# **A Universal Background-Supervised and Rectification Loss Enhanced Prototypical Network for Few-Shot Relation Classification** **(UR-PN)**

## 1. Files Introduction

1. The "Checkpoint" is used to save the trained model. The model we proposed is saved in the "matpn_tri" folder. To facilitate code reproduction, we provide a trained model: 5-5-1e-5.pth.tar. You can download the model at https://pan.baidu.com/s/1BDwknH1b5Mj8FTXzhsYDyA?pwd=y6nu.
2. The datasets we use is in the "data". It includes the training set "train_fewrel", the validation set "val_fewrel", and three test sets "val_fewrel" (named FewRel1.0-Val in the paper), "test_fewrel"  (named FewRel1.0 in the paper), and "val_pubmed_new" (named FewRel2.0 in the paper). There are two files containing the class names and descriptions. The "pid2name" corresponds to fewrel1.0 dataset. The "newpidname" corresponds to fewrel2.0 dataset.
3. In the "fewshot_re_kit" are related configuration files, including "data_loader", "sentence_encoder", etc.
4. All models are saved in the "models". The model we proposed is marked as "matpn_tri".
5. The "Pretrain" saves configuration files related to embedding and encoding. Download from https://pan.baidu.com/s/1VThkF9ieBEEeuLZWKJW0Xw?pwd=qqi9.

## 2. Train

You can train the model using the following command.

```
sh run_train.sh
```

Alternatively, you can also train the model using the following python command.

```
python train_demo.py \
    --trainN 5 --N 5 --K 5 --Q 1 --dot \
    --model matpn_tri --encoder bert --hidden_size 768 --val_step 2000 --lr 1e-5 \
    --pretrain_ckpt pretrain \
    --batch_size 2 --save_ckpt checkpoint/matpn_tri/5-5-1e-5.pth.tar \
    --cat_entity_rep \
    --backend_model bert
```

However, we recommend the first command. Note that when you want to modify the external information used, you need to make the following changes in file "sentence_encoder".

```
class BERTSentenceEncoder(nn.Module):
...
	def tokenize_rel(self, raw_tokens):
        # token -> index
        tokens = ['[CLS]']
        name, description = raw_tokens
        # [CLS] + label words + [SEP]
        for token in name.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
        tokens.append('[SEP]')

        # [CLS] label words + [SEP] + label descriptions
        for token in description.split(' '):
            token = token.lower()
            tokens += self.tokenizer.tokenize(token)
            ...
```

## 3. Test

We used three test sets. You can test the model using the following command.

```
sh run_eval.sh
```

Alternatively, you can also test the model using the following python command.

```
python test_demo.py \
    --trainN 5 --N 5 --K 5 --Q 1 --dot \
    --model matpn_tri --encoder bert --hidden_size 768 --val_step 2000 --test test_fewrel \
    --batch_size 2 --only_test \
    --load_ckpt checkpoint/checkpoint/matpn_tri/5-5-1e-5.pth.tar \
    --pretrain_ckpt pretrain \
    --cat_entity_rep \
    --test_iter 1000 \
    --backend_model bert
```

However, we recommend the first command. 

1. If you want to use the first test set to test the model's data fitting ability, you need to set the value of "--test" to "val_fewrel".

2. If you want to use the second test set to test the model's intra-domain generalization ability, you need to set the value of "--test" to "test_fewrel".

3. If you want to use the thrid test set to test the model's cross-domain generalization ability, you need to set the value of "--test" to "val_pubmed_new". Note that when you use dataset "val_pubmed_new" for testing, you need to make the following changes in file "data_loader".

   ```
   # use fewrel1.0
   # pid2name = 'pid2name'
   # pid2name_path = os.path.join(root, pid2name + ".json")
   
   # use fewrel2.0 for test
   pid2name = 'newpidname'
   pid2name_path = os.path.join(root, pid2name + ".json")
   ```

## 4. Results

| Dataset       | 5-way-1-shot | 5-way-5-shot | 10-way-1-shot | 10-way-5-shot |
| :------------ | :----------: | :----------: | :-----------: | :-----------: |
| FewRel1.0-Val |  90.41±0.11  |  97.56±0.09  |  83.56±0.21   |  94.17±0.15   |
| FewRel1.0     |  85.59±0.10  |  94.40±0.09  |  76.05±0.21   |  87.77±0.07   |
| FewRel2.0     |  75.43±0.13  |  90.31±0.07  |  64.96±0.23   |  78.79±0.13   |



## 5. Cite

If you would like to cite this article or use data published in this article, please use the citation format uploaded later. f you have any questions, please contact email Wangzg24@mails.jlu.edu.cn.
