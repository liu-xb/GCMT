# [Graph Consistency based Mean-Teaching for Unsupervised Domain Adaptive Person Re-Identification](https://www.ijcai.org/proceedings/2021/121)
<img src='ijcai21.jpg' width = '100%'>   

Codes of our IJCAI 2021 paper "Graph Consistency based Mean-Teaching for Unsupervised Domain Adaptive Person Re-Identification. Xiaobin Liu, Shiliang Zhang. IJCAI 2021". If you find this paper useful, please kindly cite our paper as follows:

    @inproceedings{ijcai2021-121,
    title     = {Graph Consistency Based Mean-Teaching for Unsupervised Domain Adaptive Person Re-Identification},
    author    = {Liu, Xiaobin and Zhang, Shiliang},
    booktitle = {Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, {IJCAI-21}},
    publisher = {International Joint Conferences on Artificial Intelligence Organization},
    editor    = {Zhi-Hua Zhou},
    pages     = {874--880},
    year      = {2021},
    month     = {8},
    note      = {Main Track}
    doi       = {10.24963/ijcai.2021/121},
    url       = {https://doi.org/10.24963/ijcai.2021/121},
    }

## Training & Performance
Run `sh train.sh` will train the unsupervised model on Market-1501 dataset. The unsupervised model along with training log can be found in folder `log/`. The trained GCMT model achieves 74.0% and 90.4% in mAP and Rank1 accuracy, respectively. 

We find that applying the temperature parameter when computing weights in the teacher graph in Eqn.(5) slightly improves the performance. The resulting model is called GCMTv2 and can be obtained by running `sh train_v2.sh`. The model trained by soft triplet loss can be obtained by running `sh soft_triplet_train.sh`. All the aforementioned models and logs can be found in folder `log/`. The comparison among GCMT, GCMTv2, soft triplet loss, and baseline is shown in the following figure. 
<img src='accuracy.jpg' width = '100%' >
It can be inferred from the figure that more epochs may improve the performance. We will try more epochs in the future. (As DukeMTMC-reID dataset is no longer available, we do not release models related with this dataset.)

## About more epochs
Training the model for 400 epochs, the mAP accuracy on Market-1501 dataset reachs ~78%!
