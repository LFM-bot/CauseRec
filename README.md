# CauseRec
Pytorch implementation for paper: CauseRec: Counterfactual User Sequence Synthesis for Sequential Recommendation (SIGIR21)

We implement CauseRec (CauseItem) in Pytorch. We generate positive and negative samples by for loop, if there is enough memory, we can change the function counterfactual_neg_sample_for_loop/ counterfactual_pos_sample_for_loop to counterfactual_neg_sample/counterfactual_pos_sample for parallel computation, which can speed up the training. 


## Datasets
The book dataset is available (obtained from https://github.com/THUDM/ComiRec).

## Quick Start
You can run the model with the following code:
```
python runCauseRec.py --dataset book --max_len 20
```
