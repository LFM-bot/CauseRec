import argparse
from src.train.trainer import load_trainer

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Model
    parser.add_argument('--model', default='CauseRec', type=str)
    parser.add_argument('--model_type', default='Sequential', choices=['Sequential', 'Knowledge'])
    parser.add_argument('--embed_size', default=128, type=int)
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--interest_num', default=20, type=int)
    parser.add_argument('--replace_ratio', default=0.5, type=float)
    parser.add_argument('--neg_size', default=4, type=int, help='counterfactual neg sample size')
    parser.add_argument('--pos_size', default=1, type=int, help='counterfactual pos sample size')
    parser.add_argument('--bank_size', default=1024, type=int, help='volume of memory bank')
    parser.add_argument('--lamda_1', default=1., type=float,
                        help='weight of FO (Counterfactual and Observation) contrastive loss')
    parser.add_argument('--lamda_2', default=1., type=float, help='weight of II (Interest and Items) contrastive loss')
    parser.add_argument('--sample_size', default=-1, type=int, help='sampled neg item for cross-entropy loss')
    parser.add_argument('--hidden_dropout', default=0.5, type=float, help='hidden state dropout rate')
    parser.add_argument('--initializer_range', default=0.02, type=float, help='transformer params initialize range')
    parser.add_argument('--loss_type', default='CUSTOM', type=str, choices=['CE', 'BPR', 'BCE', 'CUSTOM'])
    # Data
    parser.add_argument('--dataset', default='book', type=str)
    parser.add_argument('--data_aug', action='store_false', help='data augmentation')
    parser.add_argument('--max_len', default=20, type=int, help='max sequence length')
    # Training
    parser.add_argument('--epoch_num', default=150, type=int)
    parser.add_argument('--train_batch', default=1024, type=int)
    parser.add_argument('--learning_rate', default=1e-3, type=float)
    parser.add_argument('--l2', default=0, type=float, help='l2 normalization')
    parser.add_argument('--patience', default=10, help='early stop patience')
    parser.add_argument('--seed', default=-1, help='random seed, -1 means no fixed seed')
    parser.add_argument('--mark', default='', help='log suffix mark')
    # Evaluation
    parser.add_argument('--split_type', default='valid_and_test', choices=['valid_only', 'valid_and_test'])
    parser.add_argument('--split_mode', default='PS', type=str, help='[LS (leave-one-out), LS_R@0.x, PS (pre-split)]')
    parser.add_argument('--eval_mode', default='full', help='[uni100, pop100, full]')
    parser.add_argument('--k', default=[5, 10, 20, 50], help='rank k for each metric')
    parser.add_argument('--metric', default=['hit', 'ndcg'], help='[hit, ndcg, mrr, recall]')
    parser.add_argument('--valid_metric', default='hit@10', help='specifies which indicator to apply early stop')

    config = parser.parse_args()

    trainer = load_trainer(config)
    trainer.start_training()


