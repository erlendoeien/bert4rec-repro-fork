from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.rss_paper.common_benchmark_config import *

BASE_KWARGS = {"optimizer": Adam(beta_2=0.98),               
               "target_builder": FullMatrixTargetsBuilder,
               "metric": KerasNDCG(40),
              }
VAL_METRIC_THRESH = 10
PRED_TRUNCATE_AT = 4000

recommenders = {
          "SASRec-vanilla": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, 
                            dropout_rate=0.2,
                            num_heads=1,
                            num_blocks=2,
                            vanilla=True, 
                            embedding_size=50,
                    ),
            BCELoss(),
            ShiftedSequenceSplitter,
            optimizer=Adam(beta_2=0.98),
            target_builder=lambda: NegativePerPositiveTargetBuilder(HISTORY_LEN), 
            metric=BCELoss(),
            ),

        # BCE
        "Sasrec-continuation-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            SequenceContinuation,
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
        "Sasrec-rss-bce": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
        
        # LambdaRank
        "Sasrec-continuation-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=PRED_TRUNCATE_AT),
            SequenceContinuation,
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
    
        "Sasrec-rss-lambdarank": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            LambdaGammaRankLoss(pred_truncate_at=PRED_TRUNCATE_AT),
            lambda: RecencySequenceSampling(0.2, exponential_importance(0.8)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
}

def get_recommenders(recommenders = dict(), filter_seen = False, filter_recommenders = set()):
    result = {}
    for recommender_name in recommenders:
        if recommender_name in filter_recommenders:
                continue
        if filter_seen:
            result[recommender_name] =\
                lambda recommender_name=recommender_name: FilterSeenRecommender(recommenders[recommender_name]())
        else:
            result[recommender_name] = recommenders[recommender_name]
    return result

USERS_FRACTIONS = [1]

DATASET = "mooc-cube-x"

RECOMMENDERS = get_recommenders(recommenders, filter_seen=False)
METRICS = [HIT(1), HIT(5), HIT(10), NDCG(5), NDCG(10), MRR(), MAP(5), MAP(10), NDCG(50)]


N_VAL_USERS=4096
# Total number of users
# Infer # validation users (10% of train set)
# Removing validation users
MAX_TEST_USERS = 303634# - int(303634*0.1) #138493
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
    
