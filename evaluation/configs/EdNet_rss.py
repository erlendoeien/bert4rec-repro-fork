from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.MOOC_rss_seen import *

# OVERRIDE LAMBDA GAMMA RANK PRED TRUNCATE TO NUMBER OF ITEMS
# Not changed through cold, warm5, warm10
PRED_TRUNCATE_AT = 971

recommenders = {**recommenders, **{
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
            )}
        }

RECOMMENDERS = get_recommenders(recommenders, filter_seen=False)
DATASET = "ednet"
N_VAL_USERS = 1024
# Total number of users in the dataset
MAX_TEST_USERS = 17906
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)

    
