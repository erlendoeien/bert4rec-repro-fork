from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.MOOC_rss_seen import *
from aprec.evaluation.configs.EdNet_rss_warm5 import MAX_TEST_USERS
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import linear_importance, log_importance
from aprec.evaluation.configs.EdNet_rss import PRED_TRUNCATE_AT


DATASET = "ednet_warm5"
recommenders = {"SASRec-rss-linear-bce": lambda: dnn(
                    SASRec(max_history_len=HISTORY_LEN, vanilla=False),
                    BCELoss(),
                    lambda: RecencySequenceSampling(0.2, linear_importance()),
                    #**BASE_KWARGS
                    optimizer=Adam(beta_2=0.98),
                    target_builder=FullMatrixTargetsBuilder, 
                    metric=KerasNDCG(VAL_METRIC_THRESH)
),
                "SASRec-rss-log-bce": lambda: dnn(
                    SASRec(max_history_len=HISTORY_LEN, vanilla=False),
                    BCELoss(),
                    lambda: RecencySequenceSampling(0.2, log_importance()),
                    #**BASE_KWARGS
                    optimizer=Adam(beta_2=0.98),
                    target_builder=FullMatrixTargetsBuilder, 
                    metric=KerasNDCG(VAL_METRIC_THRESH)
                ),
                "SASRec-rss-linear-lambda": lambda: dnn(
                    SASRec(max_history_len=HISTORY_LEN, vanilla=False),
                    LambdaGammaRankLoss(pred_truncate_at=PRED_TRUNCATE_AT),
                    lambda: RecencySequenceSampling(0.2, linear_importance()),
                    #**BASE_KWARGS
                    optimizer=Adam(beta_2=0.98),
                    target_builder=FullMatrixTargetsBuilder, 
                    metric=KerasNDCG(VAL_METRIC_THRESH), 
                    ),
                "SASRec-rss-log-lambda": lambda: dnn(
                    SASRec(max_history_len=HISTORY_LEN, vanilla=False),
                    LambdaGammaRankLoss(pred_truncate_at=PRED_TRUNCATE_AT),
                    lambda: RecencySequenceSampling(0.2, log_importance()),
                    #**BASE_KWARGS
                    optimizer=Adam(beta_2=0.98),
                    target_builder=FullMatrixTargetsBuilder, 
                    metric=KerasNDCG(VAL_METRIC_THRESH), 
                    )
               }


RECOMMENDERS = get_recommenders(recommenders, filter_seen=False)

    
