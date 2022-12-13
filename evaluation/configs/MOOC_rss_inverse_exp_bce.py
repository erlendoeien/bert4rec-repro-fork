from aprec.evaluation.configs.MOOC_rss_seen import *
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.recency_sequence_sampling import inv_exponential_importance

DATASET = "mooc-cube-x_warm5"

# Testing variations of alpha for RSS
recommenders = {
    f"Sasrec-rss-bce-{1.0}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(1.0)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
        f"Sasrec-rss-bce-{0.9}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.9)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
    f"Sasrec-rss-bce-{0.7}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.7)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
    f"Sasrec-rss-bce-{0.8}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.8)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
                f"Sasrec-rss-bce-{0.6}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.6)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
                f"Sasrec-rss-bce-{0.5}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.5)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
                    f"Sasrec-rss-bce-{0.4}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.4)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
                    f"Sasrec-rss-bce-{0.3}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.3)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
                    f"Sasrec-rss-bce-{0.2}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.2)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
                    f"Sasrec-rss-bce-{0.1}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.1)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
                    f"Sasrec-rss-bce-{0.0}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, inv_exponential_importance(0.0)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ),
               }


RECOMMENDERS = get_recommenders(recommenders, filter_seen=False)

    
