from aprec.evaluation.split_actions import LeaveOneOut
from aprec.evaluation.configs.MOOC_rss_seen import *
from aprec.evaluation.configs.EdNet_rss_warm5 import MAX_TEST_USERS
import numpy as np

DATASET = "ednet_warm5"
# Testing variations of alpha for RSS
alphas = np.linspace(0, 1, 11)
recommenders = {f"Sasrec-rss-bce-{alpha:.2f}": lambda: dnn(
            SASRec(max_history_len=HISTORY_LEN, vanilla=False),
            BCELoss(),
            lambda: RecencySequenceSampling(0.2, exponential_importance(alpha)),
            #**BASE_KWARGS
            optimizer=Adam(beta_2=0.98),
            target_builder=FullMatrixTargetsBuilder, 
            metric=KerasNDCG(VAL_METRIC_THRESH), 
            ) for alpha in alphas}


RECOMMENDERS = get_recommenders(recommenders, filter_seen=False)

    
