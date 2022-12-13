from aprec.evaluation.configs.MOOC_baselines import *
from aprec.evaluation.split_actions import LeaveOneOut

DATASET = "ednet_warm5"
N_VAL_USERS = 1024
# Total number of users in the dataset
MAX_TEST_USERS = 17906
SPLIT_STRATEGY = LeaveOneOut(MAX_TEST_USERS)
