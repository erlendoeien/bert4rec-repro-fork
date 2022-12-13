from aprec.evaluation.configs.MOOC_rss_seen import *
from aprec.recommenders.dnn_sequential_recommender.models.bert4rec.bert4rec import BERT4Rec
from aprec.losses.mean_ypred_ploss import MeanPredLoss
from aprec.recommenders.dnn_sequential_recommender.target_builders.items_masking_target_builder import ItemsMaskingTargetsBuilder
from aprec.recommenders.dnn_sequential_recommender.targetsplitters.items_masking import ItemsMasking
from aprec.recommenders.dnn_sequential_recommender.history_vectorizers.add_mask_history_vectorizer import AddMaskHistoryVectorizer
from aprec.recommenders.vanilla_bert4rec import VanillaBERT4Rec
from aprec.recommenders.random_recommender import RandomRecommender

def vanilla_bert4rec(time_limit):
    recommender = VanillaBERT4Rec(training_time_limit=time_limit, num_train_steps=10000000)
    return recommender

DATASET = "mooc-cube-x_warm5"
# Copoed from repro paper, just adjusted to 3600 training time as the others ("our_bert4rec")
def bert4rec_repro(relative_position_encoding=False, sequence_len=50, 
                 rss = lambda n, k: 1, layers=2, arch=BERT4Rec, 
                 masking_prob=0.2, max_predictions_per_seq=20):
        model = arch( max_history_len=sequence_len)
        recommender = DNNSequentialRecommender(model, train_epochs=10000, early_stop_epochs=200,
                                               batch_size=64,
                                               training_time_limit=3600, 
                                               loss = MeanPredLoss(),
                                               debug=True, sequence_splitter=lambda: ItemsMasking(masking_prob=masking_prob,
                                                                                                  max_predictions_per_seq=max_predictions_per_seq,
                                                                                                  recency_importance=rss), 
                                               targets_builder= lambda: ItemsMaskingTargetsBuilder(relative_positions_encoding=relative_position_encoding),
                                               val_sequence_splitter=lambda: ItemsMasking(force_last=True),
                                               metric=MeanPredLoss(), 
                                               pred_history_vectorizer=AddMaskHistoryVectorizer(),
                                               )
        return recommender

    
# Already have Vanilla SASRec
recommenders = {
    "random": lambda : RandomRecommender(),
    "top": top_recommender, 
    "bert4rec-repro": bert4rec_repro,
    "bert4rec-original": lambda : vanilla_bert4rec(3600)
}
RECOMMENDERS = get_recommenders(recommenders, filter_seen=False, filter_recommenders= ("bert4rec-original", "bert4rec-repro"))
