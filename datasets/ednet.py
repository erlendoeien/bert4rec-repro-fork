from aprec.api.action import Action
import dask.dataframe as dd
from pathlib import Path

VALID_ITEM_TYPES = {"item_id"}
    
def get_ednet_dataset(item_type="item_id"):
    if item_type not in VALID_ITEM_TYPES:
        raise ValueError(f"unknown mooc cube x {item_type=}")
    
    print("Loading")
    base_path = Path("~/fall_project/EdNet/")
    df = dd.read_parquet(base_path / "single_lecture_events_kt4.parquet").compute()
    
    print("outputting", (df.shape, df.columns))
    # TODO: Use data for contextual features?
    return [Action(user_id, item_id, timestamp, data=None) 
              for _, (user_id, item_id, timestamp) in 
        df[["user_id", item_type, "timestamp"]].iterrows()]

if __name__ == "__main__":
    print(get_ednet_dataset())
      

    
    
    