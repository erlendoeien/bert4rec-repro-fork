from aprec.api.action import Action
import dask.dataframe as dd
from pathlib import Path

VALID_ITEM_TYPES = {"video_id", "interaction_session"}

def prepare_data(ddf):
    
    # Remove segment specific features
    raw_ddf = ddf.drop(columns=["end_point", "start_point", 
                                    "speed", "gap"])
    
    # Convert floats which are ints to ints
    raw_ddf[["session_id", "local_start_time"]] = (raw_ddf[
        ["session_id", "local_start_time"]
    ].astype(int))
    
    # Keep only the first interaction within a session interaction
    raw_ddf = raw_ddf.drop_duplicates(subset=["user_id",
                                              "video_consecutive_id", 
                                              "seg_rep_count", "video_id", "session_id"])
    # Create interaction_session column
    raw_ddf["interaction_session"] = (raw_ddf["user_id"].astype(str) +
        "-" + raw_ddf["video_consecutive_id"].astype(str) +
        "-" + raw_ddf["session_id"].astype(str))
    
    return raw_ddf

def get_video_id_clean(ddf):
    # Make so only video_ids in separate sessions are returned
    # Return ddf
    raise NotImplementedError

    
# Choose between interaction ids and video_ids
def get_mooc_cube_x_dataset(item_type="video_id"):
    if item_type not in VALID_ITEM_TYPES:
        raise ValueError(f"unknown mooc cube x {item_type=}")
    
    print("Loading")
    # Load as Dask DataFrame (30 partitions)
    base_path = Path("~/fall_project/MOOCCubeX/")
    relations_path = base_path / "relations"
    session2video_id_path = relations_path / "session2video_id_clean_thresh_20_partitions_30"
    raw_ddf = dd.read_parquet(session2video_id_path)
    print("Prepare")
    clean_ddf = prepare_data(raw_ddf)
    
    df = clean_ddf.compute()
    
    print("outputting", (df.shape, df.columns))
    # TODO: Use data for contextual features?
    return [Action(user_id, item_id, timestamp, data=None) 
              for _, (user_id, item_id, timestamp) in 
        df[["user_id", item_type, "local_start_time"]].iterrows()]

if __name__ == "__main__":
    print(get_mooc_cube_x_dataset())
      

    
    
    