import pysrt
import glob
import pandas as pd
import re

def load_dataset(path):

    scripts = []
    episode_num = []

    path = path + "/*.srt"
    files = glob.glob(pathname=path)

    for file in files:
        
        lines = []
        subs = pysrt.open(file, encoding='iso-8859-1')

        for sub in subs:
            lines.append(
                sub
                .text
                .replace("\n", " ")
                .replace("<i>", ""))
        
        episode_name = file.split("/")[-1]
        season, episode = get_season_episode(episode_name)
        scripts.append(" ".join(lines))
        episode_num.append(episode)
    
        
    df = pd.DataFrame.from_dict({"episode": episode_num, "script": scripts})
    return df

def get_season_episode(episode_name):
    match = re.search(r"\[(\d{3})\]", episode_name)

    if match:
        episode_number = match.group(1)  
        season = episode_number[0]      
        episode = episode_number[1:]   
        
        return (int(season), int(episode))
    else:
        return (0, 0)
