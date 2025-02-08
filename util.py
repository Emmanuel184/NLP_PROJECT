import pysrt

def print_subs():
    subs = pysrt.open("/Users/homedirel/ML/DATA/NLP_DATA/4_season_office/The Office [401] Fun Run English.srt", encoding='iso-8859-1')
    for sub in subs:
        print(f"{sub}\n aaaa")