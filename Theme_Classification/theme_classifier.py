import torch
from transformers import pipeline
from nltk import sent_tokenize
from collections import defaultdict
import numpy as np
import pandas as pd
from utils import load_dataset
import os

MODEL_NAME = "facebook/bart-large-mnli"

class ThemeClassifier():

    def __init__(self, theme_list):
        self.model_name = MODEL_NAME
        self.device = 0 if torch.cuda.is_available() else 'cpu'
        self.theme_list = theme_list
        self.theme_classifier = self.load_model(self.device)

    def load_model(self, device):
        theme_classifier = pipeline(
            "zero-shot-classification",
            model=MODEL_NAME,
            device=device
        )
    
        return theme_classifier

    def get_theme_inference(self, script):

        script_sentence = sent_tokenize(script)
        script_batches = []
        
        for i in range(0, len(script_sentence), 20):
            sent = " ".join(script_sentence[i:i+20])
            script_batches.append(sent)
        
        theme_output = self.theme_classifier(
            script_batches[:2],
            self.theme_list,
            multi_label=True
        )

        themes = defaultdict(list)

        for output in theme_output:
            for label, score in zip(output["labels"], output["scores"]):
                themes[label].append(score)
        
        themes = {theme: np.mean(np.array(score)) for theme, score in themes.items()}

        return themes

    def get_themes(self, dataset_path):

        df = load_dataset(dataset_path)
        df = df.head(2)
        output_themes = df["script"].apply(self.get_theme_inference)
        themes_df = pd.DataFrame(output_themes.tolist())
        
        df[themes_df.columns] = themes_df

        return df


