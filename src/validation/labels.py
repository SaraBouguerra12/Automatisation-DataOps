"""Table de correspondance des schémas de labels (source -> schéma unifié)."""
import pandas as pd

LABEL_MAP = pd.DataFrame([
    dict(source="hateful_memes", label_raw="0", label="non_hate", label_task="hate_detection", hate_binary=0, label_source="annotateur"),
    dict(source="hateful_memes", label_raw="1", label="hate", label_task="hate_detection", hate_binary=1, label_source="annotateur"),
    dict(source="sentiment140", label_raw="0", label="negative", label_task="sentiment", hate_binary=pd.NA, label_source="auto (emoticônes)"),
    dict(source="sentiment140", label_raw="4", label="positive", label_task="sentiment", hate_binary=pd.NA, label_source="auto (emoticônes)"),
    dict(source="reddit", label_raw="", label="unlabeled", label_task="none", hate_binary=pd.NA, label_source="non_labellise"),
])
