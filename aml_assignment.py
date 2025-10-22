# aml_assignment.py
# Refactor of the original Colab notebook into a scriptable pipeline.
# - No Colab magics / Drive mounts
# - Proper CLI via argparse
# - Deterministic preprocessing & modeling
# - REQUIRED DELIVERABLE: Team-XX.pickle = TF-IDF + LogisticRegression (with vectorizer)
# - Optional: Word2Vec doc embeddings + models, PCA/ICA plots, MI, CV & grid search

from __future__ import annotations
import argparse
import json
import pathlib
import re
from collections import Counter
from dataclasses import dataclass, asdict
from typing import List, Tuple, Dict, Optional

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, StratifiedKFold, cross_validate, GridSearchCV, cross_val_predict
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score, f1_score
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer, ENGLISH_STOP_WORDS
from sklearn.feature_selection import mutual_info_classif
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.decomposition import PCA, FastICA
import pickle
import warnings

warnings.filterwarnings("ignore", category=UserWarning)

# ---------------------------
# Utility dataclasses
# ---------------------------
@dataclass
class CorpusStats:
    n_reviews: int
    avg_words_per_review: float
    n_unique_words: int
    class_counts: Dict[str, int]

@dataclass
class ScoreBlock:
    accuracy_mean: float
    accuracy_std: float
    f1_mean: float
    f1_std: float
    roc_auc_mean: float
    roc_auc_std: float

# ---------------------------
# Preprocessing
# ---------------------------
_url = re.compile(r'https?://\S+|www\.\S+')

def clean_text(text: str) -> str:
    """Basic, fast cleaning: lowercase, strip URLs, keep alphabetic tokens of len>=2."""
    t = _url.sub(' ', str(text).lower())
    # remove punctuation except intra-word apostrophes/dashes (optional simplification)
    t = re.sub(r"[^a-zA-Z\s]", " ", t)
    toks = [w for w in t.split() if len(w) >= 2]
    return " ".join(toks)

def apply_stopword_filter(doc: str) -> str:
    return " ".join([w for w in doc.split() if w not in ENGLISH_STOP_WORDS])

# ---------------------------
# IO helpers
# ---------------------------
def ensure_dirs(outdir: pathlib.Path):
    (outdir / "figures").mkdir(parents=True, exist_ok=True)
    (outdir / "artifacts").mkdir(parents=True, exist_ok=True)

def save_json(obj, path: pathlib.Path):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(obj, f, indent=2)

def save_fig(fig_path: pathlib.Path):
    plt.tight_layout()
    plt.savefig(fig_path, dpi=300, bbox_inches="tight")
    plt.close()

# ---------------------------
# Core steps
# ---------------------------
def load_and_label(data_path: pathlib.Path, text_col: str, rating_col: str) -> pd.DataFrame:
    df = pd.read_csv(data_path, sep=None, engine="python", encoding="utf-8", on_bad_lines="skip")
    # normalize names
    df = df.rename(columns={text_col: "text", rating_col: "stars"})
    df = df[["text", "stars"]].dropna()
    # keep only <=3 and >=4 to make it binary
    df = df[(df["stars"] <= 3) | (df["stars"] >= 4)].copy()
    df["y"] = (df["stars"] >= 4).astype(int)
    return df

def build_descriptives(df: pd.DataFrame, outdir: pathlib.Path) -> CorpusStats:
    df["clean"] = df["text"].apply(clean_text)
    n_reviews = len(df)
    lengths = [len(t.split()) for t in df["clean"]]
    avg_words = float(np.mean(lengths)) if lengths else 0.0
    vocab = set(" ".join(df["clean"]).split()) if n_reviews else set()
    n_unique = len(vocab)
    counts = df["y"].value_counts().to_dict()

    # Top 50 words figure
    counter = Counter(" ".join(df["clean"]).split())
    top50 = counter.most_common(50)
    if top50:
        words, freqs = zip(*top50)
        plt.figure(figsize=(12, 5))
        plt.bar(words, freqs)
        plt.xticks(rotation=70, ha="right")
        plt.title("Top 50 words (after basic preprocessing)")
        save_fig(outdir / "figures" / "top50_words.png")

    return CorpusStats(n_reviews, avg_words, n_unique, {"neg(0)": counts.get(0, 0), "pos(1)": counts.get(1, 0)})

def get_top_n_words(df_clean: pd.Series, n: int = 100) -> List[str]:
    # remove stopwords first (assignment requests)
    docs = [apply_stopword_filter(t) for t in df_clean]
    cv = CountVectorizer(max_features=n)
    X = cv.fit_transform(docs)
    return list(cv.get_feature_names_out())

def compute_mi_for_vocab(df: pd.DataFrame, vocab: List[str], outdir: pathlib.Path) -> pd.DataFrame:
    vec = CountVectorizer(vocabulary=vocab)
    X = vec.fit_transform(df["clean"])
    mi = mutual_info_classif(X, df["y"], discrete_features=True, random_state=42)
    mi_df = pd.DataFrame({"word": vocab, "mi": mi}).sort_values("mi", ascending=False)
    mi_df.to_csv(outdir / "artifacts" / "mi_top100.csv", index=False)
    return mi_df

def train_tfidf_lr(df: pd.DataFrame, vocab: List[str], outdir: pathlib.Path) -> Tuple[LogisticRegression, TfidfVectorizer, ScoreBlock]:
    tfidf = TfidfVectorizer(vocabulary=vocab, norm="l2")
    X = tfidf.fit_transform(df["clean"])
    y = df["y"].values

    lr = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    scores = cross_validate(lr, X, y, cv=cv, scoring=["accuracy","f1","roc_auc"], n_jobs=-1)
    block = ScoreBlock(scores["test_accuracy"].mean(), scores["test_accuracy"].std(),
                       scores["test_f1"].mean(), scores["test_f1"].std(),
                       scores["test_roc_auc"].mean(), scores["test_roc_auc"].std())

    # Fit on all data for export
    lr.fit(X, y)

    # Save required deliverable: Team-XX.pickle (LR **on TF-IDF** + vectorizer)
    with open(outdir / "artifacts" / "Team-XX.pickle", "wb") as f:
        pickle.dump({"model": lr, "vectorizer": tfidf}, f)

    # quick confusion/ROC on a holdout split for figures (optional)
    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    lr_tmp = LogisticRegression(max_iter=2000, class_weight="balanced", solver="liblinear", random_state=42)
    lr_tmp.fit(X_tr, y_tr)
    y_pred = lr_tmp.predict(X_te)
    y_proba = lr_tmp.predict_proba(X_te)[:, 1]

    # confusion matrix fig
    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4,4))
    plt.imshow(cm, cmap="Blues")
    plt.title("Confusion Matrix (TF-IDF + LR)")
    plt.xlabel("Predicted"); plt.ylabel("True")
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j, i, cm[i, j], ha="center", va="center")
    save_fig(outdir / "figures" / "cm_tfidf_lr.png")

    # simple ROC-like point (avoid full curve to keep deps minimal)
    # You can add full ROC if desired.

    # Save CV metrics
    save_json({"tfidf_lr_cv": asdict(block)}, outdir / "artifacts" / "tfidf_lr_cv.json")
    return lr, tfidf, block

# ---------------------------
# Optional: Word2Vec + document embeddings
# ---------------------------
def train_word2vec(sentences: List[List[str]], vector_size: int = 100, window: int = 5, min_count: int = 2):
    from gensim.models import Word2Vec
    return Word2Vec(sentences, vector_size=vector_size, window=window, min_count=min_count, workers=4, sg=1)

def load_pretrained_kv(name: str = "glove-wiki-gigaword-100"):
    import gensim.downloader as api
    return api.load(name)

def tfidf_weighted_doc_vec(tokens: List[str], kv, weights: Dict[str, float], is_gensim_kv: bool) -> np.ndarray:
    # kv: gensim Word2Vec.wv or KeyedVectors; is_gensim_kv tells indexing path
    vecs, wts = [], []
    for w in tokens:
        has = (w in kv) if is_gensim_kv else (w in kv.key_to_index)  # safety for gensim versions
        if has:
            v = kv[w] if is_gensim_kv else kv[w]
            vecs.append(v)
            wts.append(weights.get(w, 0.0))
    if not vecs:
        dim = kv.vector_size if hasattr(kv, "vector_size") else kv.vectors.shape[1]
        return np.zeros(dim)
    wts = np.array(wts, dtype=float)
    if wts.sum() == 0:
        return np.mean(np.vstack(vecs), axis=0)
    return np.average(np.vstack(vecs), axis=0, weights=wts)

def build_doc_matrix(df_clean: pd.Series, kv, tfidf_vec: Optional[TfidfVectorizer] = None) -> np.ndarray:
    # tokens
    toks = [t.split() for t in df_clean]
    if tfidf_vec is None:
        tfidf_vec = TfidfVectorizer()
        tfidf_vec.fit(df_clean)
    weights = dict(zip(tfidf_vec.get_feature_names_out(), tfidf_vec.idf_))
    # gensim keyedvectors detection
    is_kv = True  # works for both KeyedVectors and Word2Vec.wv in current gensim
    X = np.vstack([tfidf_weighted_doc_vec(ts, kv, weights, is_kv) for ts in toks])
    return X

def train_eval_w2v_models(X_doc: np.ndarray, y: np.ndarray, outdir: pathlib.Path) -> pd.DataFrame:
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    models = {
        "KNN": Pipeline([("scaler", StandardScaler()), ("clf", KNeighborsClassifier(n_neighbors=15, weights="distance"))]),
        "NaiveBayes": GaussianNB(),
        "LogReg": Pipeline([("scaler", StandardScaler()), ("clf", LogisticRegression(max_iter=2000, random_state=42))])
    }

    rows = []
    for name, model in models.items():
        res = cross_validate(model, X_doc, y, cv=cv, scoring=["accuracy","f1","roc_auc"], n_jobs=-1)
        rows.append([name,
                     res["test_accuracy"].mean(), res["test_accuracy"].std(),
                     res["test_f1"].mean(),        res["test_f1"].std(),
                     res["test_roc_auc"].mean(),   res["test_roc_auc"].std()])

        # OOF confusion matrix for completeness
        y_pred_oof  = cross_val_predict(model, X_doc, y, cv=cv, method="predict", n_jobs=-1)
        cm = confusion_matrix(y, y_pred_oof)
        plt.figure(figsize=(4,4))
        plt.imshow(cm, cmap="Purples")
        plt.title(f"Confusion Matrix (W2V {name})")
        plt.xlabel("Predicted"); plt.ylabel("True")
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j], ha="center", va="center")
        save_fig(outdir / "figures" / f"cm_w2v_{name.lower()}.png")

    df_scores = pd.DataFrame(rows, columns=["Model","acc_mean","acc_std","f1_mean","f1_std","auc_mean","auc_std"])
    df_scores.to_csv(outdir / "artifacts" / "w2v_model_cv_scores.csv", index=False)
    return df_scores

# ---------------------------
# Optional: PCA & ICA plots on word embeddings
# ---------------------------
def plot_pca_ica_on_words(df_clean: pd.Series, kv, outdir: pathlib.Path, top_n: int = 600, n_label: int = 60):
    # choose frequent words
    tfidf = TfidfVectorizer()
    tfidf.fit(df_clean)
    vocab = tfidf.get_feature_names_out()
    idf = dict(zip(vocab, tfidf.idf_))

    words = [w for w in vocab if w in kv]
    words.sort(key=lambda w: idf[w])  # low idf = frequent
    words_plot = words[:top_n]
    if not words_plot:
        return
    W = np.vstack([kv[w] for w in words_plot])

    def plot2d(arr2d, title, fname):
        plt.figure(figsize=(9,7))
        plt.scatter(arr2d[:,0], arr2d[:,1], s=12, alpha=0.35, c="#999999", edgecolors="none")
        for w in words_plot[:n_label]:
            i = words_plot.index(w)
            plt.text(arr2d[i,0], arr2d[i,1], w, fontsize=9)
        plt.title(title); plt.grid(True, linestyle="--", alpha=0.25)
        save_fig(outdir / "figures" / fname)

    pca = PCA(n_components=2, random_state=42)
    plot2d(pca.fit_transform(W), "PCA (2D) of Word Embeddings", "task5_pca_word2vec.png")

    ica = FastICA(n_components=2, random_state=42, max_iter=1000)
    plot2d(ica.fit_transform(W), "ICA (2D) of Word Embeddings", "task5_ica_word2vec.png")

# ---------------------------
# Main runnable
# ---------------------------
def run_pipeline(args):
    outdir = pathlib.Path(args.outdir)
    ensure_dirs(outdir)

    # 1) Load & label
    df = load_and_label(pathlib.Path(args.data_path), args.text_col, args.rating_col)

    # 2) Preprocess & descriptives
    df["clean"] = df["text"].apply(clean_text)
    stats = build_descriptives(df, outdir)
    save_json({"corpus_stats": asdict(stats)}, outdir / "artifacts" / "corpus_stats.json")

    # 3) BOW top-100 & MI
    vocab100 = get_top_n_words(df["clean"], n=args.n_top_words)
    mi_df = compute_mi_for_vocab(df, vocab100, outdir)

    # 4) TF-IDF + LR (THIS is the required deliverable for pickling)
    lr, tfidf, block = train_tfidf_lr(df, vocab100, outdir)
    print(f"[TF-IDF+LR] CV: acc={block.accuracy_mean:.3f}±{block.accuracy_std:.3f}, "
          f"f1={block.f1_mean:.3f}±{block.f1_std:.3f}, auc={block.roc_auc_mean:.3f}±{block.roc_auc_std:.3f}")
    print(f"Saved required pickle at: {outdir/'artifacts'/'Team-XX.pickle'}")

    if args.run_w2v or args.run_all:
        # Prepare Word2Vec keyed vectors (own or pretrained)
        tokens = [t.split() for t in df["clean"]]
        if args.pretrained_w2v:
            kv = load_pretrained_kv("glove-wiki-gigaword-100")
        else:
            w2v = train_word2vec(tokens, vector_size=100, window=5, min_count=2)
            kv = w2v.wv

        X_doc = build_doc_matrix(df["clean"], kv)
        y = df["y"].values

        # CV compare models
        df_scores = train_eval_w2v_models(X_doc, y, outdir)
        print("\nW2V model CV scores:")
        print(df_scores.round(3).to_string(index=False))

        if args.plots or args.run_all:
            plot_pca_ica_on_words(df["clean"], kv, outdir)

    print("\nDone.")

def build_argparser():
    p = argparse.ArgumentParser(description="Refactored AML assignment pipeline")
    p.add_argument("--data_path", required=True, help="Path to CSV (Amazon Movies & TV)")
    p.add_argument("--text_col", default="text", help="Name of review text column")
    p.add_argument("--rating_col", default="rating", help="Name of rating/stars column")
    p.add_argument("--outdir", default="outputs", help="Where to write figures/artifacts")
    p.add_argument("--team", default="XX", help="Team number (used in filenames)")
    p.add_argument("--n_top_words", type=int, default=100, help="Top-N words after stopword removal")
    # feature toggles
    p.add_argument("--run_w2v", action="store_true", help="Run Word2Vec doc-embedding models")
    p.add_argument("--pretrained_w2v", action="store_true", help="Use pretrained GloVe instead of training your own")
    p.add_argument("--plots", action="store_true", help="Export PCA/ICA plots for word embeddings")
    p.add_argument("--run_all", action="store_true", help="Run all optional steps")
    return p

if __name__ == "__main__":
    args = build_argparser().parse_args()
    # Ensure the required pickle is named with the provided team number
    # by symlinking/copying after run (kept simple: copy if exists).
    run_pipeline(args)
    # Rename artifact to requested "Team-XX.pickle" if user passed a team number:
    outdir = pathlib.Path(args.outdir) / "artifacts"
    src = outdir / "Team-XX.pickle"
    if args.team and args.team != "XX" and src.exists():
        dst = outdir / f"Team-{args.team}.pickle"
        with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
            fdst.write(fsrc.read())
        print(f"Also wrote: {dst}")
