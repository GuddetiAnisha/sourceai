"""Educational decision tree trained on synthetic policy labels, with hard overrides."""
from functools import lru_cache
import numpy as np
import pandas as pd
from sklearn.tree import DecisionTreeClassifier, export_text
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix

FEATURES = ['risk', 'delivery', 'quality', 'ai_readiness']

@lru_cache(maxsize=1)
def train_model():
    rng = np.random.default_rng(42)
    x = pd.DataFrame(rng.uniform(0, 100, (2400, 4)), columns=FEATURES)
    proxy = .55*x.risk + .25*(100-x.delivery) + .15*(100-x.quality) + .05*(100-x.ai_readiness)
    y = np.where(proxy >= 55, 'High', np.where(proxy >= 30, 'Medium', 'Low'))
    train_x, test_x, train_y, test_y = train_test_split(x, y, test_size=.25, stratify=y, random_state=42)
    model = DecisionTreeClassifier(max_depth=4, min_samples_leaf=30, random_state=42).fit(train_x, train_y)
    predictions = model.predict(test_x)
    return model, {'synthetic_holdout_accuracy': float(accuracy_score(test_y, predictions)), 'holdout_rows': len(test_y), 'confusion_matrix': confusion_matrix(test_y, predictions, labels=['Low', 'Medium', 'High']).tolist()}

def classify(suppliers):
    model, _ = train_model()
    out = suppliers.copy()
    out['model_risk'] = model.predict(out[FEATURES])
    out['risk_class'] = out.model_risk
    critical = (out.risk >= 80) | (out.delivery < 60)
    out.loc[critical, 'risk_class'] = 'High'
    out['risk_reason'] = out.apply(lambda r: ('Override: risk >= 80 or delivery < 60' if r.risk >= 80 or r.delivery < 60 else f'Tree leaf predicts {r.model_risk}') + f'; risk={r.risk:g}, delivery={r.delivery:g}, quality={r.quality:g}, readiness={r.ai_readiness:g}', axis=1)
    return out

def model_rules():
    return export_text(train_model()[0], feature_names=FEATURES)
