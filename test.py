from hmmClass import HmmTagger
import pandas as pd

data = pd.read_pickle("data/test/indonesian_ner_test.pkl")
df = pd.DataFrame(data)
x = df['words']
y = df['ner']

tagger = HmmTagger()
tagger.load_parameters("model/indonesian_best_ner_model.json")

tagger.confusion_matriks(
    x=x,
    y=y,
    neg="O"
)