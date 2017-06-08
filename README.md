NegNeg
======
Method for negation detection

Warning: Preprocess all the data with same vect_lemma (CountVectorizer) because sentences are shuffled so
different words will go to CountVectorizer.

There are 4 versions of classifier:
* _sig-bow_ - detecting negation signal with feautures from "bag of words"
* _sig_ - detecting negation signal
* _sco-bow_ - detecting negation scope with feautures from "bag of words"
* _sco_ - detecting negation scope

Useful commands
---------------
**Show dataframes side by side**
```
import pandas as pd
pd.concat([Y_test, X_test[['has_sk_prefix', 'word_without_prefix_exist']]], axis=1)
```
