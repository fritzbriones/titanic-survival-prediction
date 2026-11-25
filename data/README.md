# Data

This project uses the data from Kaggle's **Titanic - Machine Learning from Disaster** competition:
https://www.kaggle.com/competitions/titanic/data

The files are not included here, because Kaggle's competition rules don't allow re-sharing the data. To get them, sign in to Kaggle, accept the competition rules, and download:

| File | Rows | Contents |
|---|---|---|
| `train.csv` | 891 | Passenger details plus `Survived` (0 = died, 1 = survived) |
| `test.csv` | 418 | Passenger details without `Survived`; predict these |
| `gender_submission.csv` | 418 | Example submission (all women survive) |

On Kaggle Notebooks the files are at `/kaggle/input/competitions/titanic/`. The notebook finds them automatically.

Citation: Will Cukierski. *Titanic - Machine Learning from Disaster.* Kaggle, 2012.
