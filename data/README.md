# Data

This project uses the data from Kaggle's **Titanic - Machine Learning from Disaster** competition:
https://www.kaggle.com/competitions/titanic/data

The files are not included here, because Kaggle's competition rules don't allow re-sharing the data. To get them, sign in to Kaggle, accept the competition rules, and download:

| File | Rows | Contents |
|---|---|---|
| `train.csv` | 891 | Passenger details plus `Survived` (0 = died, 1 = survived) |
| `test.csv` | 418 | Passenger details without `Survived`; predict these |
| `gender_submission.csv` | 418 | Example submission (all women survive) |

To run locally, put the three files in this `data/` folder (CSV files here are git-ignored). On Kaggle Notebooks they are at `/kaggle/input/competitions/titanic/`. The notebooks find them automatically in either place.

Citation: Will Cukierski. *Titanic - Machine Learning from Disaster.* Kaggle, 2012.
