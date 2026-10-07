# Dataset — Credit Card Fraud Detection

## Source
- **Provider:** ULB Machine Learning Group
- **Platform:** Kaggle — [Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **License:** Open Database License (ODbL)

## File
| File | Size | Rows | Columns |
|------|------|------|---------|
| `creditcard.csv` | ~144 MB | 284,807 | 31 |

## Dataset Description
This dataset contains credit card transactions made by European cardholders over two days in September 2013.

## Columns
| Column | Type | Description |
|--------|------|-------------|
| `Time` | float64 | Seconds elapsed since the first transaction in the dataset |
| `V1`–`V28` | float64 | PCA-transformed features (original features are confidential for privacy) |
| `Amount` | float64 | Transaction amount in euros |
| `Class` | int64 | Target variable: **0** = Legitimate, **1** = Fraud |

## Class Distribution (Real)
| Class | Count | Percentage |
|-------|-------|------------|
| Legitimate (0) | 284,315 | 99.8273% |
| Fraudulent (1) | 492 | 0.1727% |
| **Total** | **284,807** | **100%** |

**Imbalance Ratio:** 1:577 — for every 1 fraudulent transaction, there are 577 legitimate ones.

## Key Statistics
- **Time span:** ~48 hours (172,792 seconds)
- **Missing values:** 0
- **Fraud mean amount:** $122.21 | **median:** $9.25 | **max:** $2,125.87
- **Legit mean amount:** $88.29 | **median:** $22.00 | **max:** $25,691.16

## Privacy Note
Features V1–V28 are the result of PCA transformation applied by the original researchers to protect cardholder privacy. The original feature names and their business meanings are confidential and cannot be recovered from the transformed data.

## .gitignore Note
`creditcard.csv` is excluded from Git tracking via `.gitignore` due to its 144 MB size exceeding GitHub's file size limit. Download it from Kaggle and place it in this `data/` folder before running any scripts.
