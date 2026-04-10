# Data Layout

This project expects the Lending Club dataset to be organized like this:

```text
data/
  raw/
    accepted_2007_to_2018Q4.csv
  processed/
    prepared_dataset.csv
```

## Expected Raw File

- `data/raw/accepted_2007_to_2018Q4.csv`

This is the file used by the training and EDA commands in the project README.

## Notes

- Keep the original downloaded CSV in `data/raw/`.
- Store any cleaned or engineered outputs in `data/processed/`.
- The repository does not commit the real dataset because it is too large.
