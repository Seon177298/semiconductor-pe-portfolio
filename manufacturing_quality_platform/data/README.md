# Data

This project seeds SQLite from the public UCI SECOM files in `../secom_quality_8d/data/raw` when available. If those files are absent, `scripts/seed_database.py` downloads the UCI files into `data/raw/`.

Optional MVTec AD data can be placed under:

```text
data/mvtec/bottle/
```

MVTec AD is for non-commercial research and educational use. Do not present it as company field data.

