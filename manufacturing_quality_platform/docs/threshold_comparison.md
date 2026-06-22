# Threshold Comparison

기준 DB: `python3 scripts/seed_database.py` 실행 후 `data/manufacturing_quality.db`

이 표는 threshold를 낮게 잡을 때 recall은 올라가지만 false alarm이 급증하고, threshold를 높게 잡을 때 missed defect가 생기는 trade-off를 보여준다.

| threshold | precision_defect | recall_defect | false_alarm_rate | false_alarm_count | missed_defect_count | tp | tn |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 0.10 | 0.069 | 1.000 | 0.952 | 1393 | 0 | 104 | 70 |
| 0.50 | 0.126 | 0.683 | 0.338 | 494 | 33 | 71 | 969 |

## 해석

- `threshold=0.10`: 결함을 놓치지 않는 방향이다. 하지만 정상 sample 1,463개 중 1,393개가 알람으로 잡혀 운영자가 감당하기 어렵다.
- `threshold=0.50`: false alarm은 줄지만 결함 104개 중 33개를 놓친다.
- 그래서 포트폴리오의 핵심은 모델 점수 자체가 아니라 `threshold preview -> policy apply -> alert refresh -> 8D follow-up` 흐름이다.

## Reproduce

```bash
sqlite3 -header -column data/manufacturing_quality.db "
WITH thresholds(threshold) AS (VALUES (0.10),(0.50)),
classified AS (
  SELECT threshold, true_defect, defect_score >= threshold AS predicted
  FROM quality_predictions CROSS JOIN thresholds
),
counts AS (
  SELECT
    threshold,
    SUM(true_defect=1 AND predicted=1) AS tp,
    SUM(true_defect=0 AND predicted=0) AS tn,
    SUM(true_defect=0 AND predicted=1) AS fp,
    SUM(true_defect=1 AND predicted=0) AS fn
  FROM classified
  GROUP BY threshold
)
SELECT threshold, tp, tn, fp, fn FROM counts;"
```
