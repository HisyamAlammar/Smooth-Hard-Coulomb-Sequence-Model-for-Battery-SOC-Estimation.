**Table 1. Raw Data Integrity Audit**

| Temperature | Drive files | Raw rows | Dropped/NaN rows | Voltage violations (<2.5 or >4.25 V) | Current spikes (|I|>20 A) | SOC anomalies before clipping | SOC rows audited |
| --- | --- | --- | --- | --- | --- | --- | --- |
| 40 C | 12 | 633806 | 22 | 0 | 0 | 0 | 633784 |
| 25 C | 12 | 927095 | 23 | 0 | 0 | 0 | 927072 |
| 10 C | 11 | 847097 | 34 | 0 | 0 | 0 | 847063 |
| 0 C | 11 | 778005 | 21 | 0 | 0 | 0 | 777984 |
| -10 C | 12 | 760615 | 32 | 0 | 0 | 0 | 760583 |
| -20 C | 12 | 552061 | 12 | 0 | 0 | 0 | 552049 |

**Table 2. Scenario Composition After v4 Split-Before-Windowing**

| Scenario | Temperature | Train windows | Validation windows | Test windows |
| --- | --- | --- | --- | --- |
| A | 40 C | 0 | 0 | 6148 |
| A | 25 C | 8199 | 790 | 0 |
| A | 10 C | 7442 | 675 | 0 |
| A | 0 C | 0 | 7619 | 0 |
| A | -10 C | 0 | 0 | 7387 |
| A | -20 C | 0 | 0 | 5379 |
| B | 40 C | 4245 | 448 | 1079 |
| B | 25 C | 6346 | 789 | 1712 |
| B | 10 C | 5750 | 674 | 1520 |
| B | 0 C | 5286 | 618 | 1395 |
| B | -10 C | 5107 | 597 | 1342 |
| B | -20 C | 3723 | 412 | 962 |
