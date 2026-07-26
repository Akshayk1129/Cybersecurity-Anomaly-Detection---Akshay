# 📋 Data Validation Report

**Generated:** 2026-07-26 01:47:03

---

## Summary

| Severity | Count |
|----------|-------|
| 🔴 ERROR | 0 |
| 🟡 WARNING | 2 |
| 🔵 INFO | 7 |
| 🟢 PASS | 10 |

---

## Dataset Overview

**🔵 INFO:** Loaded **100,000** rows × **30** columns from `Synthetic_Cybersecurity_Access_Logs_v2.xlsx`.

---

## Schema & Data Types

**🔵 INFO:** Detected columns and their data types:

| Column | Dtype | Non-Null Count |
|--------|-------|----------------|
| `event_id` | `int64` | 100,000 |
| `entity_id` | `object` | 100,000 |
| `entity_type` | `object` | 100,000 |
| `department` | `object` | 100,000 |
| `timestamp` | `datetime64[ns]` | 100,000 |
| `login_hour` | `int64` | 100,000 |
| `is_off_hours` | `bool` | 100,000 |
| `day_of_week` | `object` | 100,000 |
| `source_ip` | `object` | 100,000 |
| `geo_location` | `object` | 100,000 |
| `geo_velocity_kmph` | `float64` | 100,000 |
| `resource_accessed` | `object` | 100,000 |
| `resource_type` | `object` | 100,000 |
| `resource_sensitivity` | `object` | 100,000 |
| `auth_method` | `object` | 100,000 |
| `login_status` | `object` | 100,000 |
| `session_duration_min` | `float64` | 100,000 |
| `command_sequence` | `object` | 100,000 |
| `device_fingerprint` | `object` | 100,000 |
| `os` | `object` | 100,000 |
| `browser` | `object` | 58,922 |
| `protocol` | `object` | 100,000 |
| `privilege_level` | `object` | 100,000 |
| `bytes_uploaded` | `int64` | 100,000 |
| `bytes_downloaded` | `int64` | 100,000 |
| `failed_attempts` | `int64` | 100,000 |
| `new_device` | `bool` | 100,000 |
| `new_location` | `bool` | 100,000 |
| `label` | `object` | 100,000 |
| `anomaly_type` | `object` | 100,000 |

---

## Missing Values

**🟡 WARNING:** Found **1** column(s) with missing values:

| Column | Missing Count | Missing % | Severity |
|--------|---------------|-----------|----------|
| `browser` | 41,078 | 41.08% | WARNING |

---

## Duplicate Rows

**🟢 PASS:** No duplicate rows detected.

---

## Timestamp Validation

**🟢 PASS:** Column `timestamp`: All values are valid timestamps.

**🔵 INFO:** Column `timestamp` range: **2026-01-01 00:01:04** -> **2026-06-30 23:22:18** (span: 180 days).

---

## Categorical Validation

**🟢 PASS:** Column `entity_type`: All values match expected set ({'Edge Device', 'Service Account', 'User'}).

**🟢 PASS:** Column `label`: All values match expected set ({'Normal', 'Anomaly'}).

**🟢 PASS:** Column `anomaly_type`: All values match expected set.

**🔵 INFO:** Unique value counts for categorical columns:

| Column | Unique Values | Sample Values |
|--------|---------------|---------------|
| `entity_type` | 3 | Service Account, User, Edge Device |
| `department` | 17 | Data-Pipeline, Finance-Ops, Monitoring, Engineering, Backup-Service, Plant-Floor, Security, Building-Automation, ... |
| `day_of_week` | 7 | Thursday, Friday, Saturday, Sunday, Monday, Tuesday, Wednesday |
| `geo_location` | 9 | Chennai, Bengaluru, Mumbai, Delhi, Hyderabad, Pune, Singapore, New York, ... |
| `resource_accessed` | 11 | API_Gateway, Cloud_Storage, ERP_Server, Finance_DB, SCADA_Server, Backup_Server, Source_Code_Repo, IoT_Gateway, ... |
| `resource_type` | 7 | API, Storage, Application, Database, OT, Repository, Mail |
| `resource_sensitivity` | 4 | Medium, High, Critical, Low |
| `auth_method` | 5 | API_Key, OAuth, Certificate, Password, MFA |
| `login_status` | 2 | Success, Failed |
| `os` | 10 | Linux-Server, Windows-Server, Unknown, RTOS-Firmware, Kali Linux, Embedded-Linux, Windows10, Windows11, ... |
| `protocol` | 4 | HTTPS, REST, SSH, MQTT |
| `privilege_level` | 5 | Service, PowerUser, Device, User, Admin |
| `label` | 2 | Normal, Anomaly |
| `anomaly_type` | 8 | Normal, Device Spoofing, Credential Stuffing, Impossible Travel, Low-and-Slow Exfiltration, Insider Drift, Lateral Movement, Brute Force |

---

## Numeric Range Validation

**🟡 WARNING:** Column `session_duration_min`: **24** negative value(s) detected (min = -15.8). These may indicate data quality issues.

**🟢 PASS:** Column `bytes_uploaded`: All values are non-negative.

**🟢 PASS:** Column `bytes_downloaded`: All values are non-negative.

**🟢 PASS:** Column `failed_attempts`: All values are non-negative.

**🟢 PASS:** Column `geo_velocity_kmph`: All values are non-negative.

**🔵 INFO:** Numeric column statistics and IQR-based outlier counts:

| Column | Min | Q1 | Median | Q3 | Max | IQR Outliers |
|--------|-----|-----|--------|-----|-----|--------------|
| `login_hour` | 0.00 | 7.00 | 10.00 | 12.00 | 23.00 | 4,588 |
| `geo_velocity_kmph` | 0.00 | 0.00 | 0.00 | 0.00 | 126391.80 | 498 |
| `session_duration_min` | -15.80 | 5.00 | 23.20 | 45.70 | 112.40 | 8 |
| `bytes_uploaded` | 1.00 | 183.00 | 317.00 | 533.00 | 4829.00 | 17,262 |
| `bytes_downloaded` | 1.00 | 821.00 | 2570.50 | 4250.00 | 62261.00 | 3,414 |
| `failed_attempts` | 0.00 | 0.00 | 0.00 | 0.00 | 34.00 | 700 |

---

## Label Consistency

**🟢 PASS:** `label` and `anomaly_type` are fully consistent.

**🔵 INFO:** **Label Distribution:**

- `Normal`: 98,000 (98.00%)
- `Anomaly`: 2,000 (2.00%)

**Anomaly Type Distribution:**

- `Normal`: 98,000 (98.00%)
- `Credential Stuffing`: 350 (0.35%)
- `Brute Force`: 350 (0.35%)
- `Low-and-Slow Exfiltration`: 300 (0.30%)
- `Lateral Movement`: 300 (0.30%)
- `Device Spoofing`: 250 (0.25%)
- `Impossible Travel`: 250 (0.25%)
- `Insider Drift`: 200 (0.20%)

---

## Entity Coverage

**🔵 INFO:** Total unique entities: **500**

**Events by Entity Type:**

| Entity Type | Entity Count | Event Count | Avg Events/Entity |
|-------------|-------------|-------------|-------------------|
| `Edge Device` | 50 | 16,031 | 320.6 |
| `Service Account` | 50 | 24,211 | 484.2 |
| `User` | 400 | 59,758 | 149.4 |

---
