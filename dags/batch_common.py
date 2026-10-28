import hashlib
import json
import logging
import math
import os
import re
import tempfile
from datetime import date
from pathlib import Path

import numpy as np

FEATURES = ["age", "tenure_months", "monthly_spend", "support_tickets_30d"]
LIMITS = [(18, 100), (0, 240), (0, 10000), (0, 100)]
ARTIFACTS = Path(os.environ.get("BATCH_ARTIFACTS", "/opt/airflow/include/artifacts"))
log = logging.getLogger(__name__)


def canonical(value):
    return json.dumps(value, sort_keys=True, ensure_ascii=False, allow_nan=False,
                      separators=(",", ":")).encode()


def digest(content):
    return hashlib.sha256(content).hexdigest()


def save(value):
    content = canonical(value)
    sha = digest(content)
    ARTIFACTS.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=ARTIFACTS, delete=False) as file:
        file.write(content)
        temporary = Path(file.name)
    temporary.replace(ARTIFACTS / f"{sha}.json")
    return {"sha256": sha}


def read(ref):
    sha = ref["sha256"]
    if not isinstance(sha, str) or not re.fullmatch(r"[0-9a-f]{64}", sha):
        raise ValueError("Invalid artifact reference")
    content = (ARTIFACTS / f"{sha}.json").read_bytes()
    if digest(content) != sha:
        raise ValueError("Artifact checksum mismatch")
    return json.loads(content)


def business_date(explicit, interval_start):
    if explicit:
        return date.fromisoformat(explicit).isoformat()
    if interval_start is None:
        raise ValueError("business_date is required without a data interval")
    return interval_start.in_timezone("Europe/Moscow").to_date_string()


# Зад. 2.
# 1. «Пустой список запрещён»
# 2. «customer_id должен быть непустой строкой»
# 3. «уникальной в пределах дня»
# 4. «business_date каждой строки совпадает с day»
# 5. «Признаки должны быть конечными числами, строки, bool, NaN и бесконечности запрещены»
# 6. «Диапазоны включают границы: возраст 18-100, стаж 0-240 месяцев, траты 0-10000, обращения 0-100»
# 7. «Возвращает исходный список только после проверки всех строк»
# 8. «Не заменяйте плохое значение нулём и не отбрасывайте клиента молча, ошибка должна остановить
#    инференс до публикации. При нарушении бросайте ValueError»
# 9. «при отсутствии обязательного ключа допустим KeyError»
def validate_rows(rows, day):
    if not rows:  # 1
        raise ValueError("Empty day")
    seen = set()
    for row in rows:
        customer_id = row["customer_id"]  # 9
        if type(customer_id) is not str or not customer_id:  # 2
            raise ValueError(f"Invalid customer_id {customer_id!r}")
        if customer_id in seen:  # 3
            raise ValueError(f"Duplicate customer_id {customer_id}")
        seen.add(customer_id)
        if row["business_date"] != day:  # 4
            raise ValueError(f"{customer_id}: business_date {row['business_date']!r} != {day}")
        for name, (low, high) in zip(FEATURES, LIMITS):  # 6
            value = row[name]  # 9
            # type() вместо isinstance тк bool — подкласс int
            if type(value) not in (int, float) or not math.isfinite(value):  # 5
                raise ValueError(f"{customer_id}: {name}={value!r} is not a finite number")
            if not low <= value <= high:  # 6
                raise ValueError(f"{customer_id}: {name}={value} outside [{low}, {high}]")
    return rows  # 7, 8


def validate_model(model):
    if model["feature_names"] != FEATURES:
        raise ValueError("Feature order mismatch")
    for field in ("mean", "scale", "weights"):
        if len(model[field]) != len(FEATURES):
            raise ValueError("Model dimensions mismatch")
        if any(type(v) not in (int, float) or not math.isfinite(v) for v in model[field]):
            raise ValueError("Invalid model coefficient")
    if any(v <= 0 for v in model["scale"]) or type(model["bias"]) not in (int, float) or not math.isfinite(model["bias"]):
        raise ValueError("Invalid model scale or bias")


def predict(rows, model):
    validate_model(model)
    matrix = np.asarray([[row[name] for name in FEATURES] for row in rows], dtype=np.float64)
    normalized = (matrix - np.asarray(model["mean"])) / np.asarray(model["scale"])
    logits = normalized @ np.asarray(model["weights"]) + model["bias"]
    scores = np.round(1 / (1 + np.exp(-np.clip(logits, -700, 700))), 8)
    return [{"customer_id": row["customer_id"], "score": float(score), "label": int(score >= 0.5)}
            for row, score in zip(rows, scores)]


# Зад. 3.
# 10. «batch_size должен быть целым от 1 до 1000, иначе бросайте ValueError, в том числе для True
#     (помните про особенности питона) и нецелых значений вроде 2.5»
# 11. «Последовательно режьте уже проверенный список на непересекающиеся части и на каждую
#     вызывайте predict(batch, model) ровно один раз»
# 12. «Порядок клиентов сохраняется, последний, вероятно неполный, батч надо обязательно обработать»
# 13. «Используйте уже загруженный model, не открывайте модель внутри цикла»
# 14. «Выведите в лог номер и фактический размер каждой пачки»
def infer_batches(rows, model, batch_size):
    if type(batch_size) is not int or not 1 <= batch_size <= 1000:  # 10
        raise ValueError(f"batch_size must be an integer in [1, 1000], got {batch_size!r}")
    predictions = []
    for number, start in enumerate(range(0, len(rows), batch_size), start=1):  # 11
        batch = rows[start:start + batch_size]  # 12
        log.info("Batch %d: %d rows", number, len(batch))  # 14
        predictions.extend(predict(batch, model))  # 11, 12, 13
    return predictions


# Зад. 4.
# 15. «Количество и множество customer_id должны точно совпадать с expected_ids, дубликаты запрещены»
# 16. «score - конечное число от 0 до 1 включительно»
# 17. «label - значение типа int, равное 0 или 1 (bool запрещён, как и в задании 2)
#     и совпадающее с int(score >= 0.5)»
def validate_predictions(predictions, expected_ids):
    ids = [p["customer_id"] for p in predictions]
    if len(ids) != len(expected_ids) or len(set(ids)) != len(ids) or set(ids) != set(expected_ids):  # 15
        raise ValueError("Predicted customer_ids do not match the validated rows")
    for p in predictions:
        score, label = p["score"], p["label"]
        if type(score) not in (int, float) or not math.isfinite(score) or not 0 <= score <= 1:  # 16
            raise ValueError(f"{p['customer_id']}: invalid score {score!r}")
        if type(label) is not int or label != int(score >= 0.5):  # 17
            raise ValueError(f"{p['customer_id']}: label {label!r} does not match score {score}")


# Зад. 5.
# 18. «До изменения БД запретите пустой predictions ... бросайте ValueError, в сообщении
#     которого есть слово Empty»
# 19. «Границы транзакции задавайте через with conn.transaction()»
# 20. «Возьмите транзакционную advisory-блокировку по паре "дата, версия модели":
#     pg_advisory_xact_lock(hashtextextended(%s, 0)) с параметром day + ':' + model_version»
# 21. «Удалите прежние строки этой даты и версии»
# 22. «Вставьте новые строки параметризованным SQL», «не подставляйте данные в SQL через f-string»
# 23. «Зафиксируйте транзакцию только после всех успешных вставок. Исключение должно вызывать
#     rollback и пробрасываться наружу», «Не делайте отдельный commit после DELETE»
def replace_day(conn, day, model_version, predictions):
    if not predictions:  # 18
        raise ValueError("Empty predictions: refusing to replace the published day")
    with conn.transaction(), conn.cursor() as cur:  # 19, 23
        cur.execute("SELECT pg_advisory_xact_lock(hashtextextended(%s, 0))", (f"{day}:{model_version}",))  # 20
        cur.execute("DELETE FROM predictions WHERE business_date = %s AND model_version = %s", (day, model_version))  # 21
        cur.executemany(  # 22
            "INSERT INTO predictions (business_date, model_version, customer_id, score, label) "
            "VALUES (%s, %s, %s, %s, %s)",
            [(day, model_version, p["customer_id"], p["score"], p["label"]) for p in predictions],
        )
