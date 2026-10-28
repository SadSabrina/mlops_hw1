from datetime import timedelta

import pendulum
import requests
from airflow.providers.postgres.hooks.postgres import PostgresHook
from airflow.sdk import BaseHook, Param, dag, get_current_context, task

import batch_common as core

MODEL_VERSION = "v1"  # 12
HTTP_TIMEOUT = 10  # 12


# Общий GET для данных и модели
# 1. «Запрос GET {host}/basic/{day}.json с явным таймаутом 10 секунд и проверкой HTTP-статуса
#    (например, raise_for_status())»
# 2. «Запрос модели GET {host}/models/v1.json по тем же правилам»
def fetch(url):
    response = requests.get(url, timeout=HTTP_TIMEOUT)  # 1
    response.raise_for_status()  # 1
    return response


# Задание 1. Настройки DAG из таблицы в hse_mlops/mlops-hw-1-base/ASSIGNMENT.md
# 3. «dag_id — hw_basic_batch_inference»
# 4. «start_date — 1 сентября 2026, 00:00, Europe/Moscow»
# 5. «schedule — None»
# 6. «catchup — False»
# 7. «max_active_runs — 1»
# 8. «tags — homework, basic»
# 9. «business_date: по умолчанию "2026-09-01", тип string, формат date»
# 10. «batch_size: по умолчанию 32, тип integer, minimum 1, maximum 1000»
# 11. «default_args для всех задач: retries=2, retry_delay 5 секунд, execution_timeout 2 минуты»
# 12. «на верхнем уровне допустимы только импорты, константы и объявление DAG»
@dag(
    dag_id="hw_basic_batch_inference",  # 3
    start_date=pendulum.datetime(2026, 9, 1, tz="Europe/Moscow"),  # 4
    schedule=None,  # 5
    catchup=False,  # 6
    max_active_runs=1,  # 7
    tags=["homework", "basic"],  # 8
    params={
        "business_date": Param("2026-09-01", type="string", format="date"),  # 9
        "batch_size": Param(32, type="integer", minimum=1, maximum=1000),  # 10
    },
    default_args={  # 11
        "retries": 2,
        "retry_delay": timedelta(seconds=5),
        "execution_timeout": timedelta(minutes=2),
    },
)
def hw_basic_batch_inference():
    # extract
    # 13. «Дата определяется через business_date(...) из параметра business_date и data_interval_start
    #     контекста (если его нет, передаётся None)»
    # 14. «Базовый URL источника берётся из поля host подключения source_api»
    # 15. «Если business_date в ответе не совпадает с запрошенной датой, бросайте ValueError»
    # 16. «model_sha256 считается через digest от точных байтов тела ответа»
    # 17. «Выход: save(...) объекта с ключами business_date, rows, model, model_sha256.
    #     В XCom только ссылка»
    # 18. «Повторы наследуются (retries=2)»
    @task  # 18
    def extract():
        context = get_current_context()
        day = core.business_date(context["params"]["business_date"], context.get("data_interval_start"))  # 13
        host = BaseHook.get_connection("source_api").host  # 14

        payload = fetch(f"{host}/basic/{day}.json").json()  # 1
        if payload["business_date"] != day:  # 15
            raise ValueError(f"Source returned {payload['business_date']!r} for requested {day}")

        model_response = fetch(f"{host}/models/{MODEL_VERSION}.json")  # 2
        return core.save({  # 17
            "business_date": day,
            "rows": payload["rows"],
            "model": model_response.json(),
            "model_sha256": core.digest(model_response.content),  # 16
        })

    # infer
    # 19. «вызовите validate_rows (задание 2), затем infer_batches (задание 3) с моделью из артефакта
    #     и batch_size, затем validate_predictions (задание 4) со списком customer_id проверенных строк»
    # 20. «Выход: save(...) объекта с ключами business_date, model_version ("v1"), predictions,
    #     model_sha256. В XCom только ссылка»
    # 21. «переопределите retries=0 на уровне задачи»
    @task(retries=0)  # 21
    def infer(ref):
        data = core.read(ref)
        rows = core.validate_rows(data["rows"], data["business_date"])  # 19
        predictions = core.infer_batches(rows, data["model"], get_current_context()["params"]["batch_size"])  # 19
        core.validate_predictions(predictions, [row["customer_id"] for row in rows])  # 19
        return core.save({  # 20
            "business_date": data["business_date"],
            "model_version": MODEL_VERSION,
            "predictions": predictions,
            "model_sha256": data["model_sha256"],
        })

    # publish
    # 22. «Внутри задачи получите новое соединение PostgresHook(postgres_conn_id="warehouse").get_conn()»
    # 23. «До вызова replace_day не выполняйте на этом соединении запросов»
    # 24. «закройте соединение в finally»
    # 25. «Выход: словарь business_date, model_version, rows (число опубликованных строк)»
    @task
    def publish(ref):
        result = core.read(ref)
        conn = PostgresHook(postgres_conn_id="warehouse").get_conn()  # 22
        try:
            core.replace_day(conn, result["business_date"], result["model_version"], result["predictions"])  # 23
        finally:
            conn.close()  # 24
        return {  # 25
            "business_date": result["business_date"],
            "model_version": result["model_version"],
            "rows": len(result["predictions"]),
        }

    # Итого
    # 26. «Ровно три задачи extract, infer, publish с зависимостями extract >> infer >> publish»
    # 27. «каждая задача получает ссылку, возвращённую предыдущей»
    publish(infer(extract()))  # 26, 27


hw_basic_batch_inference()
