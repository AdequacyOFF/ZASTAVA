import psycopg2
from loguru import logger
from psycopg2 import connect

# Подключение к серверу PostgreSQL
conn = psycopg2.connect(
    host="localhost", user="postgres", password="postgres", database="zastava"
)

# Получаем список таблиц базы данных
cur = conn.cursor()
cur.execute("SELECT table_name FROM information_schema.tables")
table_names = cur.fetchall()
logger.debug(table_names)

# Создаем курсор для выполнения запросов
cur = conn.cursor()

# Выполняем запрос на выборку всех строк из таблицы people
cur.execute("SELECT  *  FROM people")
rows = cur.fetchall()
logger.debug(rows)

# Закрываем соединение с базой данных
conn.close()
