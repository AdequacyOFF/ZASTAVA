import psycopg2
from loguru import logger
from psycopg2 import connect

from dotenv import dotenv_values
class DB:
    def __init__(self):
        config = dotenv_values(".env")
        if not config:
            raise ValueError("Не найден файл .env")
        self.host: str = config['DB_HOST']
        self.port: int = int(config['DB_PORT'])
        self.database: str = config['DB_DATABASE']
        self.user: str = config['DB_USER']
        self.password: str = config['DB_PWD']
        self.conn = psycopg2.connect(
            host=self.host, user=self.user, password=self.password, database=self.database, port=self.port
        )
        self.cursor = self.conn.cursor()

        # Создание таблицы, если она не существует
        self.cursor.execute(
            """
            CREATE TABLE IF NOT EXISTS people (
                id SERIAL PRIMARY KEY,
                surname VARCHAR(255),
                name VARCHAR(255),
                patronymic VARCHAR(255),
                rank VARCHAR(255),
                photo_path VARCHAR(255)
            )
        """
        )
        self.conn.commit()

    def __del__(self):
        self.cursor.close()
        self.conn.close()

    # Выполнение SQL-запроса для получения информации о пользователе по пути
    def get_user_by_image_path(self, image_path: str) -> tuple | None:

        self.cursor.execute(f"SELECT * FROM people WHERE photo_path='{image_path}'")
        user_data = self.cursor.fetchone()
        return user_data

    def add_user(self,
                 surname: str = '',
                 name: str = '',
                 patronymic: str = '',
                 rank: str = '',
                 photo_path: str = '') -> None:
        # Вставка данных в таблицу
        self.cursor.execute(
            """
            INSERT INTO people (surname, name, patronymic, rank, photo_path)
            VALUES (%s, %s, %s, %s, %s)
        """,
            (surname, name, patronymic, rank, photo_path),
        )
        self.conn.commit()

    def get_all_users(self) -> list:
        self.cursor.execute("SELECT  *  FROM people")
        rows = self.cursor.fetchall()
        return rows
