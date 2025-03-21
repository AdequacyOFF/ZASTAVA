Чтобы заработала программа на компе должен быть PostgreSQL, python 11
1)создаем виртуальное окружение python -m venv venv
2)заходим в него venv\Scrpts\activate
3)Устанавливаем dlib(лежит в корне) pip install dlib-19.24.1-cp311-cp311-win_amd64.whl
4)устанавливаем зависимости pip install -r requirments.txt
5)В pgadmin создаем базу данных zastava, пароль сам ставь не забудь поменять в коде строка 222 и 306
6)В бд делаем запрос 
CREATE TABLE IF NOT EXISTS people (
    id SERIAL PRIMARY KEY,
    surname VARCHAR(255),
    name VARCHAR(255),
    patronymic VARCHAR(255),
    rank VARCHAR(255),
    photo_path VARCHAR(255)
)
7)после установки всех зависимостей и подключения БД делаем команду python main.py
8)чтобы распознавание лица работало надо добавить юзера(можно сделать в приложении)
9)ДВА ВИДЕОПОТОКА НЕ БУДУТ РАБОТАТЬ С ОДНОЙ КАМЕРЫ, ПОЭТОМУ КАК С ОДНИМ ЗАКОНЧИЛ - ВЫКЛЮЧИ ЕГО!
10) Удачи