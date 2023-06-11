# Используем базовый образ Python
FROM python:3.9

# Установка зависимостей
RUN pip install networkx scipy numpy tqdm

# Создание рабочей директории
WORKDIR /app

# Копирование файлов в образ
COPY independent_set.py /app

# Запуск команды при запуске контейнера
CMD ["python", "independent_set.py"]