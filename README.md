# Project_mlflow

MLflow - это инструмент для управления жизненным циклом 
машинного обучения: отслеживание экспериментов, управление и деплой моделей и проектов.

###  Предварительные требования
```python
git clone  https://github.com/VoroninMaxim/Project_mlflow.git
```
### Перейдите в каталог проекта:
```python
cd Project_mlflow
```
Создайте виртуальное окружение (необязательно, но рекомендуется):
```python
python -m venv venv
```
Активируйте виртуальное окружение:
На macOS/Linux::
```python
source venv/bin/activate
```
### Установите необходимые пакеты:
```python
pip install -r requirements.txt
```
### Запуск сервера MLflow
```python
mlflow server --host 127.0.0.1 --port 5000
```
Теперь вы должны иметь возможность получить доступ 
к вашему проекту и MLflow UI по адресу http://127.0.0.1:5000.

### Запустите основного файла приложения:
```python
python app.py
```


![Example Image](https://github.com/VoroninMaxim/Project_mlflow/blob/main/metrics_plot.png)
