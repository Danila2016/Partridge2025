Russian version [below](#по-русски)

There is an equality for the number 2025:
$2025 = 45^2 = (1+2+\ldots+9)^2 = 1^3 + 2^3 + \ldots + 9^3$

Therefore, it is possible to state the following problem:

*How many combinations of 1 square of size 1, 2 squares of size 2, 3 squares of size 3, ..., 8 squares of size 8, 9 squares of size 9 exist such that they fill the square of size 45 with no intersection?*

This repository contains 3 main files:

1. cuda9_3.py - main file for computing the combinations
2. process9.py - auxillary file for processing results
3. plot.py - auxillary file for visualization

Use requirements.txt file to install pip libraries (preferably in a virtual environment).

Computation requires Nvidia GPU with CUDA installed.

For a quick test set N = int32(8) and SIZE = int32(36) in cuda9_3.py. You should get 2332 after running cuda9_3.py and process9.py.
```
python cuda9_3.py > result9.txt
python process9.py
```

You can see a few examples of combinations of SIZE 45 in the file 100examples.zip.

More details on the solution in [https://habr.com/ru/articles/889410/](my article) (in Russian).

## По-русски

Задача: Принимая во внимание равенство

$2025 = 45^2 = (1+2+\ldots+9)^2 = 1^3 + 2^3 + \ldots + 9^3$

*Найти сколько всего вариантов расположить 1 квадратик размера 1, 2 квадратика размера 2, 3 квадратика размера 3, ... , 8 квадратиков размера 8, 9 квадратиков размера 9 в квадрате со строной 45, чтобы они не пересекались.*

Этот репозиторий содержит 3 основных файла:

1. cuda9_3.py - главный файл для нахождения всех вариантов заполнения квадрата
2. process9.py - вспомогательный файл для обработки результатов
3. plot.py - вспомогательный файл для визуализации заполнений

Используйте файл requirements.txt для установки библиотек с помощью pip (лучше будет, если вы создадите virtual environment).

Для вычисления необходима графическая карта Nvidia с установленной библиотекой CUDA.

Для быстрой проверки кода установите N = int32(8) и SIZE = int32(36) в файле cuda9_3.py. У вас должно получиться 2332 после запуска файлов cuda9_3.py и process9.py.
```
python cuda9_3.py > result9.txt
python process9.py
```

Примеры вариантов заполнения квадрата со стороной 45 находятся в файле 100examples.zip.

Подробности решения в моей [https://habr.com/ru/articles/889410/](статье на Хабре).
