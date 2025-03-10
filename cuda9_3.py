#!/usr/bin/env python3

import os
import sys
import time
import math
from random import randint

import numpy as np
try:
    from numba import cuda
except Exception as exc:
    print("numba cuda not found")
    class cuda:
        jit = lambda f: f
            
from numpy import uint8, uint32, uint64, int32, int64
from tqdm import tqdm

"""
Перебор вариантов заполнения квадрата 45x45 квадратиками:
1x1 - 1 шт.
2x2 - 2 шт.
...
9x9 - 9 шт.

Автор - Данила Потапов

Варианты выводятся на экран, поэтому запускать программу нужно с сохранением результатов в файл
python cuda9_3.py > result9.txt

"""

#N = int32(8)
#SIZE = int32(36)  # сторона большого квадрата

N = int32(9)
SIZE = int32(45)  # сторона большого квадрата

INIT_BATCH = 6  # число квадратиков, которые предзаполняются в подзадаче на первом этапе вычислений

LIMIT = int64(1000000)  # максимальное количество итераций в подзадаче.   int64(-1) - no limit

# |1|22|22|333|333|333|4444|4444|4444|4444|5555 - максимально возможное число промежутков <= 11

@cuda.jit
def cuda_run_core(A, result, batch):
    """
    ядро (kernel) - функция, вычисляемая на GPU в каждом потоке

    A - набор подзадач в виде массива [M x (batch+1)], где каждая строка задаёт сначала количество
        квадратиков, а потом идут размеры квадратиков в порядке заполнения сверху вниз, слева направо
    
    result - какое-то возвращаемое значение (в данном случае количество уже не имеет значение,
        т.к. подсчёт делается в функции process9, которая учитывает варианты, выведенные на экран)

    batch - максимальное количество квадратиков в спецификации подзадачи
    """

    cnt = int64(0)  # количество вариантов (устарело)

    pos = cuda.grid(1)  # индекс подзадачи
    if pos >= A.size // (batch+1):  # pos != 0
        return

    #if pos % 10000 == 0:
    #    print(pos)
    
    nxt = cuda.local.array(45*12, int32)  # SIZE*12   # массив концов промежутков для каждого слоя
    prv = cuda.local.array(45*12, int32)  # SIZE*12   # массив начал промежутков для каждого слоя

    for i in range(SIZE*12):
        nxt[i] = -1
        prv[i] = -1
    nxt[0] = SIZE
    prv[0] = 0

    variants = cuda.local.array(10, int32)  # N+1   # количество квадратиков размера i
    for i in range(1, N+1):
        variants[i] = i
    
    iter = int32(0)  # глубина рекурсии
    y = int32(0)  # текущий слой

    # Стеки для сохранения локальных переменных
    ss = cuda.local.array(45, int32)  # [iter] SIZE  # размеры квадратиков
    yy = cuda.local.array(45, int32)  # [iter] SIZE  # индексы слоёв
    ww = cuda.local.array(45, int32)  # [iter] SIZE  # ширины промежутков
    xx0 = cuda.local.array(45, int32)  # [iter] SIZE  # начала выбранных промежутков
    xx1 = cuda.local.array(45, int32)  # [iter] SIZE  # концы выбранных промежутков

    prv_old = cuda.local.array(45, int32)  # [iter] SIZE  # !!! начало предыдущего промежутка при слиянии или -1
    nxt_old = cuda.local.array(45, int32)  # [iter] SIZE  # !!! конец следующиего промежутка при слиянии или -1

    for i in range(SIZE):
        ss[i] = 0
    s = int32(0)  # текущий размер квадратика

    step = int64(0)  # количество итераций
    while iter >= 0:  
        step += 1
        if step == LIMIT:
            cnt = -1
            break

        if s >= N:
            # Откат на предыдущий уровень рекурсии
            if s == N:
                variants[s] += 1

            iter -= 1

            if iter < 0:
                break

            y = yy[iter]
            s = ss[iter]
            w = ww[iter]
            x0 = xx0[iter]
            x1 = xx1[iter]
            if iter < SIZE - 1:
                ss[iter+1] = 0

            # Восстановление множества промежутков ####################
            if s < w:
                # Случай, когда квадратик помещался с запасом

                prv[y*12] = x0
                nxt[y*12] = x1

                if y + s < SIZE:
                    if prv_old[iter] >= 0:
                        for i in range((y+s)*12, (y+s+1)*12):
                            if nxt[i] == x0+s:
                                nxt[i] = x0
                                break
                    else:
                        for i in range((y+s)*12, (y+s+1)*12):
                            if prv[i] >= x0:
                                prv[i] = prv[i+1]
                                nxt[i] = nxt[i+1]
                            if nxt[i] == -1:
                                break
                    
            elif s == w:
                # Случай, когда квадратик помещался "впритык"

                prev_nxt = int32(-2)
                prev_prv = int32(-2)
                for i in range(y*12, (y+1)*12):
                    if nxt[i] == -1 or nxt[i] > x0:
                        if prev_nxt != -2:
                            z = prev_nxt
                            prev_nxt = nxt[i]
                            nxt[i] = z
                        else:
                            prev_nxt = nxt[i]
                            nxt[i] = x1
                        if prev_prv != -2:
                            z = prev_prv
                            prev_prv = prv[i]
                            prv[i] = z
                        else:
                            prev_prv = prv[i]
                            prv[i] = x0
                    if nxt[i] == -1:
                        break

                if y + s < SIZE:
                    if prv_old[iter] >= 0:
                        if nxt_old[iter] >= 0:
                        
                            prev_nxt = int32(-2)
                            prev_prv = int32(-2)
                            for i in range((y+s)*12, (y+s+1)*12):
                                if nxt[i] == nxt_old[iter]:
                                    nxt[i] = x0
                                
                                if nxt[i] == -1 or nxt[i] > x0:
                                    if prev_nxt != -2:
                                        z = prev_nxt
                                        prev_nxt = nxt[i]
                                        nxt[i] = z
                                    else:
                                        prev_nxt = nxt[i]
                                        nxt[i] = nxt_old[iter]
                                    if prev_prv != -2:
                                        z = prev_prv
                                        prev_prv = prv[i]
                                        prv[i] = z
                                    else:
                                        prev_prv = prv[i]
                                        prv[i] = x0 + s
                                if nxt[i] == -1:
                                    break                        
                        else:
                            for i in range((y+s)*12, (y+s+1)*12):
                                if prv[i] == prv_old[iter]:
                                    nxt[i] = x0
                                    break
                    elif nxt_old[iter] >= 0:
                        for i in range((y+s)*12, (y+s+1)*12):
                            if nxt[i] == nxt_old[iter]:
                                prv[i] = x0 + s
                                break

                    else:
                        for i in range((y+s)*12, (y+s+1)*12):
                            if nxt[i] > x0:
                                nxt[i] = nxt[i+1]
                                prv[i] = prv[i+1]
                            if nxt[i] == -1:
                                break
    
        # Шаг вперёд #####################

        yy[iter] = y

        if s > 0:
            variants[s] += 1
        s += 1
        ss[iter] = s
        #print(iter, s)
        
        x0 = prv[12*y]
        #print("x0", x0)
        xx0[iter] = x0

        x1 = nxt[12*y]
        #print("x1", x1)
        xx1[iter] = x1

        w = x1 - x0
        ww[iter] = w


        if s == N+1:
            continue

        # Проверка на соответствие первых квадратиков спецификации подзадачи
        if iter < A[(batch+1)*pos] and s != A[(batch+1)*pos + iter + 1]:
            variants[s] -= 1
            continue

        if y + s > SIZE:
            s = N+1
            continue

        if s > w:
            s = N+1
            continue            

        #  Квадратик размера 1 не может стоять в крайней строке и столбце
        if s == 1 and (y == 0 or x0 == 0 or x0 == SIZE-1):
            variants[s] -= 1
            continue

        if variants[s] == 0:
            variants[s] -= 1
            continue

        if s < w:
            # Случай, когда квадратик помещается с запасом

            variants[s] -= 1
            prv[y*12] = x0 + s
            
            if y + s < SIZE:
                prv_old[iter] = int32(-1)
                ii = int32(-1)
                for i in range((y+s)*12, (y+s+1)*12):
                    if nxt[i] == -1:
                        break
                    if nxt[i] == x0:
                        ii = i
                        break
                if ii != -1:
                    prv_old[iter] = prv[ii]
                    nxt[ii] = x0+s
                else:
                    prev_nxt = int32(-2)
                    prev_prv = int32(-2)
                    for i in range((y+s)*12, (y+s+1)*12):
                        if nxt[i] == -1 or nxt[i] > x0:
                            if prev_nxt != -2:
                                z = prev_nxt
                                prev_nxt = nxt[i]
                                nxt[i] = z
                            else:
                                prev_nxt = nxt[i]
                                nxt[i] = x0 + s
                            if prev_prv != -2:
                                z = prev_prv
                                prev_prv = prv[i]
                                prv[i] = z
                            else:
                                prev_prv = prv[i]
                                prv[i] = x0
                        if nxt[i] == -1:
                            break

        elif s == w:
            # Случай, когда квадратик помещается "впритык"

            variants[s] -= 1

            for i in range(y*12, (y+1)*12):
                nxt[i] = nxt[i+1]
                prv[i] = prv[i+1]
                if nxt[i] == -1:
                    break

            if y + s < SIZE:
                prv_old[iter] = -1
                nxt_old[iter] = -1

                ii = int32(-1)
                for i in range((y+s)*12, (y+s+1)*12):
                    if nxt[i] == -1:
                        break
                    if nxt[i] == x0:
                        ii = i
                        break
                if ii != -1: # gap before
                    if prv[ii+1] == x0 + s: # gap after
                        prv_old[iter] = prv[ii]
                        nxt_old[iter] = nxt[ii+1]
                        for i in range(ii+1, (y+s+1)*12):
                            nxt[i] = nxt[i+1]
                            prv[i] = prv[i+1]
                            if nxt[i] == -1:
                                break
                        nxt[ii] = nxt_old[iter]
                    else: # only before
                        prv_old[iter] = prv[ii]
                        nxt[ii] = x0 + s
                
                else:  # no gap before
                    ii = int32(-1)
                    for i in range((y+s)*12, (y+s+1)*12):
                        if prv[i] == x0 + s:
                            ii = i
                            break
                        if nxt[i] == -1:
                            break
                    if ii != -1: # gap after
                        nxt_old[iter] = nxt[ii]
                        prv[ii] = x0
                    else: # no gaps before and after
                        prev_nxt = int32(-2)
                        prev_prv = int32(-2)
                        for i in range((y+s)*12, (y+s+1)*12):
                            if nxt[i] == -1 or nxt[i] > x0:
                                if prev_nxt != -2:
                                    z = prev_nxt
                                    prev_nxt = nxt[i]
                                    nxt[i] = z
                                else:
                                    prev_nxt = nxt[i]
                                    nxt[i] = x0 + s
                                if prev_prv != -2:
                                    z = prev_prv
                                    prev_prv = prv[i]
                                    prv[i] = z
                                else:
                                    prev_prv = prv[i]
                                    prv[i] = x0
                            if nxt[i] == -1:
                                break
            
                        
        if iter == SIZE-1:
            # Вывод на печать найденного заполнения
            print(ss[0]*1000000 + ss[1]*10000 + ss[2]*100 + ss[3],
                ss[4]*1000000 + ss[5]*10000 + ss[6]*100 + ss[7],
                ss[8]*1000000 + ss[9]*10000 + ss[10]*100 + ss[11],
                ss[12]*1000000 + ss[13]*10000 + ss[14]*100 + ss[15],
                ss[16]*1000000 + ss[17]*10000 + ss[18]*100 + ss[19],
                ss[20]*1000000 + ss[21]*10000 + ss[22]*100 + ss[23],
                ss[24]*1000000 + ss[25]*10000 + ss[26]*100 + ss[27],
                ss[28]*1000000 + ss[29]*10000 + ss[30]*100 + ss[31],
                ss[32]*1000000 + ss[33]*10000 + ss[34]*100 + ss[35],
                ss[36]*1000000 + ss[37]*10000 + ss[38]*100 + ss[39],
                ss[40]*1000000 + ss[41]*10000 + ss[42]*100 + ss[43],
                ss[44]*1000000)
            cnt += 1
            iter += 1
            s = N+1
        else:
            # Поиск следующего незаполненного промежутка
            iter += 1
            s = ss[iter]
            for i in range(y, SIZE):
                if nxt[i*12] != -1:
                    y = i
                    break
    
    result[pos] = cnt  # устарело

# Генерация подзадач
A = []  # список подзадач
path = []  # текущая генерируемая подзадача
def f(iter, variants, batch):
    """ Рекурсивная функция для генерации подзадач
    iter - глубина рекурсии или же размер уже сгенерированной подзадачи
    variants - количество ранее использованных квадратиков каждого размера
    batch - целевой размер подзадачи
    """
    global path

    if iter == batch:
        A.append(tuple(path))
        return

    for i in range(1, N+1):
        if variants[i] > 0:
            variants[i] -= 1
            path.append(i)

            f(iter+1, variants, batch)
        
            variants[i] += 1
            path = path[:-1]


def decode(a):
    """ Декодирование координат и размеров квадратиков из кодировки a
    a - массив размеров квадратиков в порядке заполнения сверху вниз, слева направо
    
    Возвращает:
    boxes - Массив квадратиков [(y1, x1, s1), (y2, x2, s2), ... ]
    """
    
    M = [[0]*SIZE for i in range(SIZE)]  # карта заполнения большого квадрата
    y, x  = 0, 0
    boxes = []
    # invariant: y, x - next free place
    for s in a:
        if x + s > SIZE:
            return []
        elif y + s > SIZE:
            return []
        else:
            for j in range(x, x+s):
                if M[y][j] != 0:
                    return []
            for i in range(y, y+s):
                for j in range(x, x+s):
                    M[i][j] = s
            boxes.append((y, x, s))
            x += s
            for i in range(y, SIZE):
                for j in range(x, SIZE):
                    if M[i][j] == 0:
                        break
                if x < SIZE and M[i][j] == 0:
                    break
                x = 0
                if i == SIZE-1 and j == SIZE-1:
                    i, j = SIZE, SIZE
            y, x = i, j
    return boxes

def flip(boxes):
    """ flip horizontally - Отражение квадратиков относительно вертикальной оси """
    new_boxes = []
    for y, x, s in boxes:
        new_boxes.append((y, SIZE-x-s, s))
    return sorted(new_boxes)

def rotate90(boxes):
    """ rotate90 CCW - Поворот квадратиков против часовой стрелки """
    new_boxes = []
    for y, x, s in boxes:
        new_boxes.append((SIZE-x-s, y, s))
    return sorted(new_boxes)
    
def encode(boxes):
    """ Кодирование квадратиков в Bouwkampcode (tablecode) """
    M = [[0]*SIZE for i in range(SIZE)]  # карта заполнения большого квадрата
    a = []
    prev_y = 0
    for y, x, s in boxes:
        if y != prev_y:
            found_hole = False
            for i in range(prev_y, y):
                for j in range(SIZE):
                    if M[i][j] == 0:
                        found_hole = True  # potentially can happen in rotate90
                        break
                if found_hole:
                    break
            if found_hole:
                break
        
        found_hole = False
        for j in range(x):
            if M[y][j] == 0:
                found_hole = True
                break
        if found_hole:
            break

        prev_y = y

        for i in range(y, y+s):
            for j in range(x, x+s):
                M[i][j] = s
        a.append(s)

    return a

def is_less(a, b):
    """ Сравнение двух кодировок в лексикографическом порядке.
    Стандартное сравнение Питона не устраивает, т.к. [1,2] < [1,2,3], но в этом случае нам нужно
    подождать следующего квадратика, чтобы узнать, какой из вариантов меньше. """
    for i in range(min(len(a), len(b))):
        if a[i] < b[i]:
            return True
        elif a[i] > b[i]:
            return False
    return False


def main():
    global A, path

    A_new = [(i,) for i in range(2, N+1)]  # список актуальных подзадач
    prev_batch = 1
    batch = INIT_BATCH  # целевой размер подзадачи

    result = int64(0)  # устарело
    
    stage = 0  # номер этапа
    while True:
        stage += 1
        print("stage", stage, "cur. result", result, file=sys.stderr)

        A = []
        for x in A_new:
            variants = list(range(0, N+1))
            path = list(x)
            for y in path:
                variants[y] -= 1
            f(prev_batch, variants, batch)

        A_filtered = []
        for a in A:
            # Проверка, что 1 не стоит в первой строке
            if 1 in a:
                i = a.index(1)
                if sum(a[:i]) < SIZE:
                    continue

            # Проверка подзадачи на каноничность
            if sum(a) >= SIZE:
                boxes = decode(a)
                if boxes == []:
                    continue
                a1 = encode(flip(boxes))
                if is_less(a1, a):
                    continue
                a2 = encode(rotate90(boxes))
                if is_less(a2, a):
                    continue
                a3 = encode(rotate90(flip(boxes)))  # i.e. transpose
                if is_less(a3, a):
                    continue
            A_filtered.append(a)

        # Добавляем размер подзадач в начало каждой строки
        A_orig = [(batch,) + a for a in A_filtered]
        A = A_orig[:]

        num = len(A)
        print("combinations", num, file=sys.stderr)

        if num > 10**9:
            print("Error: too low LIMIT")
            return

        # Подготовка к запуску на GPU
        A = np.ravel(np.array(A, dtype=int32))
        B = np.zeros(num, dtype=int64)
        threadsperblock = 2**9
        blockspergrid = max(1, (num + (threadsperblock - 1)) // threadsperblock)
        A, B = cuda.to_device(A), cuda.to_device(B)
        cuda_run_core[blockspergrid, threadsperblock](A, B, batch)
        A, B = A.copy_to_host(), B.copy_to_host()
    
        # Синхронизация, чтобы вся печать завершилась
        cuda.synchronize()  # ensure everything is printed
        
        # Отбор актуальных подзадач
        A_new = []
        for i, x in enumerate(B):
            if x == -1:
                A_new.append(A_orig[i][1:])
            else:
                result += x

        if len(A_new) == 0:
            break

        prev_batch = batch
        batch += 1

        
    print("final result (deprecated)", result)  # устарело  
    

if __name__ == '__main__':
    main()
