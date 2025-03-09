import os, json
from tqdm import tqdm
from cuda9_3_release import N, SIZE, rotate90, flip, decode, encode, is_less

"""
Вспомогательный скрипт для агрегации и подсчёта результатов
"""

# Файл, в который был сохранён вывод основного скрипта
DUMP_PATH = "result9.txt"  # "cuda9_3.txt"

json_path = '2025_all.json'
if os.path.exists(json_path):
    if os.path.exists(DUMP_PATH):
        print("Warning: Loading from JSON, not using", DUMP_PATH)
    with open(json_path, 'r') as f:
        all_combinations = json.load(f)
        set_combinations = set(map(tuple, all_combinations))
else:
    print("Loading from", DUMP_PATH)
    with open(DUMP_PATH) as f:
        lines = f.readlines()
        lines = [x for x in lines if x[0] in '0123456789']
        print("Total lines:", len(lines))
        
    set_all_combinations = set([])
    for line in lines:
        # Парсим кодировку заполнения
        a = []
        for x in map(int, line.rstrip().split()):
            a.append(x // 1000000)
            if len(a) == SIZE:
                break
            a.append(x // 10000 % 100)
            if len(a) == SIZE:
                break
            a.append(x // 100 % 100)
            if len(a) == SIZE:
                break
            a.append(x % 100)
            if len(a) == SIZE:
                break
        set_all_combinations.add(tuple(a))

    print("Intermediate count:", len(set_all_combinations))  # not equal to final_result because some interm. combinations may be non-canonic 

    # Верификация заполнений и фильтрация неканонических
    set_combinations = set([])
    for a in tqdm(set_all_combinations):    
        assert(sum([s**2 for s in a]) == SIZE*SIZE), "Сумма площадей квадратиков не равна площади большого квадрата"
        cnt = [0]*(N+1)
        for s in a:
            cnt[s] += 1
        assert(cnt == list(range(N+1))), "Количества квадратиков не соответствуют условию"
        e = encode
        d = decode
        f = flip
        r = rotate90
        da = d(a)
        fda = f(d(a))
        rda = r(d(a))
        # Проверка на "каноничность"
        if is_less(e(fda), a) or is_less(e(rda), a) or is_less(e(r(fda)), a) or is_less(e(r(rda)), a) or \
            is_less(e(r(r(fda))), a) or is_less(e(r(r(rda))), a) or is_less(e(r(r(r(fda)))), a):
            continue
        set_combinations.add(tuple(a))

    # Выводим финальный результат
    print(len(set_combinations))

    # Сохранение финального результата
    with open(json_path, 'wb') as fw:
        all_combinations = sorted(set_combinations)
        lines = ["["]
        for a in all_combinations:
            lines.append(str(list(a)).replace(" ", "") + ",")
        lines[-1] = lines[-1][:-1]
        lines.append("]")        
        
        with open(json_path, 'w') as fw:
            for x in lines:
                fw.write(x + "\n")
        

print(len(set_combinations))

"""
Дополнительная проверка того, что заполнения, найденные скриптом на С++ были найдены

def load_combinations(file):
    with open(file) as f:
        lines = f.readlines()    

    i = 0
    set_combinations = set([])
    while True:
        while i < len(lines) and ')' not in lines[i]:
            i += 1
        if i == len(lines):
            break

        i += 1
        a = []
        for y in range(i, i+SIZE):
            ln = list(map(int, lines[y].rstrip().split()))
            for x in range(SIZE):
                if ln[x] != 0:
                    a.append(ln[x])
        i += SIZE

        assert(sum([s**2 for s in a]) == SIZE*SIZE), breakpoint()
        cnt = [0]*(N+1)
        for s in a:
            cnt[s] += 1
        assert(cnt == list(range(N+1)))
        e = encode
        d = decode
        f = flip
        r = rotate90
        if is_less(e(f(d(a))), a) or is_less(e(r(d(a))), a) or is_less(e(r(f(d(a)))), a) or is_less(e(r(r(d(a)))), a) or is_less(e(r(r(f(d(a))))), a) or \
            is_less(e(r(r(r(d(a))))), a) or is_less(e(r(r(r(f(d(a)))))), a):
            continue
        #result += 1
        set_combinations.add(tuple(a))
    return sorted(set_combinations)

for file in ['2025_2_12.txt', '2025_2_13.txt', '2025_2_14.txt', '2025_2_15.txt', '2025_2_16.txt']:
    print(file)
    cur_combinations = load_combinations(file)
    combinations_in_range = []
    for c in set_combinations:
        if not is_less(c, cur_combinations[0]) and not is_less(cur_combinations[-1], c):
            combinations_in_range.append(c)
    print(len(cur_combinations))
    assert(sorted(combinations_in_range) == cur_combinations), breakpoint()

"""