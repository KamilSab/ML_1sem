import math
import json
import folium
from python_tsp.heuristics import solve_tsp_simulated_annealing
import numpy as np


# === 1. Загрузка точек из файла JSON ===
def load_points_from_file(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        raw = json.load(f)
    points = []
    names = []
    for item in raw:
        lat, lon, priority, name = item
        points.append((lat, lon, priority))
        names.append(name)
    return points, names

def calc_distance(p1, p2):
    """
    Вычисляет расстояние между двумя точками на Земле по формуле гаверсинуса.
    Принимает две точки в формате (широта, долгота).
    Возвращает расстояние в километрах.
    """
    R = 6371  # Радиус Земли в километрах

    # Преобразуем координаты из градусов в радианы
    lat1, lon1 = p1[0], p1[1]
    lat2, lon2 = p2[0], p2[1]
    dlat = math.radians(lat2 - lat1)
    dlon = math.radians(lon2 - lon1)

    # Формула гаверсинуса
    a = math.sin(dlat / 2) ** 2 + \
        math.cos(math.radians(lat1)) * math.cos(math.radians(lat2)) * math.sin(dlon / 2) ** 2

    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
    return R * c # Возвращаем расстояние в км


# === 3. Функция ввода параметров пользователем ===
def get_user_input():
    print("Выберите способ передвижения:")
    print("1 - Пешком (5 км/ч)")
    print("2 - На велосипеде (15 км/ч)")
    print("3 - На автомобиле (80 км/ч)")

    # Проверка корректности ввода способа передвижения
    while True:
        choice = input("Введите номер: ").strip()
        if choice in ['1', '2', '3']:
            break
        print("Неверный ввод. Пожалуйста, введите 1, 2 или 3.")

    # Соответствие выбора скорости
    speeds = {'1': 5, '2': 15, '3': 80}
    speed = speeds[choice]

    # Проверка корректности ввода времени
    while True:
        try:
            max_time = float(input("Введите максимальное время поездки в часах: "))
            if max_time > 0:
                break
            print("Время должно быть положительным числом.")
        except ValueError:
            print("Пожалуйста, введите число.")

    return speed, max_time


# === 4. Функция создания матрицы расстояний ===
def create_distance_matrix(points):
    """
    Создает матрицу расстояний между всеми точками маршрута.
    Принимает список точек в формате (широта, долгота, приоритет).
    Возвращает квадратную матрицу numpy, где элемент [i][j] - расстояние между i-й и j-й точкой.
    """
    n = len(points)
    distance_matrix = np.zeros((n, n))

    for i in range(n):
        for j in range(n):
            if i != j:
                distance_matrix[i][j] = calc_distance(points[i], points[j])

    return distance_matrix


# === 5. Функция поиска оптимального маршрута ===
def find_optimal_route(points, city_names, distance_matrix, speed, max_time):
    # Решаем задачу коммивояжера (нахождение приблизительно оптимального пути)
    permutation, _ = solve_tsp_simulated_annealing(distance_matrix)

    # Инициализация переменных для хранения лучшего маршрута
    best_route = []
    best_priority = 0
    best_distance = 0
    best_time = 0

    # Перебираем возможные длины маршрута от максимальной до минимальной
    for route_length in range(len(points), 1, -1):
        current_route = permutation[:route_length]  # Текущий вариант маршрута
        current_distance = 0
        current_priority = 0
        current_time = 0
        valid_route = True  # Флаг валидности маршрута

        # Расчет параметров для текущего маршрута
        for i in range(len(current_route) - 1):
            from_idx = current_route[i]
            to_idx = current_route[i + 1]
            dist = distance_matrix[from_idx][to_idx]  # Расстояние между точками
            time = dist / speed  # Время перемещения

            # Проверка на превышение лимита времени
            if current_time + time > max_time:
                valid_route = False
                break

            # Обновление параметров маршрута
            current_distance += dist
            current_time += time
            current_priority += points[from_idx][2]  # Учет приоритета точки

        # Добавление последней точки маршрута
        if valid_route and len(current_route) > 0:
            last_idx = current_route[-1]
            current_priority += points[last_idx][2]  # Приоритет последней точки

            # Проверка, является ли текущий маршрут лучшим
            if current_priority > best_priority:
                best_priority = current_priority
                best_route = current_route
                best_distance = current_distance
                best_time = current_time

    return best_route, best_distance, best_time, best_priority


# === 6. Основная функция программы ===
def main():
    SPEED_KMH, MAX_TIME = get_user_input()

    # Загружаем данные о точках маршрута
    try:
        points_list, city_names = load_points_from_file('trip.json')
    except FileNotFoundError:
        print("Файл 'trip.json' не найден. Используем тестовые данные.")
        points_list = [
            (55.7558, 37.6176, 1),  # Москва
            (59.9343, 30.3351, 8),  # Санкт-Петербург
            (47.2355, 39.7078, 6),  # Ростов-на-Дону
            (54.9803, 73.3757, 5),  # Новосибирск
            (56.8389, 60.6057, 7),  # Екатеринбург
            (55.1599, 61.4029, 10)  # Челябинск
        ]
        city_names = [
            "Москва", "Санкт-Петербург", "Ростов-на-Дону",
            "Новосибирск", "Екатеринбург", "Челябинск"
        ]

    # Строим матрицу расстояний между всеми точками
    distance_matrix = create_distance_matrix(points_list)

    # Находим оптимальный маршрут
    best_route, best_distance, best_time, best_priority = find_optimal_route(
        points_list, city_names, distance_matrix, SPEED_KMH, MAX_TIME
    )

    # Вывод результатов
    if best_route:
        # Преобразование времени в часы, минуты, секунды
        total_seconds = int(best_time * 3600)
        hours, remainder = divmod(total_seconds, 3600)
        minutes, seconds = divmod(remainder, 60)

        # Вывод информации о маршруте
        print("\n✅ Оптимальный маршрут:")
        for idx in best_route:
            print(f"- {city_names[idx]} (Приоритет: {points_list[idx][2]})")
        print(f"\nОбщая длина маршрута: {best_distance:.2f} км")
        print(f"Время на дорогу: {hours} ч {minutes} мин {seconds} сек")
        print(f"Суммарный приоритет: {best_priority}")

        # Создание карты с маршрутом
        map_center = [points_list[best_route[0]][0], points_list[best_route[0]][1]]
        my_map = folium.Map(location=map_center, zoom_start=4)

        # Добавление маркеров для всех точек
        for idx, point in enumerate(points_list):
            color = 'red' if idx in best_route else 'blue'  # Цвет маркера
            name = city_names[idx]
            folium.Marker(
                location=[point[0], point[1]],
                popup=f"{name}\nПриоритет: {point[2]}",
                icon=folium.Icon(color=color, icon='info-sign')
            ).add_to(my_map)

        # Добавление линии маршрута
        path_coords = [[points_list[i][0], points_list[i][1]] for i in best_route]
        folium.PolyLine(path_coords, color="green", weight=2.5, opacity=1).add_to(my_map)

        # Сохранение карты в HTML-файл
        my_map.save('optimized_route.html')
        print("\n🗺 Карта сохранена как 'optimized_route.html'")
    else:
        print("❌ Не удалось найти подходящий маршрут.")


# Точка входа в программу
if __name__ == "__main__":
    main()