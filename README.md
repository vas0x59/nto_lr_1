# Репозиторий команды CPC

## Состав команды:
#### Никита Локтев - лидер команды
#### Василий Юрьев - программист
#### Артемий Одышев - программист
#### Лев Синюков - инженер - техник

### Запуск:
```bash
./main.py
```


### Файлы программы:
    * main.py (основной файл, точка входа)
    * walls.py и walls_alg.py (для определения стен)
    * fires.py (для нахождения пожаров и пострадавших)  
    * utils.py
### Необходимые файлы для запуска:
    * legs_mask.png (маска элементов конструкции коптера)
    

## Используемые методы решения:
### Взлет и запись координат старта:
#### Для определения координат стартовой позиции при взлете записывалась телеметрия относительно aruco_map
### Распознавание стен, возгораний и пострадавших:
#### Для распознавания использовалась библиотека технического зрения OpenCV
# Ход работы
## Программная часть
### Программная часть:
#### День 1 (Работа над 1 – ым заданием):
https://github.com/vas0x59/nto_lr_1/tree/04de252c36aae7a5ef017b4eb266a80871742ff0
1.	Взлет с точки старта
2.	Движение к точке страта мониторинга
3.	Полет по периметру помещения
4.	Обнаружение очагов возгораний и вывод их координат в терминал
5.	Посадка в точке старта
6.	Тренировка полетов
### День 2 (Работа на 1- ым заданием):
https://github.com/vas0x59/nto_lr_1/tree/7a3fda14e6d3a8ff5ef2894c01f103911e1a07e1
1.	Обнаружение стен и определение их контуров
2.	Визуализация в rviz всех стен помещения
3.	Визуализация в rviz очагов возгораний красными квадратами
4.	Тренировка полетов
5.	Зачетная попытка, сдача решения первого задания
### День 3 (работа над 2 – ым заданием):
https://github.com/vas0x59/nto_lr_1/tree/2076af1286e124495de0df4966b13127e75d6973
1.	Расчет длин стен помещения
2.	Обнаружение пострадавших 
3.	Определение координат местоположения пострадавших
4.	Визуализация в rviz местоположение пострадавших синими квадратами
5.	Зачетная попытка, сдача решения второго задания
### День 4 (работа над 3 – ым заданием):
https://github.com/vas0x59/nto_lr_1/tree/cc21dc51ef959846604056c37140f60939edd4a8
1.	Следование по протяженности стен по всему помещению
2.	Сброс капсул пожаротушения на очаги возгорания
3.	Возврат в стартовую позицию обратным путем
4.	Тренировка полетов
5.	Зачетная попытка, сдача третьего задания

## Инженерная часть:
### День № 1-2 (14-15.03.23):
1. Разработка идеи 
2. Реализация 3D модели в собранном и разобранном виде
1. Доработка 3D модели (облегчение, сверка с размерами коптера) 
2. Печать 3D модели на 3D принтере (2 детали)
3. Создание сборочного чертежа
4. Создание спецификации
5. Написание инструкции по сборке, обслуживанию и эксплуатации устройства
### День № 3 (16.03.23):
1. Печать 3D модели на 3D принтере (2 детали)
2. Сборка устройства и последующая доработка 
3. Написание кода для ардуино
4. Отладка кода для корректной работы устройства
### День № 4 (17.03.23):
1. Отладка работы устройства на коптере
2. Проверка работы устройства на коптере

# Полезная нагрузка
https://drive.google.com/drive/folders/13_4vVUnz3LR-pDygIHaopILK8Pyg_QzK

# Скриншоты моделей в САПРе

![image](https://github.com/vas0x59/nto_lr_2/blob/main/photo1679040179%20(1).jpeg)
![image](https://github.com/vas0x59/nto_lr_2/blob/main/photo1679040179.jpeg)

# Код ардуино для устройства

```c++
#include <Servo.h>

const int P1 = 35;
const int P0 = 90;
const int P2 = 155;

boolean b1 = false;
boolean b2 = false;

boolean b1_prev = false;
boolean b2_prev = false;

Servo servo;

void setup() {
  servo.attach(9);
  servo.write(P0);
  pinMode(11, INPUT);
  pinMode(12, INPUT);
}

void loop() {
  b1 = digitalRead(11);
  b2 = digitalRead(12);

  if (b1 == 1 and b1_prev == 0)
  {
    servo.write(P1);
    delay(1000);
    servo.write(P0);
  } else if (b2 == 1 and b2_prev == 0)
  {
    servo.write(P2);
    delay(1000);
    servo.write(P0);
  }
  b1_prev = digitalRead(11);
  b2_prev = digitalRead(12);
}
```

# Фото команды 
![image](https://github.com/vas0x59/nto_lr_1/blob/main/msg632116981-76186.jpg)
