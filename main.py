import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor


def physics_model(v0, angle, C_d, mass, A=0.01, dt=0.01):
  '''
  physics_model(v0[m/s], angle[degree], C_d, mass[kg], A=0.01, dt=0.01[s])
  Функция physics_model получает значения v0(начальной скорости), 
  angle(угла наклона), C_d(коэф сопротивления), mass(массы).
  С помощью уравнений Эйлера находит траекторию полета объекта.
  Возвращает массив trajectory с координатами x,y.
  '''
  import numpy as np
  angle_rad = np.radians(angle)
  g = 9.80665 #Ускорение свободного падения [m/s**2]
  rho = 1.225 #Плотность воздуха [kg/m**3]
  vx = v0 * np.cos(angle_rad) #Проекция скорости тела на ось X
  vy = v0 * np.sin(angle_rad) #Проекция скорости тела на ось Y
  x, y = 0, 0 #Координата тела по оси X и Y
  trajectory = [[0,0]]
  while y >= 0:
      v = np.sqrt(vx**2 + vy**2) #Общий вектор скорости 
      F_drag = 0.5 * rho * C_d * A * v**2 #Сила лобового сопротивления
      ax = -(F_drag / mass) * (vx / v) #Ускорение по оси X
      ay = -g - (F_drag / mass) * (vy / v) #Ускорение по оси Y
      #Изменение скорости за промежуток dt 
      vx += ax * dt 
      vy += ay * dt
      #Изменение координаты за промежуток dt
      if (vy * dt + y) >= 0: 
        x += vx * dt 
        y += vy * dt
      else:
        y = 0
        break
      trajectory.append([x, y])
  return np.array(trajectory)



def hybrid_predict(v0, angle, mass, C_d, wind):
  """
  hybrid_predict(v0[m/s], angle[degree], mass[kg], C_d, wind[m/s])
  Функция получает значения v0(начальной скорости), 
  angle(угла наклона), C_d(коэф сопротивления), mass(массы), 
  wind(ветра).
  Находит ошибку в дальности полета согласно модели и корректирует
  Возвращает скорректированную дальность полета x_hybrid,
  траекторию физ. модели physics_traj
  """
  physics_traj = physics_model(v0, angle, C_d, mass)
  x_physics = physics_traj[-1, 0] #Конечная точка полета
  
  input_features = [[v0, angle, mass, C_d, wind, x_physics]]
  error = model.predict(input_features)[0]

  x_hybrid = x_physics + error #Создание гибридной конечной точки
  return x_hybrid, physics_traj



#Обрабатывание датасета
df = pd.read_csv('ballistic_dataset.csv')
df['error'] = df['x_real'] - df['x_physics']
X = df[['v0', 'angle', 'C_d', 'mass', 'wind', 'x_physics']] 
Y = df['error']

#Обучение модели
model = CatBoostRegressor(iterations=500, learning_rate=0.05, verbose=100)
model.fit(X, Y)

#Получение искомых значений
x_hybrid, physics_traj = hybrid_predict \
 (v0=5, angle=45, mass=10.0, C_d=0.2, wind=1.0) 

#Создание графика полета
plt.plot(physics_traj[:, 0], physics_traj[:, 1], label='Траектория') 
plt.scatter(x_hybrid, 0, label='Предсказанное значение')
plt.grid()
plt.legend()
plt.show()
