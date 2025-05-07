import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from catboost import CatBoostRegressor


def physics_model(v0, angle, C_d, mass, A=0.01, dt=0.01):
  '''
  Функция physics_model получает значения v0(начальной скорости), 
  angle(угла наклона), C_d(коэф сопротивления), mass(массы).
  С помощью уравнений Эйлера находит траекторию полета объекта.
  Возвращает массив trajectory с координатами x,y.
  '''
  import numpy as np
  angle_rad = np.radians(angle)
  g = 9,80665
  rho = 1.225
  vx = v0 * np.cos(angle_rad)
  vy = v0 * np.sin(angle_rad)
  x, y = 0, 0
  trajectory = []
  while y >= 0:
      v = np.sqrt(vx**2 + vy**2)
      F_drag = 0.5 * rho * C_d * A * v**2
      ax = -(F_drag / mass) * (vx / v)
      ay = -g - (F_drag / mass) * (vy / v)
      vx += ax * dt
      vy += ay * dt
      x += vx * dt
      y += vy * dt
      trajectory.append([x, y])
  return np.array(trajectory)



def hybrid_predict(v0, angle, mass, C_d, wind):
  """
  Функция получает значения v0(начальной скорости), 
  angle(угла наклона), C_d(коэф сопротивления), mass(массы), 
  wind(ветра).
  Находит ошибку в дальности полета согласно модели и корректирует
  Возвращает скорректированную дальность полета x_hybrid,
  траекторию физ. модели physics_traj
  """
  physics_traj = physics_model(v0, angle, C_d, mass)
  x_physics = physics_traj[-1, 0]
  
  input_features = [[v0, angle, mass, C_d, wind, x_physics]]
  error = model.predict(input_features)[0]

  x_hybrid = x_physics + error
  return x_hybrid, physics_traj



df = pd.read_csv('ballistic_dataset.csv')
df['error'] = df['x_real'] - df['x_physics']
X = df[['v0', 'angle', 'C_d', 'mass', 'wind', 'x_physics']]
Y = df['error']

model = CatBoostRegressor(iterations=500, learning_rate=0.05, verbose=100)
model.fit(X, Y)

x_hybrid, physics_traj = hybrid_predict(v0=5, angle=45, mass=1.0, C_d=0.2, wind=1.0)

plt.plot(physics_traj[:, 0], physics_traj[:, 1]) 
plt.scatter(x_hybrid, 0) 
plt.show()