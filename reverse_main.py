from PIL import Image
import pygame as pg
import numpy as np
import os

np.random.seed(123)
pg.font.init()


def array_to_sym(array):
  array = list(array)
  for i in range(len(array)):
    if array[i] >= 0.4:
        array[i] = 'Ж'
    else:
        array[i] = '.'

  for i in range(24):
    print(''.join(array[i*24:(i+1)*24]))
    
  return array


def sigmoid(x):
  # Сигмоидная функция активации: f(x) = 1 / (1 + e^(-x))
  return 1 / (1 + np.exp(-x))


def deriv_sigmoid(x):
  # Производная сигмоиды: f'(x) = f(x) * (1 - f(x))
  fx = sigmoid(x)
  return fx * (1 - fx)


class NeuralNetwork:
  def __init__(self):
    # случайно инициализируем веса, в среднем - 0
    self.syn0 = np.random.random((10, 182)) * 2 - 1
    self.syn1 = np.random.random((182, 364)) * 2 - 1
    self.syn2 = np.random.random((364, 576)) * 2 - 1


  def prediction(self, data):
    # Предсказание

    if type(data) == int:
      data = number_to_array(data)

    l1 = sigmoid(np.dot(data, self.syn0))
    l2 = sigmoid(np.dot(l1, self.syn1))
    l3 = sigmoid(np.dot(l2, self.syn2))

    return l3


  def compare(self, mas1, mas2):
    count = 0
    for i in range(len(mas1)):
      if mas1[i] == mas2[i]:
        count += 1
    return count/len(mas1) > 0.5


  def accuracy(self, train_X, train_Y):
    # Подсчёт точности

    count = 0
    for data, answer in zip(train_X, train_Y):
      if self.compare(self.prediction(data), answer):
        count += 1
    
    return round(count/len(train_Y) * 100, 2)


  def train(self, train_X, train_Y, epochs = 1000):
    # Обучение нейросети

    l0 = train_X
    for epoch in range(epochs):

      # Проходим вперёд по слоям 0, 1 и 2
      l1 = sigmoid(np.dot(l0, self.syn0))
      l2 = sigmoid(np.dot(l1, self.syn1))
      l3 = sigmoid(np.dot(l2, self.syn2))

      if (epoch % 200) == 0:
        print(f"Точность: {self.accuracy(train_X, train_Y)}%")
        array_to_sym(l3[0])
          
      l3_error = train_Y - l3
      l3_delta = 0.00001 * l3_error * deriv_sigmoid(l3)
      
      l2_error = l3_delta.dot(self.syn2.T)
      l2_delta = 0.00001 * l2_error * deriv_sigmoid(l2)

      l1_error = l2_delta.dot(self.syn1.T)
      l1_delta = 0.00001 * l1_error * deriv_sigmoid(l1)

      # Меняем веса
      self.syn2 += l2.T.dot(l3_delta)
      self.syn1 += l1.T.dot(l2_delta)
      self.syn0 += l0.T.dot(l1_delta)


def image_to_array(image_title):
  # Преобразование изображения в массив

  image = np.asarray(Image.open(f"dataset/{image_title}"))
  array = []
  for i in range(len(image)):
    for j in range(len(image)):
      pixel = image[i][j]
      if sum(pixel) > 600:
          pixel = 1
      else:
          pixel = 0
      array.append(pixel)

  return array


def number_to_array(number):
  array = np.array([0 for i in range(10)])
  array[number] = 1
  return array


# Определим набор данных
dataset = os.listdir("dataset")
np.random.shuffle(dataset)
if "prediction.jpg" in dataset:
  dataset.remove("prediction.jpg")

train_X, train_Y = [], []
for image in dataset:
  train_Y.append(image_to_array(image))
  answer = int(image.split('.')[0])
  item_X = [0 for i in range(10)]
  item_X[answer] = 1
  train_X.append(item_X)

train_Y = np.array(train_Y)
train_X = np.array(train_X)

# Обучение нейронной сети
network = NeuralNetwork()
print("\nНачалось обучение нейросети.")
print(f"Пример числа {train_X.argmax()}")  
network.train(train_X, train_Y)
print("Обучение закончено.")

answer = network.prediction(2)
# print(answer)

value = 0
def save_image():
  # Сохранение изображения в датасет

  global window, value

  scaled_image = pg.transform.scale(window.draw_surf, (24, 24))

  files = os.listdir("dataset")
  current_files = list(filter(lambda title: title[0] == str(value), files))
  current_files = list(map(lambda title: int(title.split('.')[1]), current_files))
  if len(current_files) > 0:
    number = max(current_files) + 1
  else:
    number = 1

  pg.image.save(scaled_image, f"dataset/{value}.{number}.jpg")
  window.draw_surf.fill((0, 0, 0))


def predict_number():
  # Предсказание цифры по изображению

  global window, network

  window.draw_surf.fill((0, 0, 0))

  answer = network.prediction(value)
  matrix = [[0] for i in range(24)]
  for i in range(24):
    item = answer[i*24 : (i+1)*24]
    matrix[i] = item
    for j in range(24):
      matrix[i][j] **= 1.5
        
  matrix = list(np.array(matrix).T)
  for i in range(len(matrix)):
    matrix[i] = list(matrix[i])

  for i in range(24):
    for j in range(24):
      item = [matrix[i][j] * 255]
      matrix[i][j] = tuple(item * 3)
      pg.draw.rect( window.draw_surf, color = matrix[i][j], rect = (i*24, j*24, 24, 24))

  return answer


class Button():
  # Гибкий класс кнопки

  def __init__(self, screen, label, rect, func):
    self.screen = screen
    self.label = label
    self.rect = rect
    self.func = func
  

  def draw(self):
    pg.draw.rect(self.screen, (150, 150, 150), self.rect)
    x, y = self.label.get_size()
    self.screen.blit(self.label, [self.rect[0] + self.rect[2] * 0.5 - x * 0.5, self.rect[1] + self.rect[3] * 0.5 - y * 0.5])


class Window():
  # Интерфейс программы

  def __init__(self):
    self.screen = pg.display.set_mode((1000, 600))
    pg.display.set_caption("Neural Network")

    self.draw_surf = pg.Surface((576, 576))
    self.draw_surf.fill((0, 0, 0))

    self.font = pg.font.Font(None, 26)
    self.font_small = pg.font.Font(None, 20)
    text_predict = self.font.render("Сгенерировать", True, (0, 0, 0))
    text_save = self.font.render("Сохранить в датасет", True, (0, 0, 0))

    button_save = Button(self.screen, text_save, (700, 450, 200, 60), save_image)
    button_predict = Button(self.screen, text_predict, (700, 250, 200, 60), predict_number)
    self.buttons = [button_save, button_predict]
    
    self.prediction_text = ""


  def update(self):
    global value
    # Отрисовка всего в окне

    self.screen.fill((255, 255, 255))
    self.screen.blit(self.draw_surf, (12, 12))
    for button in self.buttons:
      button.draw()

    text_prediction = self.font.render("Цифра: " + str(value), True, (0, 0, 0))
    text_pred_1 = self.font_small.render("Перед тем, как сгенерировать цифру", True, (0, 0, 0))
    text_pred_2 = self.font_small.render("нажмите на клавиатуре на эту цифру", True, (0, 0, 0))

    self.screen.blit(text_prediction, (650, 100))
    self.screen.blit(text_pred_1, (650, 180))
    self.screen.blit(text_pred_2, (650, 200))

    text_save_1 = self.font_small.render("Перед тем, как сохранить изображение в датасет,", True, (0, 0, 0))
    text_save_2 = self.font_small.render("нажмите на клавиатуре на цифру", True, (0, 0, 0))
    text_save_3 = self.font_small.render("соответствующую изображённой цифре", True, (0, 0, 0))
    
    self.screen.blit(text_save_1, (650, 380))
    self.screen.blit(text_save_2, (650, 400))
    self.screen.blit(text_save_3, (650, 420))

    pg.display.update()


  def run(self):
    # Условно-бесконечный цикл работы окна

    global value

    self.update()
    draw_on = False
    running = True
    while running:
      for event in pg.event.get():
        if event.type == pg.QUIT:
          running = False
        elif event.type == pg.MOUSEBUTTONDOWN:
          x, y = event.pos
          draw_on = True
          for button in self.buttons:
            if x > button.rect[0] and x < button.rect[0] + button.rect[2] and y > button.rect[1] and y < button.rect[1] + button.rect[3]:
              button.func()

        if event.type == pg.MOUSEBUTTONUP:
          draw_on = False
        if event.type == pg.MOUSEMOTION:
          if draw_on:
            x, y = event.pos
            pg.draw.rect(self.draw_surf, (255, 255, 255), (x-24, y-24, 24, 24))
        elif event.type == pg.KEYDOWN:
          if event.key == pg.K_SPACE:
            self.buttons[1].func()
          elif event.key == pg.K_s:
            self.buttons[0].func()
          elif event.key == pg.K_0:
            value = 0
          elif event.key == pg.K_1:
            value = 1
          elif event.key == pg.K_2:
            value = 2
          elif event.key == pg.K_3:
            value = 3
          elif event.key == pg.K_4:
            value = 4
          elif event.key == pg.K_5:
            value = 5
          elif event.key == pg.K_6:
            value = 6
          elif event.key == pg.K_7:
            value = 7
          elif event.key == pg.K_8:
            value = 8
          elif event.key == pg.K_9:
            value = 9
      
      self.update()
    pg.quit()


window = Window()
window.run()
