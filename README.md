### Обучение каскада Хаара с помощью google colaboratory

Создать два каталока pos и neg в которые загрузить картинки
Добавить каталоги в архив smpls.zip и загрузить его на гугл диск


Монтировать диск
```python
from google.colab import drive
drive.mount('/content/drive')
```
Скопировать и разархивировать картинки 
(в каталоге content  появятя каталоги "pos" и "neg"  содержащие по 500 картинок)

```python
!cp "/content/drive/My Drive/smpls.zip" /content
!unzip  /content/smpls.zip
```
Создать файл со списком pos картинок

```python
import os
pos_dir = "pos/"

pos_list = ""
for name in os.listdir(pos_dir):
  if name.endswith(".png"):
    if len(pos_list)<1:
      pos_list =  pos_dir + name + "  1  0 0 300 30"
    else:
      pos_list = pos_list + "\r\n" +  pos_dir + name + "  1  0 0 300 30"

with open('positives.txt', 'r+') as the_file:
    the_file.write(pos_list)
```

Создать файл со списком neg картинок

```python
neg_dir = "neg/"

neg_list = ""
for name in os.listdir(neg_dir):
  if name.endswith(".png"):
    if len(neg_list)<1:
      neg_list =  neg_dir + name
    else:
      neg_list = neg_list + "\r\n" +  neg_dir + name

with open('negatives.txt', 'r+') as the_file:
    the_file.write(neg_list)
```

Создать папку classifier для результатов обучения каскада хаара,
или очистить ее, а так же удалить файл векторов если он существует

```python
!rm /content/positives.vec
!rm -rf /content/classifier
!mkdir /content/classifier
```
Из списка pos файлов создать векторный файл
-w ширика картинок в наборах, -h высота картинок в наборах

```python
!opencv_createsamples -info "positives.txt" -vec "positives.vec" -num 500  -w 50 -h 30
```

Обучить каскад Хаара
-vec путь до файла pos векторов,
-bg путь до списка neg картиок,
-w ширика картинок в наборах, -h высота картинок в наборах

Результаты сохранятся в папке classifier,
если параметры обучения изменяются - ее надо очистить

```python
!opencv_traincascade -data classifier -vec /content/positives.vec -bg /content/negatives.txt\
   -numStages 20 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 500\
   -numNeg 500 -w 50 -h 30 -mode ALL -precalcValBufSize 1024\
   -precalcIdxBufSize 1024
```

Если обучение прервано раньше - чтоб скомпилировать готовый к использованию каскад, нужно в команде обучения заменить numStages на то число эпох  (например 4) которые уже обучились и лежат в каталоге classifier

```python
!opencv_traincascade -data classifier -vec /content/positives.vec -bg /content/negatives.txt\
   -numStages 4 -minHitRate 0.999 -maxFalseAlarmRate 0.5 -numPos 500\
   -numNeg 500 -w 50 -h 30 -mode ALL -precalcValBufSize 1024\
   -precalcIdxBufSize 1024
```

В итоге, в каталоге classifier появится файл  cascade.xml

Тест каскада
```python
import cv2
from IPython.display import display

def colab_imshow(frame):
    if os.path.exists('temp.jpg'):
        os.remove('temp.jpg')

    height, width = frame.shape[:2]
    frame = cv2.resize(frame, (250, int((250 / width) * height)), interpolation=cv2.INTER_CUBIC)
    cv2.imwrite('pic.jpg', frame)
    display(Image.open('pic.jpg'))
    os.remove('pic.jpg')

haarCascade = cv2.CascadeClassifier('/content/classifier/cascade.xml')
image = cv2.imread('/content/test.jpg')
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

haar_feed = haarCascade.detectMultiScale(gray) 

print("Found {0} pieces!".format(len(haar_feed)))

for (x, y, w, h) in haar_feed:
    cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)

colab_imshow(image)
```




