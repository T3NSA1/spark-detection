# Spark Detection

## Установка

Из корня репозитория:

```powershell
python -m venv venv
.\venv\Scripts\activate
pip install -r requirements.txt
```

## Модель

Основная модель для текущего запуска:

```text
models/sparks_core_grinder_yolo11s_best.pt
```

Она детектит сварку как `welding_arc`, а `fire` и `grinder_spark` используются как отрицательные/подавляющие классы.

## Запуск на папке с видео

Положи видео в:

```text
data/raw/videos
```

Запусти:

```powershell
python src\run_inference.py `
  --input data\raw\videos `
  --output outputs\inference `
  --backend yolo `
  --model models\sparks_core_grinder_yolo11s_best.pt `
  --max-width 960 `
  --yolo-conf 0.25 `
  --yolo-imgsz 640 `
  --device cpu
```

Если есть NVIDIA GPU:

```powershell
python src\run_inference.py `
  --input data\raw\videos `
  --output outputs\inference `
  --backend yolo `
  --model models\sparks_core_grinder_yolo11s_best.pt `
  --max-width 960 `
  --yolo-conf 0.25 `
  --yolo-imgsz 640 `
  --device 0
```

## Запуск на одном видео

```powershell
python src\run_inference.py `
  --input data\raw\videos\test-sparks-easy.mp4 `
  --output outputs\inference `
  --backend yolo `
  --model models\sparks_core_grinder_yolo11s_best.pt `
  --max-width 960 `
  --yolo-conf 0.25 `
  --yolo-imgsz 640 `
  --device cpu
```

## Результат

После запуска появятся:

```text
outputs/inference/videos
outputs/inference/predictions
```

В `videos` лежат видео с нарисованными боксами.  
В `predictions` лежат CSV с результатами по кадрам.

