# VSR
This service provides the inference for Visual Speech Recognition.


## Clone repository
```
git clone https://github.com/Namadgi/auto_avsr.git
cd auto_avsr
git checkout ts
git pull origin ts
```

## Install prerequisites
```
pip install -r requirements.txt
```

## Install additional libraries
```
git clone https://github.com/hhj1897/face_detection.git
cd face_detection
git lfs pull
pip install -e .
cd ..
```
```
git clone https://github.com/hhj1897/face_alignment.git
cd face_alignment
pip install -e . --pre
cd ..
```

## Download model weights
```
mkdir model_weights
pip install gdown
gdown 19GA5SqDjAkI5S88Jt5neJRG-q5RUi5wi
mv vsr_trlrwlrs2lrs3vox2avsp_base.pth model_weights/vsr_base.pth
```

## Build and run
```
mkdir model_store
chmod +x archiver.sh
./archiver.sh
docker build -t vsr ./
docker run --rm --gpus all -p 8080:8080 vsr
```

## Send inference
`
curl http://0.0.0.0:8080/predictions/vsr -F "data=@./demo/file.avi"
`

## Result
```
{
  "code": 0,
  "description": "Successful check",
  "result": "HELLO WORLD"
}
```
