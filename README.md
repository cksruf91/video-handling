video-handling
-----

* python : 3.12
* KeyLibs
  * openai==1.54.3
  * opencv-python==4.10.0.84
  * pydub==0.25.1

* 수행 명령어
```shell
export VIDEO="some/video.mp4" 
export NAME="SameCoolName"

python main.py -t chunk -v $VIDEO -i "data/image/$NAME" # chunking
python main.py -t desc -v $VIDEO -i "data/image/$NAME" -o "data/output/$NAME.parquet" # desc
python main.py -t keyword -v $VIDEO -o "data/output/$NAME.parquet" # keyword
python main.py -t stt -v $VIDEO -a "data/audio/$NAME.mp3" -s "data/output/$NAME.txt"  # mp3 
```
