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

python main.py -t chunk desc keyword stt -i $VIDEO -p "data/$NAME" -o "data/output/$NAME.json"
# or
python main.py -t all -i $VIDEO -p "data/$NAME" -o "data/output/$NAME.json"
```

* video format : mp4
