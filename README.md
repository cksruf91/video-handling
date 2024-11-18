video-handling
-----

* python : 3.12
* KeyLibs
  * openai==1.54.3
  * opencv-python==4.10.0.84
  * pydub==0.25.1

1. video frame 분할
```shell
python main.py -t chunk -v "{file}.mp4" -s "{save_dir}"
```
2. STT
```shell
python main.py -t stt -v "{file}.mp4" -a "{save_file}.mp3" -s "{save_file}.txt"
```

3. 이미지 설명(openAI)
```shell
python main.py -t chunk -v "{file}.mp4" -i "{save_dir}" -s "{dataframe}.parquet"
```