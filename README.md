video-handling
-----

1. video frame 분할
```shell
python main.py -b chunk -v "{file}.mp4" -s "{save_dir}"
```

2. 이미지 설명(openAI)
```shell
python main.py -b chunk -v "{file}.mp4" -i "{save_dir}" -s "{dataframe}.parquet"
```