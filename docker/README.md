Docker

# python-ffmpeg image build
```shell
docker build -t python-ffmpeg:3.12.3 .
```

# download video from m3u8
download mp4 file from m3u8 via ffmpeg library
```shell
URI="https://example.com/playlist.m3u8"
DOWNLOAD_DIR="download"
FILE_NAME="output.mp4"
docker run --rm -v $DOWNLOAD_DIR:/home python-ffmpeg:3.12.3 \
    ffmpeg -i $URI -c copy -bsf:a aac_adtstoasc $FILE_NAME
```
