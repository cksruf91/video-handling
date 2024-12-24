Docker

# python-ffmpeg image build
```shell
docker build -t ffmpeg:python-3.12.3 . -f docker/Dockerfile.ffmpeg
```

# download video from m3u8
download mp4 file from m3u8 via ffmpeg library
```shell
URI="https://example.com/playlist.m3u8"
DOWNLOAD_DIR="download"
FILE_NAME="output.mp4"
docker run --rm -v $DOWNLOAD_DIR:/home/download ffmpeg:python-3.12.3 \
    ffmpeg -i $URI -c copy -bsf:a aac_adtstoasc $FILE_NAME
```

# build lambda
* https://docs.aws.amazon.com/ko_kr/lambda/latest/dg/python-image.html

해당 코드는 예시로만 남김
```shell
docker build --platform linux/amd64 -t lambda_function:vidoe-1.0.0 . -f docker/Dockerfile
# test lambda in local
docker run --rm --platform linux/amd64 -p 9000:8080 lambda_function:vidoe-1.0.0
```

payload
```shell
curl "http://localhost:9000/2015-03-31/functions/function/invocations" -d "{\"foo\": \"bar\"}"
```

