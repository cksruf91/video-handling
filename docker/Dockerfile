FROM python:3.12.3-slim

# 시간 동기화
ENV TZ=Asia/Seoul
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

# 패키지 목록 업데이트 및 ffmpeg 설치
RUN apt-get update
RUN apt-get install gcc build-essential ffmpeg -y
RUN mkdir /home/download

# 작업 디렉토리 설정
WORKDIR /home/download
