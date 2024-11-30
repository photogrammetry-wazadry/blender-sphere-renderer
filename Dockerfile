FROM ubuntu:22.04

RUN mkdir /usr/local/workdir
WORKDIR /usr/local/workdir

ENV TZ=Europe/Moscow
RUN ln -snf /usr/share/zoneinfo/$TZ /etc/localtime && echo $TZ > /etc/timezone

RUN apt update && apt upgrade -y
RUN apt update && apt install git vim byobu zsh zip unzip exiftool curl -y
RUN curl -sS https://bootstrap.pypa.io/get-pip.py | python3.10
RUN apt-get -y install '^libxcb.*-dev' libx11-xcb-dev libglu1-mesa-dev libxrender-dev libxi-dev libxkbcommon-dev libxkbcommon-x11-dev libsm6 libxext6 libgomp1

RUN python3.10 -m pip install bpy==3.5.0 tqdm filetype flask pandas requests flup pyexiftool==0.4.13 psycopg2-binary
RUN python3.10 -m pip install gunicorn gevent

# Install zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)"
RUN chsh -s $(which zsh) root
RUN echo "alias python=\"python3.10\"" >> ~/.zshrc

COPY ./ ./
WORKDIR /usr/local/workdir

CMD ["/bin/zsh"]

