FROM quay.io/centos/centos:stream9
RUN dnf install -y python3.11 python3.11-pip mesa-libGL libglvnd-glx && dnf clean all

ENV PYTHONUNBUFFERED=1
ENV MODEL_PATH=/app/model/my_model.h5
ENV HAARCASCADE_PATH=/app/model/haarcascade_frontalface_default.xml

WORKDIR /app

COPY app/req.txt .
RUN pip3.11 install --no-cache-dir -r req.txt

COPY ./app /app
COPY ./model /app/model

EXPOSE 8000

ENTRYPOINT ["python3.11"]
CMD ["app.py"]
