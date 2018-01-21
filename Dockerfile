FROM python:2.7

RUN pip install numpy==1.13.1
RUN pip install nibabel==2.1.0
RUN pip install tensorflow==1.2.1
RUN pip install scipy==0.17.0

COPY . modules modules/
COPY . model model/

ADD test_BraTS_2017.py /
ADD create_modules_objects.py /
ADD . config_files config_files/
