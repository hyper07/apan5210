FROM python:3.11-slim

WORKDIR /usr/src/app

# dont write pyc files
# dont buffer to stdout/stderr
ENV PYTHONDONTWRITEBYTECODE 1
ENV PYTHONUNBUFFERED 1

COPY ../requirements/streamlit/requirements.txt /usr/src/app/requirements.txt
COPY ../requirements/streamlit/requirements-dev.txt /usr/src/app/requirements-dev.txt

# RUN apt-get update && apt-get install -y libgomp1 libglib2.0-0
RUN apt-get update && apt-get install -y libgl1 libgomp1 libglib2.0-0

# dependencies
RUN pip install --upgrade pip setuptools wheel \
    # && set -x HDF5_DIR "$(brew --prefix hdf5)" \
    && pip3 install --only-binary h5py h5py \
    && pip install -r requirements.txt \
    && pip install -r requirements-dev.txt \
    && rm -rf /root/.cache/pip

# COPY ./app /usr/src/app
