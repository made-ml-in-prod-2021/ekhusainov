FROM python:3.9-slim-buster

COPY requirements.txt .

ENV VIRTUAL_ENV "/venv"
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN python -m venv $VIRTUAL_ENV \
    && pip install -r requirements.txt

COPY validate.py /validate.py

ENTRYPOINT ["python", "validate.py"]