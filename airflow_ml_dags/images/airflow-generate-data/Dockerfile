FROM python:3.9-slim-buster

COPY requirements.txt .

ENV VIRTUAL_ENV "/venv"
ENV PATH "$VIRTUAL_ENV/bin:$PATH"

RUN python -m venv $VIRTUAL_ENV \
    && pip install -r requirements.txt

COPY generate_data.py /generate_data.py

ENTRYPOINT ["python", "generate_data.py"]