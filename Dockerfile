FROM python:3.9.13-slim

RUN mkdir student_performance
RUN cd student_performance

WORKDIR student_performance

ADD . .

RUN apt update -y && apt install awscli -y
RUN pip3 install -r requirements.txt -q

EXPOSE 8501

HEALTHCHECK CMD curl --fail https://localhost:8501/_store/health

ENTRYPOINT ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]