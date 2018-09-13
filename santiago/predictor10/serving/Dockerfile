FROM arllanos/ie_base_image

COPY /requirements.txt /tmp/

RUN cd /tmp && \
    pip install --no-cache-dir -r requirements.txt

COPY ./ /microservice/
WORKDIR /microservice

EXPOSE 5000
CMD ["python","-u","microservice.py","Model","REST","--service-type","MODEL","--persistence","0"]