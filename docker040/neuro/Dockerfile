
FROM pytorch/pytorch:0.4_cuda9_cudnn7

RUN apt-get update -y && apt-get upgrade -y && \
	apt-get -y install python3-pip vim wget supervisor less
RUN pip3 install tensorboardx flask
