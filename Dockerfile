FROM python:3.10-slim-bookworm

ENV PIP_ROOT_USER_ACTION=ignore

ARG PUID=1000
ARG PGID=1000
ARG USERNAME=evaluator

RUN groupadd -g "${PGID}" "${USERNAME}" && \
    useradd -u "${PUID}" -g "${USERNAME}" -m -s /bin/bash "${USERNAME}"

USER ${USERNAME}
WORKDIR /home/${USERNAME}

RUN pip3 install --upgrade pip
RUN pip3 install matplotlib
RUN pip3 install jaxtyping
RUN pip3 install tqdm
RUN pip3 install -U "jax[cuda12]"

ENTRYPOINT ["bash"]
