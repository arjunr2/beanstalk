FROM python:3.10-slim-bookworm

ENV PIP_ROOT_USER_ACTION=ignore

ARG PUID=1000
ARG PGID=1000
ARG USERNAME=evaluator

RUN groupadd -g "${PGID}" "${USERNAME}" && \
    useradd -u "${PUID}" -g "${USERNAME}" -m -s /bin/bash "${USERNAME}"

ENV PATH="$HOME/.local/bin:$PATH"
RUN pip3 install --upgrade pip
RUN pip3 install matplotlib jaxtyping tqdm
RUN pip3 install -U "jax[cuda12]"

USER ${USERNAME}
WORKDIR /home/${USERNAME}


ENTRYPOINT ["bash"]
