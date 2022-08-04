FROM python:3.9-slim

RUN groupadd -r evaluator && useradd -m --no-log-init -r -g evaluator evaluator

RUN mkdir -p /opt/evaluation /input /output \
    && chown evaluator:evaluator /opt/evaluation /input /output
RUN apt-get -y update
RUN apt-get -y install git

USER evaluator
WORKDIR /opt/evaluation

ENV PATH="/home/evaluator/.local/bin:${PATH}"
RUN python -m pip install --user -U pip
COPY --chown=evaluator:evaluator requirements.txt /opt/evaluation/
RUN python -m pip install --user -r requirements.txt

# Code
COPY --chown=evaluator:evaluator evaluation.py /opt/evaluation/
COPY --chown=evaluator:evaluator settings.py /opt/evaluation/
ADD --chown=evaluator:evaluator isles/ /opt/evaluation/isles/

#ADD --chown=evaluator:evaluator sample_bids/ /opt/evaluation/sample_bids/
ADD --chown=evaluator:evaluator grandchallenges/ /opt/evaluation/grandchallenges/
#ADD --chown=evaluator:evaluator gc_output/ /output/images/stroke-segmentation/
# Data
# ADD --chown=evaluator:evaluator hidden_mha_masks /opt/evaluation/mha_masks/
ADD --chown=evaluator:evaluator atlas_algo_sanitydat /opt/evaluation/mha_masks/

ENTRYPOINT "python" "-m" "evaluation"
