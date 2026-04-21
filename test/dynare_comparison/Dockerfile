FROM debian:testing

RUN apt-get update \
    && apt-get install -y --no-install-recommends dynare \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /work

COPY extract_dynare_results.m /work/
COPY run_all_dynare.sh /work/

RUN chmod +x /work/run_all_dynare.sh

ENTRYPOINT ["/work/run_all_dynare.sh"]
