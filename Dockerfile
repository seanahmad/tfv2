FROM jupyter/tensorflow-notebook:1386e2046833

#Set the working directory
WORKDIR /home/jovyan/
USER root
# Jupyter Stacks Tensorflow notebook does not have R (see image relationships https://jupyter-docker-stacks.readthedocs.io/en/latest/using/selecting.html#image-relationships)
# So, taking what is in jupyter stacks R dockerfile and putting here
# R pre-requisites
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
    fonts-dejavu \
    unixodbc \
    unixodbc-dev \
    r-cran-rodbc \
    gfortran \
    gcc && \
    rm -rf /var/lib/apt/lists/*
# Fix for devtools https://github.com/conda-forge/r-devtools-feedstock/issues/4
RUN ln -s /bin/tar /bin/gtar

# R packages
RUN conda update -n base conda && \
    conda install --quiet --yes \
    'r-base=3.6.1' \
    'r-caret=6.0*' \
    'r-crayon=1.3*' \
    'r-devtools=2.0*' \
    'r-forecast=8.7*' \
    'r-hexbin=1.27*' \
    'r-htmltools=0.3*' \
    'r-htmlwidgets=1.3*' \
    'r-irkernel=1.0*' \
    'r-nycflights13=1.0*' \
    'r-plyr=1.8*' \
    'r-randomforest=4.6*' \
    'r-rcurl=1.95*' \
    'r-reshape2=1.4*' \
    'r-rmarkdown=1.14*' \
    'r-rodbc=1.3*' \
    'r-rsqlite=2.1*' \
    'r-shiny=1.3*' \
    'r-sparklyr=1.0*' \
    'r-tidyverse=1.2*' \
    'r-corrplot=0.84'\
    'unixodbc=2.3.*' \
    && \

    #---------------
    # From https://anaconda.org/conda-forge/r-tensorflow
    # From https://anaconda.org/conda-forge/r-keras
    conda install -c conda-forge r-tensorflow && \
    conda install -c conda-forge/label/gcc7 r-tensorflow && \
    conda install -c conda-forge/label/cf201901 r-tensorflow && \
    conda install -c conda-forge r-keras && \
    conda install -c conda-forge/label/broken r-keras && \
    conda install -c conda-forge/label/cf201901 r-keras && \

    # From https://github.com/scrapinghub/splash/issues/887#issuecomment-472380823
    pip install --upgrade nbconvert && \
    #---------------

    conda clean --all -f -y && \
    fix-permissions $CONDA_DIR

# Install e1071 R package (dependency of the caret R package)
  RUN conda install --quiet --yes r-e1071

# Modules
COPY requirements.txt /home/jovyan/requirements.txt
RUN apt-get update && \
    pip install -r /home/jovyan/requirements.txt

# Add files
COPY python-binder /home/jovyan/python-binder
COPY r-binder /home/jovyan/r-binder
COPY README.md /home/jovyan/README.md
COPY ./postBuild /home/$NB_USER/postBuild

# Allow user to write to directory
RUN chown -R $NB_USER /home/jovyan && \
    chmod -R 774 /home/jovyan && \
    rm -fR /home/jovyan/work && \
    /home/$NB_USER/postBuild

# Change back to user jovyan
USER $NB_USER

# Expose the notebook port
EXPOSE 8888
