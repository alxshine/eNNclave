#!/bin/bash
# INitialization script for Docker dev environment

# >>> conda initialize >>>
# !! Contents within this block are managed by 'conda init' !!
__conda_setup="$('/usr/local/bin/conda' 'shell.bash' 'hook' 2> /dev/null)"
if [ $? -eq 0 ]; then
    eval "$__conda_setup"
else
    if [ -f "/usr/local/etc/profile.d/conda.sh" ]; then
        . "/usr/local/etc/profile.d/conda.sh"
    else
        export PATH="/usr/local/bin:$PATH"
    fi
fi
unset __conda_setup
# <<< conda initialize <<<

# eNNclave setup
source $SGX_SDK/environment

export LD_LIBRARY_PATH="${ENNCLAVE_HOME}/lib:$LD_LIBRARY_PATH"
export PYTHONPATH="${ENNCLAVE_HOME}/lib:$PYTHONPATH"

export CC=gcc
export CXX=g++