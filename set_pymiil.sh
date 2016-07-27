#!/bin/bash

# Got this from stack overflow
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )"

if [ -n "${PYMIIL_DIR}" ]; then
    if [ -d ${PYMIIL_DIR} ]; then
        # remove the previously defined PYMIIL_DIR and it's leading ':'
        PYTHONPATH=`echo $PYTHONPATH | sed -e 's#:'"${PYMIIL_DIR}"'##g'`
        # remove the previously defined PYMIIL_DIR without a leading ':'
        # couldn't get a \? escape on the : to work for some reason.
        PYTHONPATH=`echo $PYTHONPATH | sed -e 's#'"${PYMIIL_DIR}"'##g'`
    fi
fi

PYMIIL_DIR=$DIR
# If the python path already has directories in it, append it to the back
if [ -n "$PYTHONPATH" ]; then
    PYTHONPATH=$PYTHONPATH:$PYMIIL_DIR
else
    PYTHONPATH=$PYMIIL_DIR
fi

export PYMIIL_DIR
export PYTHONPATH
