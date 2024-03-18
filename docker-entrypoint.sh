#!/bin/sh
set -e

# if [[ "$1" = "serve" ]]; then
shift 1
torchserve --start --ncs --ts-config /home/model-server/config.properties \
    --model-store /home/model-server/model-store/ \
    --models vsr.mar
# else
    # eval "$@"
# fi

# prevent docker exit
tail -f /dev/null