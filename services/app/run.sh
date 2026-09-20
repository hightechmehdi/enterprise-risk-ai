#!/bin/bash

docker run --rm -it \
-v "$(pwd):/home/user/app" \
-p 8501:8501 \
-e FASTAPI_URL="$FASTAPI_URL" \
iris-streamlit:$1
