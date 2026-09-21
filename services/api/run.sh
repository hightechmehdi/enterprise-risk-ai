#!/bin/bash

docker run --rm -it \
-v "$(pwd):/home/user/app" \
-p 8000:8000 \
-e MLFLOW_TRACKING_URI="$MLFLOW_TRACKING_URI" \
-e MODEL_NAME="$MODEL_NAME" \
-e MODEL_ALIAS="$MODEL_ALIAS" \
-e AWS_ACCESS_KEY_ID="$AWS_ACCESS_KEY_ID" \
-e AWS_SECRET_ACCESS_KEY="$AWS_SECRET_ACCESS_KEY" \
iris-api:$1
