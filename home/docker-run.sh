export DISPLAY=:0
xhost +
docker run -it --rm \
  --runtime nvidia \
  --shm-size=4g \
  --memory=8g \
  --memory-swap=12g \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v $(pwd)/src:/src \
  --device=/dev/video0:/dev/video0 \
  --env="DISPLAY=$DISPLAY" \
  create-dataset-with-sam2:latest
