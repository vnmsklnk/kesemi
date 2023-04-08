# Kesemi
Kesemi is a small tool that allows to calculate
the pose of one RGB frame relative to another RGB frame and the depth image corresponding to it.

## How to run

Build docker container:
```shell
docker build -t kesemi .
```

Run docker:
```shell
docker run -it --rm -p 8888:8888 kesemi
```

Run jupyter-lab inside docker:

```shell
jupyter-lab --allow-root --no-browser --ip 0.0.0.0 --port 8888
```

Play with `pipeline.ipynb` notebook