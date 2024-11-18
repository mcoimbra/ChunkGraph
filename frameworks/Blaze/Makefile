cwd = $(shell pwd)

docker_image:
	docker pull junokim8/blaze:1.0

docker_build:
	docker build -t junokim8/blaze:1.0 .

docker_run:
	@docker run --rm -it -v "/dev/nvme0n1":/mnt/nvme1/dataset junokim8/blaze:1.0 /bin/bash || true
