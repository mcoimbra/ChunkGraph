
.PHONY: chunk_graph clean csr_graph blaze_image_fetch

chunk_graph_build:
	@echo "[INFO] - Building ChunkGraph..."
	@cd apps && export CHUNK=1 && make
	@echo "[INFO] - Finished building ChunkGraph, returning to original directory."

chunk_graph_clean:
	@echo "[INFO] - Building ChunkGraph..."
	@cd apps && export CHUNK=1 && make clean
	@echo "[INFO] - Finished building ChunkGraph, returning to original directory."

clean:
	@echo "[INFO] - Cleaning repository..."
	@echo "[INFO] - Cleaning ChunkGraph..."
	@cd apps && make clean
	@echo "[INFO] - Cleaning CSRGraph..."
	@cd CSRGraph && make clean

csr_graph:
	@echo "[INFO] - Building CSRGraph..."
	@cd CSRGraph && make

blaze_image_fetch:
	@echo "[INFO] - Fetching Blaze Docker image..."
	@cd blaze && make docker_build
	@echo "[INFO] - Finished fetching Blaze Docker image."
