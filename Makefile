# Only for development
sync: 
	uv sync 

format:
	uv run ruff format -v .

lint:
	uv run ruff check --select I --fix .	

check:
	uv run pyright

unit-test:
	uv run pytest -v --cov=. --cov-report=term-missing

integration-test:
	uv run python ./test/main.py

clean:
	find . -type d -name "__pycache__" -exec rm -r {} +
	rm
