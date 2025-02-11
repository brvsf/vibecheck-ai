# Tests
test_LoadData:
	pytest tests/test_LoadData.py

test_TextCleaning:
	pytest tests/test_TextCleaning.py

test_Preprocessing:
	pytest tests/test_Preprocessing.py

test:
	pytest tests/test_LoadData.py
	pytest tests/test_TextCleaning.py
	pytest tests/test_TextCleaning.py
	pytest tests/test_Preprocessing.py


# Cleaning cache
clear:
	rm -rf __pycache__ .pytest_cache
	rm -rf src/__pycache__ src/package/__pycache__
