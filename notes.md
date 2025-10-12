step 2: spacy requires: 
```
poetry run python -m spacy download en_core_web_md
```
Или мощнее (использует transformer, требует GPU/больше RAM)
``` 
poetry run python -m spacy download en_core_web_trf 
```