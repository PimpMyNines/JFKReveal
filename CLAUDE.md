# Project Setup and Commands

## Environment
- You need to set up an OpenAI API key in the `.env` file
- For local development, use the `text-embedding-ada-002` model instead of newer models

## Common Commands
- `make setup`: Create virtual environment and install dependencies
- `make run`: Run the full pipeline
- `make run SKIP_SCRAPING=1 SKIP_PROCESSING=1`: Run only the analysis part of the pipeline
