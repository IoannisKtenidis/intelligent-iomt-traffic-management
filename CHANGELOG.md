# Changelog

All notable changes to this project will be documented in this file.

## [1.1.0] - 2026-08-20

### Added
- Added MIT `LICENSE` file.
- Added `CONTRIBUTING.md` guide.
- Added unit tests in `tests/test_iomt_simulation.py` to run a mini simulation run.
- Added GitHub Actions CI workflow in `.github/workflows/ci.yml`.

### Changed
- Restructured repository layout:
  - Moved core simulation scripts from `Core_Code/` to `src/core/`.
  - Renamed core scripts to standard snake_case (e.g. `lora_dir_aloha.py` instead of `loraDir - ALOHA.py`).
  - Moved datasets from `Core_Code/` to `data/`.
  - Moved models from `Core_Code/` to `models/`.
  - Moved runner scripts from `Execution_Scripts/` to `scripts/` (and renamed to snake_case).
  - Moved docx draft of publication from root to `docs/`.
- Updated all python scripts to load datasets and JSON models from their new directories.
- Updated `docker-compose.yml` and `Dockerfile` to mount/execute scripts from the new paths.
- Rewrote `README.md` and `README_DOCKER.md` to document the new directories and usage instructions.
