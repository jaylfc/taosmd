### Fixed
- Added `.venv/` and a bare `.venv` pattern to `.gitignore` so both a real
  `.venv` directory and a `.venv` symlink are ignored. The trailing-slash
  form alone does not match a symlink named `.venv`, which is the shape that
  was committed into PR #436 as a symlink to an absolute machine path.
