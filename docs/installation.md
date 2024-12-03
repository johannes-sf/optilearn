# Installation Guide

## Prerequisites
- Python 3.10.8+
- Poetry

## Installation


### Clone the Repository

```bash
git clone https://gitlab.com/spryfox/optilearn/multi-objective-optimization.git
cd multi-objective-optimization
```

### Set Up Virtual Environment

```bash
poetry config http-basic.spryfox-gitlab <gitlab_username> <gitlab_api_token>
poetry env use $(pyenv local)
poetry install
```

### Verify Installation

```bash
poetry run python -c "import moo_classification; print('Installation successful!')"
```