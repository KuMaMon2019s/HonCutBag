# HonCut Migration Document

## Migration Overview

**Migration Date**: 2026-07-30  
**Source Location**: `/Users/soda/knowledge-base/2026-07-28_01/`  
**Target Location**: `/Users/soda/projects/honcut/`

## Migration Reasons

1. **Disorganized Project Structure**: Code scattered across multiple date directories (2026-07-27_05, 2026-07-27_06, 2026-07-28_01, 2026-07-30_01)
2. **Version Inconsistency**: Different directories contained different code versions
3. **Lack of Standardization**: No unified project configuration and dependency management
4. **Environment Dependencies**: Missing virtual environment configuration leading to dependency conflicts

## New Project Structure

```
/Users/soda/projects/honcut/
├── pipeline/              # Core pipeline code
│   ├── src/              # Python source code (23 files)
│   │   ├── pipeline_runner.py
│   │   ├── character_factory.py
│   │   ├── seedance_client.py
│   │   └── ...
│   └── requirements.txt  # Python dependencies
├── docker/               # Docker configuration
│   └── docker-compose.yml
├── data/                 # Data directories
│   ├── input/           # Input files
│   └── output/          # Output files
├── docs/                 # Documentation
├── scripts/              # Utility scripts
├── .gitignore           # Git ignore rules
├── .env.example         # Environment variable template
├── Makefile             # Common commands
├── pyproject.toml       # Python project configuration
├── environment.yml      # Conda environment configuration
└── README.md            # Project documentation
```

## Old Directory Handling

### Archived
- `/Users/soda/knowledge-base/archived/2026-07-27_05/` - Old version code
- `/Users/soda/knowledge-base/archived/2026-07-27_06/` - Old version code

### Deleted
- `/Users/soda/knowledge-base/2026-07-30_01/` - Duplicate copy

### Retained
- `/Users/soda/knowledge-base/2026-07-28_01/` - Kept as reference (latest version)

## How to Use the New Project

### 1. Activate Virtual Environment

```bash
conda activate honcut
```

### 2. Install Dependencies

```bash
cd /Users/soda/projects/honcut
make install
```

### 3. Run Pipeline

```bash
# Using make
make run INPUT_FILE=path/to/script.txt

# Or run directly
python pipeline/src/pipeline_runner.py --input path/to/script.txt --output-dir data/output
```

### 4. Start Docker Services

```bash
make docker-up
```

## Key Improvements

1. **Unified Project Structure**: All code consolidated in `/Users/soda/projects/honcut/`
2. **Standardized Configuration**: Using `pyproject.toml` and `environment.yml`
3. **Dependency Management**: `requirements.txt` explicitly lists all dependencies
4. **Environment Isolation**: Conda virtual environment prevents dependency conflicts
5. **Docker Integration**: `docker-compose.yml` manages services
6. **Makefile**: Simplifies common operations

## Verification Checklist

- [x] 23 Python files migrated
- [x] Configuration files created (pyproject.toml, environment.yml)
- [x] Docker configuration created
- [x] Git repository initialized
- [x] `make install` test passed
- [x] `pipeline_runner.py --help` test passed
- [x] Old directories archived/deleted

## Rollback Plan

If rollback is needed, restore from:
- `/Users/soda/knowledge-base/2026-07-28_01/` - Complete backup
- `/Users/soda/knowledge-base/archived/` - Old version code

## Next Steps

1. Update all documentation to point to new project path
2. Configure CI/CD pipeline
3. Add automated tests
4. Improve Docker image build process

---

**Migration Completion Time**: 2026-07-30 22:15  
**Migration Status**: ✅ Successful
