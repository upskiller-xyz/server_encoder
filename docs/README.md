# Documentation

This directory contains both public and internal documentation.

## Public Documentation (Published to Docs Site)

These files are included in the published documentation website:

- **[index.md](index.md)** - Documentation homepage
- **[api_reference.md](api_reference.md)** - API endpoint documentation
- **[request_schema.md](request_schema.md)** - Request parameter reference
- **[CONTRIBUTING.md](CONTRIBUTING.md)** - Contribution guidelines

## Internal Documentation (Repository Only)

These files are kept in the repository but **NOT** published to the docs site:

- **[encoding_logic.md](encoding_logic.md)** - Proprietary encoding mechanics and algorithms
- **[../CLAUDE.md](../CLAUDE.md)** - Development guidelines for Claude Code
- **[../API_DOCUMENTATION.md](../API_DOCUMENTATION.md)** - Internal API documentation
- **[../enc.md](../enc.md)** - Internal encoding notes

## Building the Documentation Site

To build and preview the public documentation locally:

```bash
# Install docs dependencies
poetry install --with docs

# Serve locally (with live reload)
poetry run mkdocs serve

# Build static site
poetry run mkdocs build
```

The built site will be in the `site/` directory (excluded from git).

## Publishing Documentation

The MkDocs configuration (`mkdocs.yml`) controls which files are published.

To publish to GitHub Pages:

```bash
poetry run mkdocs gh-deploy
```

## Configuration

See [mkdocs.yml](../mkdocs.yml) for the complete documentation configuration, including the `exclude_docs` section that explicitly lists internal documentation files.
