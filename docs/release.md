---
nav_exclude: false
---
# Release Guide

This repository uses GitHub Actions to automatically publish the `vimms` package to [PyPI](https://pypi.org/) whenever a GitHub release is published. Follow the steps below to prepare a new release.

## 1. Configure Secrets

1. Obtain a PyPI API token from your PyPI account.
2. Add the token to the GitHub repository secrets as `PYPI_API_TOKEN`.

The workflow uses this secret to authenticate with PyPI when uploading the package.

## 2. Bump the Version

Update the version number in `pyproject.toml` before creating a release. Commit and push the change to the default branch.

## 3. Create a Release

1. Tag the commit with the desired version and push the tag, or use GitHub's web interface to create a new release.
2. Publishing the release triggers the `release-to-pypi` workflow. This workflow will:
   - build the package with Poetry,
   - upload the resulting files from the `dist/` directory to PyPI.

If the upload succeeds, the package will be available on PyPI under the new version.
