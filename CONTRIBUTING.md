# Contributing to ViMMS

Our goal with ViMMS is to develop a robust framework for advancing the development of new fragmentation strategies in LC-MS metabolomics. We welcome all contributions in the following: 

- Reporting a bug
- Discussing the current state of the code
- Submitting a fix
- Proposing new features

## Software bugs or adding new features

We use github to host code, to track issues and feature requests, as well as accept pull requests. Any bug reports on ViMMS are welcomed. If you have any questions regarding the framework, please feel free to ask us.

- We use GitHub issues to track public bugs. Report a bug by [opening a new issue](); it's that easy!
- For minor changes, please implement your fixes and simply submit it as pull request. 
- For big changes or to propose new features, we recommend to create an issue first to discuss any proposed modification to the codebase before implementation. 
 
The following is the appropriate procedure for contributing:

1. Fork the repo and create your branch from `master`.
2. Implement your codes.
3. If you've added code that should be tested, add tests. We use [pyTest](https://pytest.org) as the testing framework.
5. If you've changed APIs, update the [documentation](https://vimms.readthedocs.io). We use [mkdocs](https://www.mkdocs.org/) and [mkdocstrings](https://github.com/mkdocstrings/mkdocstrings) for documentation.
6. Ensure the test suite passes.
7. Make sure your code lints. For coding style, we follow [PEP8](https://www.python.org/dev/peps/pep-0008/) (with linewidth set to 99 characters). [flake8](https://flake8.pycqa.org/en/latest/) and [autopep8](https://pypi.org/project/autopep8/) have been installed as part of the virtual environment, and you can use them to ensure that your codes are PEP8-compliant.
8. Issue that pull request!

## License
Any contributions you make will be under the [MIT License](http://choosealicense.com/licenses/mit/) that covers the project. Feel free to contact the maintainers if that's a concern.
