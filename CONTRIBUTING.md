# Contributing to *pyrelational*

Welcome and thank you for considering contributing to **pyrelational** open source project!

Reading and following these guidelines will help us make the contribution process easy and effective for everyone involved.
It also communicates that you agree to respect the time of the developers managing and developing these open source projects.
In return, we will reciprocate that respect by addressing your issue, assessing changes, and helping you finalize your pull requests.

## Quicklinks

* [Code of Conduct](#code-of-conduct)
* [Getting Started](#getting-started)
    * [Build and install](#build-install)
    * [Unit Testing](#unit-testing)
    * [Continuous Integration](#continuous-integration)
    * [Issues](#issues)
    * [Pull Requests](#pull-requests)
 * [Building Documentation](#building-documentation)

## Code of Conduct

We take our open source community seriously and hold ourselves and other contributors to high standards of communication. By participating and contributing to this project, you agree to uphold our [Code of Conduct](https://github.com/RelationRx/pyrelational/CODE-OF-CONDUCT.md).

## Getting Started

Contributions are made to this repo via Issues and Pull Requests (PRs). A few general guidelines that cover both:

- Search for existing Issues and PRs before creating your own.
- We work hard to makes sure issues are handled in a timely manner but, depending on the impact, it could take a while to investigate the root cause. A friendly ping in the comment thread to the submitter or a contributor can help draw attention if your issue is blocking.

### Build and install
To develop pyrelational, first build and install it from source following the steps

1. Clone a copy of pyrelational from source:

   ```bash
   git clone https://github.com/RelationRx/pyrelational
   cd pyrelational
   ```

2. If you already cloned pyrelational from source, update it:

   ```bash
   git pull
   ```

3. If needed, install dependencies in your environment:

   ```bash
   pip install -r requirements.txt
   ```

   This will install all the required dependencies for the package, the examples, as well as flake8, isort, pytest-cov, and black.

4. Install pyrelational in editable mode:

   ```bash
   pip install -e ".[dev,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall pyrelational again and again.

5. Ensure that you have a working pyrelational installation by running the entire test suite with

   ```bash
   python -m pytest tests
   ```

6. Install pre-commit hooks:

   ```bash
    pre-commit install
   ```

### Unit Testing

The pyrelational testing suite is located under `tests/`.
Run the entire test suite with

```bash
python -m pytest tests
```

### Continuous Integration

pyrelational uses [GitHub Actions](https://github.com/RelationRX/pyrelational/actions) for continuous integration. `pre-commit` will ensure that the code is properly formatted before accepting commits.

### Issues

Issues should be used to report problems with the library, request a new feature, or to discuss potential changes before a PR is created. When you create a new Issue, a template will be loaded that will guide you through collecting and providing the information we need to investigate.

If you find an Issue that addresses the problem you're having, please add your own reproduction information to the existing issue rather than creating a new one. Adding a [reaction](https://github.blog/2016-03-10-add-reactions-to-pull-requests-issues-and-comments/) can also help be indicating to our maintainers that a particular problem is affecting more than just the reporter.

### Pull Requests

PRs to our libraries are always welcome and can be a quick way to get your fix or improvement slated for the next release. In general, PRs should:

- Be written in a way that is easy to understand and maintain.
- Have an attached Issue that describes the problem and the PR name should refer to the Issue.
- The PR author agrees to the [Code of Conduct](https://github.com/RelationRx/pyrelational/blob/master/CODE-OF-CONDUCT.md) and the [LICENSE](https://github.com/RelationRx/pyrelational/blob/master/LICENSE)
- Only fix/add the functionality in question **OR** address wide-spread whitespace/style issues, not both.
- Add unit or integration tests for fixed or changed functionality.
- Include documentation in the repo or on our [docs site]() #TODO.

For changes that address core functionality or would require breaking changes (e.g. a major release), it's best to open an Issue to discuss your proposal first.

In general, we follow the ["fork-and-pull" Git workflow](https://github.com/susam/gitpr)

1. Fork the repository to your own Github account
2. Clone the project to your machine
3. Create a branch locally with a succinct but descriptive name
4. Commit changes to the branch
5. Following any formatting and testing guidelines specific to this repo
6. Push changes to your fork
7. Open a PR in our repository and follow the PR template so that we can efficiently review the changes.


## Building Documentation

To build the documentation:

1. [Build and install](#getting-started) pyrelational from source.
2. Install [Sphinx](https://www.sphinx-doc.org/en/master/) via `pip install sphinx sphinx_rtd_theme`.
3. Generate the documentation via:

   ```bash
   cd docs
   make html
   ```

The documentation is now available to view by opening `docs/build/html/index.html`.
