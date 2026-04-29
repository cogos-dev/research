# Contributing to CogOS Research

Thanks for your interest. This repo holds the public research underlying CogOS — the EA/EFM thesis, the LoRO framework, and proof-of-concept experiments.

## What contributions are welcome

- **Clarifications and corrections** to existing theses and frameworks
- **Prior-art citations** that contextualize or challenge claims
- **PoC experiments** that test framework predictions empirically
- **Typos, formatting, link rot** — always welcome

## What contributions need discussion first

- **Framework-level changes** to the EA/EFM thesis or LoRO framework
- **New theoretical claims** — open an issue first with the thesis sketch
- **Large literature-review additions** — scope-check via issue before investing effort

Open a [Feature Request](https://github.com/cogos-dev/research/issues/new?template=feature.yml) to propose the shape of a large change before drafting it.

## Document conventions

- Prose docs live under `eaefm/`, `loro/`, and sibling directories
- PoC experiments live under `poc/` with their own README per experiment
- Papers live under `papers/` (preprint PDF + LaTeX source where applicable)
- Markdown uses GFM
- Citations: inline `[Author YYYY]` + bibliography section

## Reproducibility for PoC contributions

If your PoC involves code:

1. Include a self-contained README with exact run commands
2. Pin dependencies (`requirements.txt`, `environment.yml`, or equivalent)
3. Seed RNGs and note the seeds in the README
4. Document hardware (CPU / GPU / RAM) the experiment was run on

## Submitting changes

1. Fork the repo and create a branch from `main`
2. Make your changes
3. Verify markdown renders (GitHub preview is fine for smoke-testing)
4. Open a pull request using the org PR template

## Reporting issues

Use the org-level [Bug Report](https://github.com/cogos-dev/research/issues/new?template=bug.yml) or [Feature Request](https://github.com/cogos-dev/research/issues/new?template=feature.yml) forms.

## License

By contributing, you agree that your contributions will be licensed under the [MIT License](LICENSE). This includes text, figures, and PoC code.
