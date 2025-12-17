## Development Workflow (Optional but Recommended)

This project uses [`just`](https://github.com/casey/just) as a task runner
to provide a clean, consistent developer workflow.

### Install `just`

Choose the method that fits your system:

#### macOS
```sh
brew install just
```
#### Linux
# Debian / Ubuntu
```sh
sudo apt install just
```
```sh
# Arch
sudo pacman -S just
```
```sh
# Fedora
sudo dnf install just
```

#### Windows
```sh
winget install --id Casey.Just --exact
```

#### Cross-platform
```sh
cargo install just
```
## To Verify the installation
```sh
just --version
```
### Usage 
```sh
just dev     # start development server
just build   # production build
just lint    # run checks
just format # running the formatter 
```


“We implement a lightweight segmentation-based OCR pipeline as a pre-processing step, while keeping the CNN architecture unchanged.”