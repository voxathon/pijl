# pijl

A vibe-coded visual logic circuit editor and simulator. Place gates, draw wires, and
package circuits into reusable macros.

## Performance
See [BENCH.md](BENCH.md). More benchmarks coming.

## Headless
Macros run without the editor too: from Python, the command line, or as a
co-process over a pipe. See [HEADLESS.md](HEADLESS.md).

## Modding support

Modding support is extremely extensive, every aspect of pijl is moddable. See [MODDING.md](MODDING.md).

## Install

```bash
uv tool install pijl
```

or with pip:

```bash
pip install pijl
```

## Run

```bash
pijl
```

This opens the launcher: start the editor on the last project, pick, rename or
make projects, change the settings, or run a macro headless. `pijl gui` skips
the launcher and goes straight to the editor; `pijl --tui` shows the launcher in
the terminal instead of a window; `pijl prefs` shows the settings, and
`pijl prefs ui.scale=1.5` changes one.

On Windows, `pijl-gui` launches without a console window.

Projects are saved under `%APPDATA%\pijl` on Windows (set `PIJL_DATA` to use a
different location).

## Development

```bash
uv sync
uv run pytest
uv run pijl
```

## License

MIT
