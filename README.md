# pijl

A visual logic circuit editor and simulator. Place gates, draw wires, and
package circuits into reusable macros.

## Performance
See [BENCH.md](BENCH.md). More benchmarks coming.

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

On Windows, `pijl-gui` launches the editor without a console window.

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
