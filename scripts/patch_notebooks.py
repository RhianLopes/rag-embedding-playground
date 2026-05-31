import json, glob, re, os

notebooks = sorted(glob.glob("**/*.ipynb", recursive=True))
notebooks = [n for n in notebooks if ".venv" not in n and ".ipynb_checkpoints" not in n and "test_render" not in n]

for path in notebooks:
    with open(path, "r", encoding="utf-8") as f:
        nb = json.load(f)

    changed = False
    for cell in nb["cells"]:
        src = "".join(cell.get("source", []))

        if cell["cell_type"] == "markdown":
            new_src = re.sub(r"\(([^)]+)\.ipynb\)", lambda m: f"({m.group(1)}.html)", src)
            if new_src != src:
                cell["source"] = new_src
                changed = True
                src = new_src

        if cell["cell_type"] == "code" and "fig.show()" in src:
            new_src = src.replace(
                "fig.show()",
                'from IPython.display import display, HTML\nimport plotly.io as pio\ndisplay(HTML(pio.to_html(fig, full_html=False, include_plotlyjs="cdn")))'
            )
            cell["source"] = new_src
            cell["outputs"] = []
            changed = True

        if cell.get("outputs"):
            cell["outputs"] = []
            changed = True
        if cell.get("execution_count") is not None:
            cell["execution_count"] = None
            changed = True

    if changed:
        with open(path, "w", encoding="utf-8") as f:
            json.dump(nb, f, ensure_ascii=False, indent=1)
            f.write("\n")
        print(f"Patched: {path}")
    else:
        print(f"Clean:   {path}")
