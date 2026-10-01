"""CLI `sorter`. Perintah lain ditambah per milestone."""

import json
import secrets
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from statistics import mean
from typing import Annotated

import typer
from pydantic import TypeAdapter
from rich.console import Console
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from sorter.agent import TRIAGE_CATEGORIES, Workspace, build_categories, run_agent
from sorter.apply import ApplyError, apply_plan, make_playground, undo
from sorter.assign import assign
from sorter.classify import AppleModel, AppleUnavailable, classify_item, evaluate
from sorter.cluster import cluster_descriptors
from sorter.config import Settings
from sorter.describe import describe_items, sample_with_copies
from sorter.llm import ModelError, ModelUnavailable, Ollama
from sorter.plan import build_plan
from sorter.duplicates import find_duplicates
from sorter.sample import SampleFolderError, make_sample_folder
from sorter.scan import UnsafeOutputError, assert_outside, list_items
from sorter.schemas import Descriptor, Inventory, Plan, Taxonomy, Triage, Verdict
from sorter.store import Cache, Trace

app = typer.Typer(no_args_is_help=True, add_completion=False)
SETTINGS = Settings()

console = Console()

LABEL_STYLE = {
    Triage.NEEDS_MODEL: "bold cyan",
    Triage.PROJECT: "magenta",
    Triage.DATASET: "blue",
    Triage.APP: "green",
    Triage.INSTALLER: "green",
    Triage.ARCHIVE: "yellow",
    Triage.SKIPPED: "dim",
}


@app.command()
def scan(
    folder: Annotated[Path, typer.Argument(help="Folder yang dipindai.")],
    hash_content: Annotated[
        bool, typer.Option("--hash/--no-hash", help="--no-hash: jangan buka isi file sama sekali.")
    ] = True,
    details: Annotated[bool, typer.Option("--details", help="Tampilkan semua item.")] = False,
) -> None:
    """Daftar item level teratas  triage  duplikat. Baca-saja."""
    folder = _check_folder(folder)
    items = _list(folder)
    groups = find_duplicates(items, SETTINGS, check_content=hash_content)

    dup_of = {d: g.original for g in groups if g.confirmed for d in g.duplicates}
    for it in items:
        it.duplicate_of = dup_of.get(it.path)

    run_dir = _new_run_dir()
    inv = Inventory(
        run_id=run_dir.name,
        folder=folder,
        scanned_at=datetime.now(),
        content_checked=hash_content,
        items=items,
        duplicates=groups,
    )
    out = run_dir / "inventory.json"
    out.write_text(inv.model_dump_json(indent=2))

    _print_scan(inv, details)
    console.print(f"\n[dim]Tersimpan:[/dim] {out}")

@app.command()
def describe(
    folder: Annotated[Path, typer.Argument(help="Folder yang item-nya dibaca model.")],
    sample: Annotated[int, typer.Option(min=1, help="Jumlah item yang dibaca.")] = (
        SETTINGS.sample_size
    ),
    model: Annotated[str, typer.Option(help="Nama model di Ollama.")] = SETTINGS.model,
    agent_model: Annotated[
        str, typer.Option(help="Model untuk langkah agen. Kosong: sama dengan --model.")
    ] = "",
) -> None:
    """Qwen membaca cuplikan isi sampel item needs_model. Baca-saja; hasil di-cache."""
    folder = _check_folder(folder)
    items = _list(folder)
    chosen, copies = sample_with_copies(items, sample, SETTINGS)
    n_copies = sum(it.path in copies for it in chosen)
    total = sum(it.triage is Triage.NEEDS_MODEL for it in items)
    if not chosen:
        console.print("Tidak ada item needs_model di folder ini.")
        return
    
    llm = _connect(model)
    run_dir = _new_run_dir()
    trace = Trace(run_dir / "trace.jsonl")
    console.print(
        f"[bold]Describe[/bold] {len(chosen)} dari {total} item needs_model "
        f"di {folder}\n[dim]{n_copies} salinan identik tidak dikirim ke model. "
        f"Model {model} lewat Ollama lokal. Baca-saja.[/dim]\n"
    )
    try:
        results = _run_describe(chosen, copies, llm, trace)
    finally:
        trace.close()

    out = run_dir / "descriptors.json"
    out.write_bytes(TypeAdapter(list[Descriptor]).dump_json(results, indent=2))
    _print_describe(results)
    console.print(f"\n[dim]Tersimpan:[/dim] {out}\n[dim]Trace:[/dim]     {run_dir / 'trace.jsonl'}")

@app.command()
def discover(
    folder: Annotated[Path, typer.Argument(help="Folder yang disusun kategorinya.")],
    sample: Annotated[int, typer.Option(min=1, help="Jumlah item yang dibaca.")] = (
        SETTINGS.sample_size
    ),
    model: Annotated[str, typer.Option(help="Nama model di Ollama.")] = SETTINGS.model,
    agent_model: Annotated[
        str, typer.Option(help="Model untuk langkah agen. Kosong: sama dengan --model.")
    ] = "",
) -> None:
    """Describe, cluster, agen merancang kategori, lalu setiap file dipilah. Baca-saja."""
    folder = _check_folder(folder)
    items = _list(folder)
    chosen, copies = sample_with_copies(items, sample, SETTINGS)
    if not chosen:
        console.print("Tidak ada item needs_model di folder ini.")
        return
    llm = _connect(model, SETTINGS.embed_model)
    agent_model = agent_model or model
    agent_llm = _connect(agent_model, timeout=SETTINGS.agent_timeout_s)
    run_dir = _new_run_dir()
    trace = Trace(run_dir / "trace.jsonl")
    console.print(f"[bold]Discover[/bold] {folder}\n[dim]Model {model}, agen {agent_model}, "
                  f"embedding {SETTINGS.embed_model}, lewat Ollama lokal. Baca-saja.[/dim]\n")  # fmt: skip
    try:
        console.print("[bold]1/4 Describe[/bold]")
        results = _run_describe(chosen, copies, llm, trace)
        readable = [d for d in results if d.description and not d.duplicate_of
                    and d.source != "name_only"]  # fmt: skip
        unread = [d.name for d in results if d.source == "name_only" or d.error]

        console.print("\n[bold]2/4 Cluster[/bold]")
        with console.status(
            f"Membuat embedding {len(readable)} ringkasan dengan {SETTINGS.embed_model}..."
        ):
            clusters = cluster_descriptors(readable, llm, SETTINGS)
        (run_dir / "clusters.json").write_text(
            json.dumps([c.model_dump() for c in clusters], indent=2, ensure_ascii=False)
        )
        console.print(f"{len(readable)} item dikelompokkan jadi {len(clusters)} cluster awal")

        console.print(
            f"\n[bold]3/4 Agen merancang kategori[/bold] "
            f"[dim](maksimal {SETTINGS.agent_max_steps} giliran)[/dim]"
        )
        workspace = Workspace(clusters, readable, SETTINGS)
        _print_clusters(workspace)
        note = (
            f"{len(unread)} item tidak terbaca isinya dan tidak perlu dikategorikan."
            if unread
            else ""
        )
        with AgentView() as view:
            steps = run_agent(workspace, agent_llm, SETTINGS, trace, view.on_event, note)

        designed, review = workspace.final or [], {}
        if designed:
            console.print(f"\n[bold]4/4 Pilah[/bold] [dim]{len(readable)} file, masing-masing "
                          "dua kali dengan urutan kategori dibalik[/dim]")  # fmt: skip
            assigned, review = _run_assign(readable, designed, llm, trace)
            for c in designed:
                c.items = [n for n, cat in assigned.items() if cat == c.name]
            
    except (ModelUnavailable, ModelError) as exc:
        _fail(f"{exc}\nItem yang sudah dibaca tersimpan di cache.")
    finally:
        trace.close()

    taxonomy = Taxonomy(
        run_id=run_dir.name,
        folder=folder,
        model=model,
        categories=build_categories(designed, items, copies),
        unread=unread,
        review=review,
        agent_steps=steps,
        agent_finished=workspace.final is not None,
    )
    (run_dir / "descriptors.json").write_bytes(
        TypeAdapter(list[Descriptor]).dump_json(results, indent=2)
    )
    out = run_dir / "taxonomy.json"
    out.write_text(taxonomy.model_dump_json(indent=2))
    plan = build_plan(taxonomy, items, results, copies)
    (run_dir / "plan.json").write_text(plan.model_dump_json(indent=2))
    _print_taxonomy(taxonomy)
    moving = [p for p in plan.items if p.action == "move"]
    console.print(
        f"\n[dim]Tersimpan:[/dim] {out}\n[dim]Trace:[/dim]     {run_dir / 'trace.jsonl'}\n"
        f"[dim]Rencana:[/dim]   {run_dir / 'plan.json'} ({len(moving)} dipindah, "
        f"{len(plan.items) - len(moving)} tetap). Boleh diubah dulu, lalu:\n"
        f"           [bold]sorter apply {run_dir}[/bold]"
    )

@app.command()
def apply(
    run: Annotated[Path, typer.Argument(help="Folder run hasil discover, misal runs/2026...")],
    target: Annotated[Path, typer.Option(help="Lokasi playground.")] = Path("~/agent-playground"),
    yes: Annotated[bool, typer.Option("--yes", "-y", help="Lewati konfirmasi.")] = False,
) -> None:
    """Jalankan plan.json di playground (salinan folder asli). Folder asli tidak disentuh."""
    run_dir = _find_run(run)
    plan = Plan.model_validate_json((run_dir / "plan.json").read_text())
    _print_plan(plan)
    moving = sum(p.action == "move" for p in plan.items)
    if not yes and not typer.confirm(f"\nPindahkan {moving} item di playground {target}?"):
        raise typer.Exit()
    try:
        with console.status(f"Menyiapkan playground dari {plan.folder}..."):
            playground = make_playground(plan.folder, target)
        stats = apply_plan(plan, playground, run_dir / "journal.jsonl")
    except ApplyError as exc:
        _fail(str(exc))
    console.print(
        f"\n[green]{stats['dipindah']} item dipindah[/green] di {playground}"
        + (f", [yellow]{stats['tidak_ditemukan']} tidak ditemukan[/yellow]"
           if stats["tidak_ditemukan"] else "")  # fmt: skip
        + f"\n[dim]Folder asli {plan.folder} tidak berubah. Batalkan dengan:[/dim] "
        f"[bold]sorter undo {run_dir}[/bold]"
    )


@app.command("undo")
def undo_command(
    run: Annotated[Path, typer.Argument(help="Folder run yang sudah di-apply.")],
) -> None:
    """Kembalikan semua pemindahan satu run di playground."""
    run_dir = _find_run(run)
    try:
        stats = undo(run_dir / "journal.jsonl")
    except ApplyError as exc:
        _fail(str(exc))
    console.print(f"[green]{stats['dikembalikan']} item dikembalikan[/green] ke tempat semula"
                  + (f", [yellow]{stats['bentrok']} dilewati karena tempatnya sudah terisi "
                     "atau itemnya sudah tidak ada[/yellow]" if stats["bentrok"] else ""))  # fmt: skip


@app.command()
def classify(
    run: Annotated[Path, typer.Argument(help="Folder run hasil discover (sumber kategori).")],
    item: Annotated[
        Path | None, typer.Option(help="Satu file untuk diklasifikasi, seperti watcher.")
    ] = None,
) -> None:
    """Model Apple on-device memilih kategori. Tanpa --item: bandingkan dengan agen Qwen."""
    run_dir = _find_run(run)
    tax = Taxonomy.model_validate_json((run_dir / "taxonomy.json").read_text())
    plan = Plan.model_validate_json((run_dir / "plan.json").read_text())
    categories = [c for c in tax.categories if c.source == "agent"]
    try:
        model = AppleModel()
    except AppleUnavailable as exc:
        _fail(str(exc))
    if item is not None:
        _classify_one(item.expanduser().resolve(), categories, model)
        return

    folder = _check_folder(plan.folder)
    expected = {p.name: p.category for p in plan.items if p.source == "agen"}
    console.print(f"[bold]Classify[/bold] {len(expected)} item dari {folder}\n[dim]Model Apple "
                  f"on-device, {len(categories)} kategori dari {run_dir.name}. Baca-saja.[/dim]\n")  # fmt: skip
    try:
        verdicts = evaluate(_list(folder), expected, categories, model, SETTINGS,
                            lambda v: console.print(_verdict_line(v)))  # fmt: skip
    except AppleUnavailable as exc:
        _fail(str(exc))
    out = run_dir / "classify.json"
    out.write_bytes(TypeAdapter(list[Verdict]).dump_json(verdicts, indent=2))
    _print_agreement(verdicts)
    console.print(f"\n[dim]Tersimpan:[/dim] {out}")


@app.command("sample-folder")
def sample_folder(
    target: Annotated[Path, typer.Argument(help="Lokasi folder contoh.")] = Path("~/sorter-sample"),
) -> None:
    """Buat folder contoh berisi data palsu, supaya bisa mencoba tanpa menyentuh folder asli."""
    try:
        path = make_sample_folder(target)
    except SampleFolderError as exc:
        _fail(str(exc))
    console.print(f"Folder contoh dibuat: [bold]{path}[/bold]\nCoba: sorter scan {path}")

# ---------- Bantuan ----------


def _find_run(run: Path) -> Path:
    for candidate in (run, SETTINGS.runs_dir / run):
        if (candidate / "plan.json").is_file():
            return candidate
    _fail(f"plan.json tidak ditemukan di {run}. Jalankan discover dulu.")


def _fail(message: str) -> None:
    console.print(f"[red]{message}[/red]")
    raise typer.Exit(2)


def _check_folder(folder: Path) -> Path:
    folder = folder.expanduser().resolve()
    if not folder.is_dir():
        _fail(f"Folder tidak ditemukan: {folder}")
    try:
        assert_outside(SETTINGS.runs_dir, folder)
        assert_outside(SETTINGS.cache_db.parent, folder)
    except UnsafeOutputError as exc:
        _fail(str(exc))
    return folder


def _list(folder: Path):
    try:
        return list_items(folder, SETTINGS)
    except PermissionError:
        _fail(
            f"Tidak punya izin membaca {folder}. Izinkan Terminal di System Settings > "
            "Privacy & Security > Files and Folders."
        )


def _new_run_dir() -> Path:
    run_dir = SETTINGS.runs_dir / f"{datetime.now():%Y%m%d-%H%M%S}-{secrets.token_hex(2)}"
    run_dir.mkdir(parents=True)
    return run_dir

def _connect(model: str, embed_model: str | None = None, timeout: float | None = None) -> Ollama:
    try:
        llm = Ollama(SETTINGS.ollama_host, model, timeout or SETTINGS.request_timeout_s)
        llm.check()
        if embed_model:
            llm.check(embed_model)
    except (ModelUnavailable, ModelError) as exc:
        _fail(str(exc))
    return llm


def _run_assign(readable, categories, llm: Ollama, trace: Trace):
    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    with Progress(*columns, TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task("Memilah", total=len(readable))

        def on_done(name: str, category: str | None, doubt: str | None) -> None:
            trace.log("assign", name, category=category, review=doubt)
            progress.advance(task)

        assigned, review = assign(readable, categories, llm, on_done)
    console.print(f"{len(assigned)} file terpilah, [magenta]{len(review)} masuk review[/magenta] "
                  f"({len(review) / max(1, len(readable)):.0%})")  # fmt: skip
    return assigned, review


def _run_describe(chosen, copies, llm: Ollama, trace: Trace) -> list[Descriptor]:
    columns = (TextColumn("{task.description}"), BarColumn(), MofNCompleteColumn())
    with Progress(*columns, TimeElapsedColumn(), console=console, transient=True) as progress:
        task = progress.add_task("Membaca", total=len(chosen))

        def on_done(d: Descriptor) -> None:
            progress.console.print(_describe_line(d))
            progress.advance(task)

        try:
            return describe_items(
                chosen, SETTINGS, llm, Cache(SETTINGS.cache_db), trace, on_done, copies
            )
        except ModelUnavailable as exc:
            _fail(f"{exc}\nItem yang sudah selesai tersimpan di cache.")


# ---------- Tampilan scan ----------


def _print_scan(inv: Inventory, details: bool) -> None:

    by_label = defaultdict(list)
    for it in inv.items:
        by_label[it.triage].append(it)
    confirmed = [g for g in inv.duplicates if g.confirmed]
    maybe = [g for g in inv.duplicates if not g.confirmed]

    console.print(f"\n[bold]Scan[/bold] {len(inv.items)} item di {inv.folder}")
    console.print("[dim]Baca-saja: tidak ada file yang dipindah, diganti nama, atau dihapus.[/dim]")
    console.print(
        f"[bold cyan]{len(by_label[Triage.NEEDS_MODEL])}[/bold cyan] perlu dibaca model  ·  "
        f"[bold green]{len(confirmed)}[/bold green] grup duplikat identik "
        f"({sum(len(g.duplicates) for g in confirmed)} salinan)  ·  "
        f"[bold yellow]{len(maybe)}[/bold yellow] grup perlu dicek\n"
    )

    table = Table(title="Triage", title_justify="left", title_style="bold")
    table.add_column("Label")
    table.add_column("Jumlah", justify="right")
    table.add_column("Contoh", overflow="fold")
    for label, style in LABEL_STYLE.items():
        if items := by_label[label]:
            more = f" [dim]{len(items) - 4} lagi[/dim]" if len(items) > 4 else ""
            names = ", ".join(it.name for it in items[:4])
            table.add_row(f"[{style}]{label.value}[/{style}]", str(len(items)), names + more)
    console.print(table)

    if inv.duplicates:
        dt = Table(title="Duplikat (hanya saran, tidak ada yang dihapus)", title_justify="left",
                   title_style="bold")  # fmt: skip
        dt.add_column("Status")
        dt.add_column("Asli", overflow="fold")
        dt.add_column("Salinan", overflow="fold")
        dt.add_column("Catatan", style="dim")
        for g in confirmed + maybe:
            status = "[green]identik[/green]" if g.confirmed else "[yellow]belum terbukti[/yellow]"
            dt.add_row(status, g.original.name, "\n".join(p.name for p in g.duplicates), g.note)
        console.print(dt)
    if not inv.content_checked:
        console.print("[dim]Isi file tidak diperiksa (--no-hash): duplikat hanya dari nama.[/dim]")

    if details:
        for label, style in LABEL_STYLE.items():
            if not (items := by_label[label]):
                continue
            t = Table(
                title=f"[{style}]{label.value}[/{style}] ({len(items)})", title_justify="left"
            )
            t.add_column("Nama", overflow="fold")
            t.add_column("Alasan", style="dim")
            t.add_column("Salinan dari", style="green")
            for it in items:
                t.add_row(it.name, it.reason, it.duplicate_of.name if it.duplicate_of else "")
            console.print(t)

# ---------- Tampilan describe ----------


def _describe_line(d: Descriptor) -> str:
    if d.error:
        return f"[red]gagal[/red]  {d.name}  [dim]{d.error}[/dim]"
    if d.duplicate_of:
        return f"[dim]salinan  {d.name}  ← {d.duplicate_of.name}[/dim]"
    doc_type = d.description.doc_type if d.description else ""
    if d.from_cache:
        return f"[dim]cache  {d.name}  {doc_type}[/dim]"
    return (
        f"[green]ok[/green]     {d.name}  [cyan]{doc_type}[/cyan]  [dim]{d.seconds:.1f} dtk[/dim]"
    )


def _print_describe(results: list[Descriptor]) -> None:
    table = Table(title="\nHasil", title_justify="left", title_style="bold")
    table.add_column("Nama", overflow="fold")
    table.add_column("Dilihat", style="dim")
    table.add_column("Jenis", style="cyan")
    table.add_column("Ringkasan", overflow="fold")
    for d in results:
        if d.description:
            table.add_row(d.name, d.source, d.description.doc_type, d.description.summary)
        else:
            table.add_row(d.name, d.source, "[red]gagal[/red]", f"[dim]{d.error}[/dim]")
    console.print(table)

    # Yang tidak memanggil model (name_only, salinan) tidak ikut rata-rata waktu
    new = [
        d for d in results
        if not d.from_cache and not d.error and not d.duplicate_of and d.source != "name_only"
    ]  # fmt: skip
    cached = sum(d.from_cache for d in results)
    copied = sum(bool(d.duplicate_of) for d in results)
    failed = sum(bool(d.error) for d in results)
    console.print(
        f"[green]{len(new)}[/green] baru  ·  [dim]{cached} dari cache[/dim]  ·  "
        f"[dim]{copied} salinan[/dim]  ·  [red]{failed}[/red] gagal"
    )
    if new:
        by_source = defaultdict(list)
        for d in new:
            by_source[d.source].append(d.seconds)
        per_source = ", ".join(f"{s} {mean(v):.1f}" for s, v in sorted(by_source.items()))
        avg = mean(d.seconds for d in new)
        minutes = avg * SETTINGS.sample_size / 60
        color = "green" if minutes < 30 else "yellow"
        console.print(
            f"Rata-rata [bold]{avg:.1f} detik/item[/bold] [dim]({per_source})[/dim]\n"
            f"Perkiraan {SETTINGS.sample_size} item: [{color}]{minutes:.0f} menit[/{color}] "
            "[dim](target PRD di bawah 30 menit)[/dim]"
        )



# ---------- Tampilan discover ----------


def _short(value, limit: int = 90) -> str:
    text = json.dumps(value, ensure_ascii=False)
    return text if len(text) <= limit else text[: limit - 3] + "..."


def _print_clusters(ws: Workspace) -> None:
    console.print(f"Cluster awal yang dilihat agen ({len(ws.clusters)}):")
    for cid, members in sorted(ws.clusters.items()):
        flag = "  [yellow]campuran[/yellow]" if ws.mixed(members) else ""
        console.print(
            f"  [dim]{cid:>2}[/dim]  {len(members):>2} item  [cyan]{ws.kinds(members)}[/cyan]{flag}"
        )


class AgentView:
    """Menampilkan proses agen di terminal: status saat model berpikir, lalu tiap langkahnya."""

    def __init__(self) -> None:
        self.status = console.status("")
        self.seconds: float | None = None

    def __enter__(self) -> "AgentView":
        return self

    def __exit__(self, *exc) -> None:
        self.status.stop()

    def on_event(self, kind: str, data: dict) -> None:
        step = f"[dim]{data['step']:>2}[/dim]"
        if kind == "turn_start":
            self.status.update(f"Agen sedang berpikir (giliran {data['step']})...")
            self.status.start()
        elif kind == "turn_end":
            self.status.stop()
            self.seconds = data["seconds"]  # ditampilkan di baris tool pertama giliran ini
            if data["content"]:
                console.print(f"{step}  [italic]catatan agen:[/italic] {data['content']}")
        elif kind == "nudge":
            console.print(f"{step}  [yellow]agen menjawab dengan teks, bukan tool; diingatkan "
                f"({data['count']}/3)[/yellow]")  # fmt: skip
        elif kind == "stuck":
            console.print(f"{step}  [yellow]3 giliran gagal berturut-turut; agen diberi daftar "
                "cluster terbaru dan diminta propose[/yellow]")  # fmt: skip
        elif kind == "tool":
            took = f"  [dim]({self.seconds:.1f} dtk)[/dim]" if self.seconds is not None else ""
            self.seconds = None
            console.print(
                f"{step}  {_describe_tool(data['name'], data['args'], data['result'])}{took}"
            )


def _describe_tool(name: str, args: dict, r: dict) -> str:
    """Satu kalimat yang menjelaskan apa yang baru saja dilakukan agen."""
    if "error" in r:
        return f"[cyan]{name}[/cyan]({_short(args, 50)}) [red]gagal: {r['error']}[/red]"
    if name == "inspect_cluster":
        kinds = ", ".join(r["jenis_semua_item"].split(", ")[:3])
        head = f"[cyan]melihat isi cluster {args['cluster_id']}[/cyan]"
        return f"{head}: {len(r['items']) + r['lainnya']} item ({kinds})"
    if name == "peek_item":
        return f"[cyan]membuka {args['name']}[/cyan]: {r['doc_type']}, {r['summary'][:70]}"
    if name == "propose":
        cats = args.get("categories", [])
        names = ", ".join(c.get("name", "?") for c in cats)
        head = f"[cyan]mengusulkan {len(cats)} kategori[/cyan]: {names}"
        if r.get("diterima"):
            notes = "".join(f"\n      [dim]catatan: {w}[/dim]" for w in r.get("peringatan", []))
            return f"{head} → [green]diterima[/green]{notes}"
        return f"{head} → [yellow]ditolak[/yellow]:\n      " + "\n      ".join(r["kesalahan"])
    return f"[cyan]{name}[/cyan]({_short(args, 50)}) → {_short(r)}"

def _classify_one(path: Path, categories, model) -> None:
    if not path.exists():
        _fail(f"File tidak ditemukan: {path}")
    it = next(i for i in _list(path.parent) if i.name == path.name)
    if it.triage is not Triage.NEEDS_MODEL:  # sama seperti discover: aturan dulu, model belakangan
        fixed = TRIAGE_CATEGORIES[it.triage][0] if it.triage in TRIAGE_CATEGORIES else "tetap"
        console.print(f"{it.name}: [bold]{fixed}[/bold] [dim](aturan: {it.reason})[/dim]")
        return
    try:
        v = classify_item(it, categories, model, SETTINGS)
    except AppleUnavailable as exc:
        _fail(str(exc))
    console.print(_verdict_line(v))


def _verdict_line(v: Verdict) -> str:
    took = f"[dim]{v.seconds:>4.1f}s[/dim]"
    if v.error:
        return f"{took}  [red]gagal[/red]  {v.name}  [red]{v.error}[/red]"
    got = v.category or "[yellow]Tidak cocok[/yellow]"
    if v.expected is None:
        mark = ""
    elif v.category == v.expected:
        mark = "[green]sama[/green]  "
    else:
        mark = f"[yellow]beda[/yellow]  agen: {v.expected} → "
    level = f"[dim]({v.confidence})[/dim] " if v.confidence else ""
    if not v.review:
        flag = "[green]otomatis[/green] "
    elif v.second != v.category:
        flag = f"[magenta]review: jawaban kedua {v.second or 'Tidak cocok'}[/magenta] "
    else:
        flag = "[magenta]review[/magenta] "
    return f"{took}  {v.name}  {mark}[bold]{got}[/bold] {level}{flag}[dim]{v.reason}[/dim]"


def _print_agreement(verdicts: list[Verdict]) -> None:
    done = [v for v in verdicts if not v.error]
    if not done:
        return
    same = [v for v in done if v.category == v.expected]
    console.print(f"\n[bold]Sama dengan agen:[/bold] {len(same)}/{len(done)} "
                  f"({len(same) / len(done):.0%})"
                  + (f", [red]{len(verdicts) - len(done)} gagal[/red]"
                     if len(done) < len(verdicts) else ""))  # fmt: skip

    table = Table(title="\nPer kategori agen", title_justify="left", title_style="bold")
    table.add_column("Kategori agen")
    table.add_column("Item", justify="right")
    table.add_column("Sama", justify="right")
    table.add_column("Model Apple memilih", overflow="fold")
    by_cat = defaultdict(list)
    for v in done:
        by_cat[v.expected].append(v)
    for cat, vs in sorted(by_cat.items(), key=lambda kv: -len(kv[1])):
        other = Counter(v.category or "Tidak cocok" for v in vs if v.category != cat)
        table.add_row(cat, str(len(vs)), str(sum(v.category == cat for v in vs)),
                      ", ".join(f"{c} x{n}" for c, n in other.most_common()))  # fmt: skip
    console.print(table)

    held = [v for v in done if v.review]
    auto = [v for v in done if not v.review]
    console.print(
        f"[bold]Masuk review queue:[/bold] {len(held)}/{len(done)} file "
        f"({sum(v.category != v.expected for v in held)} di antaranya beda dengan agen)\n"
        f"[bold]Dipindah otomatis:[/bold] {len(auto)} file "
        f"({sum(v.category != v.expected for v in auto)} di antaranya beda dengan agen: "
        "periksa ini, karena kesalahan di sini tidak ditahan)"
    )

    parts = []
    for level in ("tinggi", "sedang", "rendah"):
        vs = [v for v in done if v.confidence == level]
        if vs:
            agree = sum(v.category == v.expected for v in vs) / len(vs)
            parts.append(f"{level}: {len(vs)} item, {agree:.0%} sama")
    console.print("[bold]Keyakinan model[/bold] (jujur bila 'tinggi' lebih sering sama): "
                  + "; ".join(parts))  # fmt: skip
    secs = sorted(v.seconds for v in done)
    console.print(f"[bold]Waktu per file:[/bold] median {secs[len(secs) // 2]:.1f} dtk, "
                  f"terlama {secs[-1]:.1f} dtk [dim](target PRD: di bawah 5 dtk)[/dim]")
    console.print("[dim]Catatan: 'sama' bukan berarti benar. Agen juga bisa salah; periksa "
                  "baris 'beda' satu per satu.[/dim]")

def _print_plan(plan: Plan) -> None:
    table = Table(title=f"Rencana untuk {plan.folder}", title_justify="left", title_style="bold")
    table.add_column("Item", overflow="fold")
    table.add_column("Tujuan", overflow="fold")
    table.add_column("Dari")
    table.add_column("Alasan", overflow="fold", style="dim")
    order = sorted(plan.items, key=lambda p: (p.action == "stay", p.dest or "", p.name.lower()))
    for p in order:
        if p.action == "stay":
            table.add_row(f"[dim]{p.name}[/dim]", "[dim]tetap[/dim]", "", p.reason)
            continue
        dest = ("[green]" if p.dest.startswith("Documents") else "[yellow]") + p.dest + "[/]"
        name = p.name + (" [red](salinan)[/red]" if p.duplicate_of else "")
        table.add_row(name, dest, p.source, p.reason[:120])
    console.print(table)

def _print_taxonomy(tax: Taxonomy) -> None:
    if not tax.agent_finished:
        console.print(
            f"\n[yellow]Agen belum menyelesaikan propose dalam {tax.agent_steps} giliran.[/yellow] "
            "Lihat trace.jsonl untuk melihat di mana ia berhenti."
        )
    table = Table(title="\nKategori usulan (belum ada file yang dipindah)", title_justify="left",
                  title_style="bold")  # fmt: skip
    table.add_column("Kategori", style="bold")
    table.add_column("Tingkat")
    table.add_column("Item", justify="right")
    table.add_column("Deskripsi", overflow="fold")
    table.add_column("Contoh", overflow="fold", style="dim")
    for c in tax.categories:
        tier = ("[green]penting[/green] → Documents" if c.tier == "important"
                else "[yellow]sementara[/yellow] → Downloads")  # fmt: skip
        name = c.name + (" [dim](aturan)[/dim]" if c.source == "triage" else "")
        count = str(len(c.items)) if c.items else "[red]0[/red]"  # kategori kosong: deskripsi meleset
        table.add_row(name, tier, count, c.description, ", ".join(c.items[:3]))
    console.print(table)
    if tax.review:
        console.print(f"\n[magenta]Masuk review ({len(tax.review)}), tidak dipindah:[/magenta]")
        for name, why in list(tax.review.items())[:15]:
            console.print(f"  {name}  [dim]{why}[/dim]")
        if len(tax.review) > 15:
            console.print(f"  [dim]... dan {len(tax.review) - 15} lagi di taxonomy.json[/dim]")
    if tax.unread:
        shown = ", ".join(tax.unread[:10]) + (f", ... ({len(tax.unread)} total)"
                                              if len(tax.unread) > 10 else "")  # fmt: skip
        console.print(f"[dim]Tidak terbaca, belum dikategorikan: {shown}[/dim]")