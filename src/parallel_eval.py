# parallel_eval.py - runs the UNCHANGED eval.py twice at the same time, one adapter per GPU.
# Same mechanism as parallel_train.py (only the CONFIG lines listed below differ; proven by a line diff).
# Run:  !python parallel_eval.py   (GPU T4 x2, Internet on)

# ---------------- EDIT ONLY THIS BLOCK ----------------
EVAL_PY = "/kaggle/working/eval.py"            # the canonical eval.py (do not edit it)
TEST_FILE = "/kaggle/input/datasets/pujanpant017/dataste001/dataset_corrected/test.jsonl"
MODEL_ID = "Qwen/Qwen2.5-3B"
WORKDIR = "/kaggle/working/launch"
STAGGER_SECONDS = 90
JOBS = [
    {"tag": "eval_qwen_s43", "gpu": 0, "set": {"MODE": "finetuned", "MODEL_ID": MODEL_ID,
        "ADAPTER_DIR": "/kaggle/input/datasets/pujanpant017/adapter4344/adapter43/adapter43", "DATA_FILE": TEST_FILE,
        "OUT_DIR": "/kaggle/working/eval_qwen_s43", "N_ITEMS": None}},
    {"tag": "eval_qwen_s44", "gpu": 1, "set": {"MODE": "finetuned", "MODEL_ID": MODEL_ID,
        "ADAPTER_DIR": "/kaggle/input/datasets/pujanpant017/adapter4344/adapter44/adapter44", "DATA_FILE": TEST_FILE,
        "OUT_DIR": "/kaggle/working/eval_qwen_s44", "N_ITEMS": None}},
]
# ------------------------------------------------------

import ast, difflib, hashlib, json, os, re, subprocess, sys, time

def _sha(path):
    return hashlib.sha256(open(path, "rb").read()).hexdigest()

def patch_script(script, overrides, dest):
    """Copy `script` to `dest`, changing ONLY the CONFIG lines named in `overrides`."""
    lines = open(script, encoding="utf-8").read().split("\n")
    start = next(i for i, l in enumerate(lines) if l.startswith("# ---") and "CONFIG" in l)
    end = next(i for i in range(start + 1, len(lines)) if lines[i].startswith("# ---"))
    changed = set()
    for i in range(start + 1, end):
        m = re.match(r"^([A-Z_]+)\s*=", lines[i])
        if m and m.group(1) in overrides:
            lines[i] = f"{m.group(1)} = {overrides[m.group(1)]!r}"
            changed.add(m.group(1))
    missing = set(overrides) - changed
    assert not missing, f"CONFIG keys not found in {script}: {missing}"
    os.makedirs(os.path.dirname(dest), exist_ok=True)
    open(dest, "w", encoding="utf-8").write("\n".join(lines))
    # safety proof: the copy may differ from the original in exactly the overridden lines
    a = open(script, encoding="utf-8").read().split("\n")
    n_diff = sum(1 for d in difflib.ndiff(a, lines) if d.startswith("- "))
    assert n_diff <= len(overrides), "patched copy differs in more lines than expected"
    return dest

def run_jobs(script, jobs, workdir, stagger):
    os.makedirs(workdir, exist_ok=True)
    try:
        n_gpu = len(subprocess.run(["nvidia-smi", "-L"], capture_output=True, text=True).stdout.strip().splitlines())
    except Exception:
        n_gpu = 0
    print(f"GPUs visible: {n_gpu}")
    assert n_gpu >= len(jobs), "need one GPU per job (turn on GPU T4 x2)"
    pristine = _sha(script)
    print("pristine script sha256:", pristine)
    procs = []
    for k, job in enumerate(jobs):
        tag = job["tag"]
        patched = patch_script(script, job["set"], f"{workdir}/{tag}_{os.path.basename(script)}")
        log = open(f"{workdir}/{tag}.log", "w")
        env = dict(os.environ, CUDA_VISIBLE_DEVICES=str(job["gpu"]), PYTHONUNBUFFERED="1")
        rec = {"tag": tag, "gpu": job["gpu"], "overrides": job["set"], "pristine_sha256": pristine,
               "patched_sha256": _sha(patched), "started_utc": time.strftime("%Y-%m-%d %H:%M:%S", time.gmtime())}
        json.dump(rec, open(f"{workdir}/{tag}_launch.json", "w"), indent=1)
        if os.environ.get("LAUNCH_DRY"):
            print("DRY RUN, patched copy written:", patched); continue
        if k > 0:
            print(f"waiting {stagger}s before starting the next job (avoids two simultaneous downloads)")
            time.sleep(stagger)
        print(f"starting {tag} on GPU {job['gpu']}")
        procs.append((tag, subprocess.Popen([sys.executable, patched], stdout=log, stderr=subprocess.STDOUT, env=env), log))
    last = 0
    while any(p.poll() is None for _, p, _ in procs):
        time.sleep(20)
        if time.time() - last > 300:
            last = time.time()
            print(time.strftime("[%H:%M:%S]"))
            for tag, p, _ in procs:
                txt = open(f"{workdir}/{tag}.log", errors="ignore").read()[-400:].replace("\r", "\n").strip().splitlines()
                print(f"  {tag} ({'running' if p.poll() is None else 'exit ' + str(p.returncode)}):", txt[-1][:150] if txt else "")
    ok = True
    for tag, p, log in procs:
        log.close()
        print(f"{tag}: exit code {p.returncode}")
        ok &= (p.returncode == 0)
        tail = open(f"{workdir}/{tag}.log", errors="ignore").read().replace("\r", "\n").strip().splitlines()[-12:]
        print("\n".join("   " + t[:200] for t in tail))
    print("ALL JOBS FINISHED OK" if ok and procs else "SOMETHING FAILED OR DRY RUN: read the tails above")

run_jobs(EVAL_PY, JOBS, WORKDIR, STAGGER_SECONDS)
