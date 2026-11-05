# D09: public distributed consumer environments and offline Transformers training

Completed/activated on both hosts2026-11-05. Launcher/fixtures e1bbb8ec7;
installer/CLI refusal maintenance37067fdde; export/verifierbb4d672ea.
`bc250-distributed` now selects `/opt/bc250-stack/tools/distributed-d09` on both.
Frozen native daily/RCCL Torch a524980/ROCm/images/kernels/daily selections remain
unchanged. D05 licensed source-derived agent exit repair remains byte-identical;
Torch source8d8b18d unchanged. No native build, package installation or reboot.

## Installed behavior

Adds ordered `--env`, `--env-file`, `--mount`, `--mount-ro`, `--shm-size` before
the training script by reusing the ordinary Torch launcher's parser/validation.
Repeated environment flags/files apply left-to-right; literal values/spaces survive.
Arguments after the script remain script arguments. Explicit read-only script
parents retain precedence over automatic mounts. Home/cwd/script paths and caller
UID/groups follow the existing launcher architecture.

After the private0600 caller-context handoff, root revalidates the fixed two-node
Python/RCCL topology, rendezvous, script/bootstrap and mandatory read-only tools.
Native loader variables and protected container paths retain ordinary validation;
DDP rendezvous, NCCL/TORCH_NCCL/TORCHELASTIC settings, Gloo interface, alternate
Python worker executable and native architecture/loader overrides are controlled.
Socket/eth0/Ring/Simple with P2P/IB/SDMA/symmetric off remains the qualified path.
The command does not choose a host venv interpreter; its pinned CPython3.12 process
can receive compatible consumer site-packages through PYTHONPATH and mounts.

Agent and each fresh worker validate actual Torch package/C-extension paths,
version and SHA256 for torch/__init__.py, _C extension, libtorch_python.so and
libtorch_hip.so against the same both-host identity file. Existing parent-side
RCCL/native profile checks remain. Then the agent installs the unchanged D05 exit
method, and the worker executes the requested script with its argv/script import
path/cwd preserved. Accidental consumer Torch replacement refuses before user
script execution. These launch checks add startup hashing, not a training-speed
optimization or universal dependency compatibility guarantee.

## Consumer/reference qualification

Uses only stack-owned frozen python-consumers on each host: Transformers5.2.0,
peft0.18.1, accelerate1.12.0, native Torch2.13.0a0+gfx1013. No user application
or model download. Owned offline random BertForMaskedLM config: vocab97, width32,
two layers/four heads/intermediate64/sequence12,22,561 trainable parameters,
dropout0/requested eager attention. FP32 GPU, independent CPU-double model;
AdamW lr.001/eps1e-5/weight_decay.01/foreachFalse. Global batch8, two ranks with
four examples each, two micros/rank and no_sync on the first.

Before and after activation, two successive global losses/every parameter gradient
and AdamW parameter update compare CPU-double: loss3e-6/5e-5, gradient3e-6/5e-4,
parameter4e-6/5e-5 (atol/rtol), unchanged bounds. The existing FP16/no_sync FSDP D08
fixture also passes through the candidate new bootstrap: independent gradients,
owned AdamW state, one-shard nonfinite skip, scaler/checkpoint exact replay.
This is direct Torch DDP integration with a tiny Transformers consumer, not HF
Trainer/Accelerate orchestration, arbitrary/new model, mixed-precision Bert,
production convergence, large-model capacity or performance qualification.

Actual external mounts under `/var/tmp/bc250-d09-fixtures/`: read-only source
folder is owned/writable by UID1000 on host but container write returns EROFS;
separate rw folder accepts owned markers. Frozen consumer overlay is read-only.
Interleaved env-file/explicit overrides and inherited caller value with spaces
match, and script receives the literal spaced argument. Cwd also contains spaces.

Actual Nik SSH→doas→public command on both hosts repeats context/env/mount checks
and CPU Gloo sum3, with torch.cuda.is_initialized() false throughout. That field
is a Torch initialization flag, not a whole-driver memory/access measurement.
Three CPU guard refusals/rank temporarily alter package metadata/expected version/
expected native digest in the isolated test process, then restore them; no runtime
files are changed.37 local tests (29 retained+8 new) include malformed private
context and parser contracts.12 actual CLI refusals/host cover protected variables,
transport/rendezvous overrides, mounts, shm size, alternate profile/shell and port;
errors occur before container execution. No broad untrusted-code isolation claim.

## Activation, recovery and evidence

EX common host locks stage immutable versioned tools, preserving active D05 command
and registry. Candidate consumer and retained FSDP pass before activation. Then
activate, live rollback to retained D05 command/byte-exact registry, reactivate,
and installed public consumer pass. Durable source payload/expected names/hashes
checked; no ordinary native profile or daily command change. D05 fallback keeps
the agent exit fix. Registry updates only tool/consumer identity/recovery metadata;
native runtime SHA, release, agent fix and old agent recovery remain identical.

Recovery helpers and install-payload: `bc250work/tools/distributed-d09/`.
Record: `evidence/distributed-d09-tool-install-r1` contains old registry/command,
activation hashes and live rollback proof. Helpers acquire EX locks themselves.
Use the [current recovery commands](../../../../BC250_DISTRIBUTED_GUIDE.md).
Old D05/D01 tools, releases/images, caches, checkpoints, SSH/boot fallbacks and
native controls/health package stay intact. No user-application inspection/test.

46 GPU worker check groups, six ordinary CPU caller groups,24 actual CLI refusals,
six CPU identity refusals,37 local tests and six acknowledged GPU guards. Retained
FSDP includes12 checkpoint refusals/four scaler mismatch refusals (overlapping
counts, not independent coverage percentages). All accepted agents exit cleanly;
GPU kernel logs/registry stable per phase. Guarded GPU runs:4CPUs/cpuset0–3,
10GiB RAM/12GiB memory+swap,512PIDs/256MiB consumer shm,600s guards/460s deadlines.
Ordinary CPU test uses public defaults with four fixture Torch CPU threads.
Black/critical Ruff+unused-import/syntax/diff pass; full upstream CI not run.
.104 r2/.105 r3, boots7344171f/e1cbfa23 unchanged. Historical Python CPU segfault/
KFD uncertainty remains separate from fixed VM heap and D05 agent-exit repair.

All controllers exit0: candidate15499/19447, retained15907/20252,
installed16774/21763. No jobs/containers/armed guards remain. Both hosts retain
`evidence/distributed-d09-r1/{consumer-candidate,retained-candidate,consumer-installed}`,
CLI-refusals.json, user-cpu/rank.json, tool recovery/payload and fixture dirs.
Owned retained FSDP checkpoint payloads stay on hosts; exports omit .pt files.
Workstation kernel/artifacts/distributed-d09 contains payload, verified-results.json
and raw archives; verifier checks installed/source hashes, both-node payload
identity, guard/kernel/worker reports, private-user results and rollback metadata.

| Archive | Bytes | SHA256 |
| --- | --- | --- |
|104-evidence-r1.tar.gz|259027|69b3293ef4cfa6c1c3d5466cd4923780e39af2ce8879231476e2d3be6c4c39ee|
|105-evidence-r1.tar.gz|242047|394d5eaf2923c77def6882f6219d79602b846af59ec1e59e4e64e9f197a58cc5|

Both host docs/memory are synchronized. **Next D10 (new planned batch):** bounded
fresh-process two-host checkpoint/restart, model/optimizer/scaler plus scheduler/
RNG/data-cursor references and strict config/world/payload refusals, using owned
fixtures through the public command. Keep optimizer-boundary/fixed-world scope;
no elastic/peer-crash or arbitrary production checkpoint promise. Capacity,
other transports and release reconstruction remain separate open work.
