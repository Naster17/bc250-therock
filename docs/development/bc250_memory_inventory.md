# BC-250 durable implementation and hardware inventory

Audited 2026-10-14 through **code milestone `ce0025df5`**. This is the detailed
project memory; [flow.md](../../flow.md) is the short resume cursor and ordered
plan. Read the relevant section here before investigating a supposedly missing
feature. Update affected entries after substantive work, preserving evidence.





## Release development storage and source audit (2026-11-10)

.104 reclaimed 44.20 GiB after retiring verified redundant proof, staging,
evidence, helper, docs and legacy workspace copies plus obsolete objects.
[Report and recovery receipts](bc250_storage_cleanup_20261110/README.md).
55.3 GiB free before final docs sync. Live evidence remains at its registered
paths, with maintenance records grouped under evidence/maintenance/20261110.
D52 source archive moved to tools/source-d52/package-r1 with an old-path link.
The tracked source audit recovered raw build variants and verified Git/LFS
identities. The reviewed audio and core patches are already present in the
local histories. Retain the observed 22-file Torch HIP output reference for D54.
Both host profiles match on the checked Torch/HIP/MIOpen binaries and image.
The profiler SDK differs by the qualified D49 repair on .104. .105 comparison
was read-only. D54 remains the next dependency and clean-build admission step.

## D53 external compiler/ROCm/RCCL payload (2026-11-10)

Implementation ba613c857; [report](bc250_source_d53_results/README.md).
Complete retained prefix/RCCL/toolchain/two ccache helpers streamed to workstation:
2,467,945,440B SHA76a018fae10972070ff414dbe5ef7c55e94f2674076cb86b8d69fd317db28d98.
28,008 entries/10,726,325,706 file bytes; every actual archive entry checked against
manifest, EXACT separate host manifest; before/after source inventory equal.
224 lexical link targets present; not ELF/header/CMake/relocation/build closure.
HIPClang55833a05 matches D51/RCCLca790dc7 retained; build SDK a6dc remains distinct
activeD49SDK2a975370.52 license-named files, license/source closure still open.
160 local tests/seven new cases/Black/pycompile pass; gzip CRC enforced. Executed
helper before verifier-only hardening preserved. Owned controller14238/child14239
finished/no jobs/guards. EXboard+SHruntime/CPU0-1/2GiBAS/600CPU/900wall/100ms/
3GiBhost+5GiBdisk reserves. Export289.099s; minhost15,165,878,272B/disk12,425,912,320B.
Proposed40GiB sequential fresh-source/build/link/new20GiBcache+5GiB reserve fails
12.4GB free. Proposal not measured mandatory peak; next D54 audit ELF/CMake/link
closure and revised storage plan with existing-cache bounded growth/no reset.
No configure/HIPify/nativebuild/GPU/prefix/kernel/app/cache changes;105 untouched.
104docs20261110-d53-final; payload local kernel/artifacts/source-d53, host metadata
in tools/source-d53/evidence/source-d53-r1. PLAN07 open/four packages remain;
historical PythonCPU/KFD/105clocksource causes still unknown/separate.

## D52 original builder preservation and fresh source assembly (2026-11-10)

Implementation f05489d31; [report](bc250_source_d52_results/README.md).
OCI original image28ad36e6:682,451,968B,14 compressed/uncompressed layers+config
verified104/local, original image untouched. External HIP/ROCm not embedded.
D47/50/51 fresh assembly69 dependencies, sourceHEAD8d8b18d/nativebasea524980,
Gloo913f01fb;21,501+78,598=100,099 source working entries. Eight branches excluded.
R1 catches CRLF working/canonical blob mismatch; R2 checks copied working bytes
against verified snapshots AND preserves canonical Git checks. Failed R1 kept.
Gitless346,534,969B archive/100,116 source+recipe entries verified local and after
space-path extraction. Private Git proof histories preserved outside snapshot.
New future pip recipes refuse old cache, portable RCCL mount; bash-n only passed,
no codegen/native build. All3239/3241 retained compile units present/hashed,
260 build-dir units/profile,103/105 include roots and38 library link rules/profile;
transitive header/ELF closure still separate.153 tests/nine new cases/static pass.
Portable tools4936B; metadata860,815B; kernel/artifacts/source-d52 and host tools/
source-d52/evidence/source-d52-* retain payloads/logs. Controllers7158/7744fail/
8899/10275done.2GiBAS/180CPU inherited/480wall/3GiBhost+5GiBdisk bounds; image export
EXboard+SHruntime/2CPU/no memory-cap claim. No GPU/native/default/kernel/app change.
D49SDK2a975370/boot2774d201r2 preserved,104docs20261110-d52-final;105 unchanged.
Next D53 matching external SDK/compiler/RCCL/toolchain payload+clean-build storage
admission. PLAN07/license/reproduction open/four packages remain; historical faults
remain separate/unfixed.

## D51 selected native build-input evidence (2026-11-10)

Implementation f9e45aace; [report](bc250_source_d51_results/README.md).
Eight D50 missing branches excluded for retained five-off-flag configs;0 family
references in3239/3241 compile entries+Ninja graphs, source CMake guards pinned.
No fetch needed for selected config; other-config recursive branches still open.
Original build image28ad36e6 distinct runtime1bb4ac0e; retained version/number/
arch exports explicit. Both image probes bind7tools/35vs49 Python inventories.
Actual CMake3.31.12 vs pip metadata4.4.0 resolved by selected binary hash; compiler
HIP hash measured now from retained RO build tree, no historical-byte claim.
Portable1,130,174B metadata/recipe/source-proof kit passes local/104 replay;
144tests/four new cases/three altered kits refused/static checks pass. No full
image/wheel/toolchain payload/license/clean-build closure claimed. Native
version Unknown preserved. Controllers32552/1044done, evidence/source-d51-r1/r2,
tools/source-d51. R2 CPU2.474s/1GiBnoswap/120s child/EXboard+SHruntime/noGPU.
D49SDK2a975370/boot2774d201r2 unchanged; native controls/apps/cache/builds preserved.
104docs20261110-d51-final,105unchanged; historical fault causes remain OPEN.
Next D52 budget original-build image export and fresh source/recipe assembly,
required generated/header/link input inventory; PLAN07OPEN/four packages remain.

## D50 portable recursive dependency contents (2026-11-10)

Implementation eb6147ea0; [report](bc250_source_d50_results/README.md).
D47 parent anchors37 direct pins;77 discovered locations/69 available/64 unique
commit payloads,75,522 blobs/107 license-copying-notice files. Eight unavailable
branches under aiter/flash-attention/mslk remain explicit and unexpanded; complete
source closure=false. Raw commit hash plus private exact tree replay binds content,
mode/symlink/gitlink/license identities. Local r1 catches Python tar trailing-slash
symlink normalization; safety-checked spelling preservation fixes it, full local r2
and104 replay pass. Gloo overlay private tree913f01fb/cudaSHA7dc2ce53 exactly retained
host source; TensorPipe/libuv included. Both CMake/version proofs retained, native
version git_version Unknown preserved rather than inventing a rebuild/source ID.
140tests/six new regressions/three altered exports refused/static checks pass.
258,723,840B portable package,82,862B metadata archive; tools/source-d50 and
 evidence/source-d50-r1; controller30288done,54.477s CPU replay/no GPU/native build.
2GiBAS/120CPU perprocess/420wall/100ms/3GiBhost+5GiBdisk reserves; no cgroup claim.
D49 runtime/boot2774d201r2 unchanged; no source/build/cache/prefix/app changes.
104docs20261110-d50-final,105unchanged. Historical Python/KFD/clock causes OPEN.
Next D51 classify eight branches/required pins and close build-version/toolchain/
wheel/generated-input inventory. PLAN07 OPEN/four packages remain.

## D49 native SDK repair activated on .104 (2026-11-10)

Implementation b6de91cb4; SDK143f1d8782; [report](bc250_sdk_d49_results/README.md).
Selected cohort sdk-d49-r1 preserves old miopen-plan02-r4 fallback. Only SDK
inode detached in27,905-entry hardlink tree; original unchanged. Baseline
runtime.json bytes retained plus explicit native-overrides.json SDK delta;
provider wrappers/registrations and D01 release/SDK metadata coherent.
Public/D09 helperd33f2504 admits HIP -1 only for pinned baseline/SDK/aliases,
rechecked under runtime lock; legacy guard retained. Normal loader, no preload.
12activation+2actualrollback+4reselection processes pass138 numerical checks,
20raw GPU traces/eight CPUnegative imports; GPU digests exact old controls,
CPU FP64 first-gradient/pinned-copy gates, strict unchanged logs/three ACKs.
Six provider registration identities pass; no new provider GPU/collective/model
qualification or performance claim.134local tests/three altered archives refused.
EX board+SH runtime tests, EX+EX selections,4GiB noswap/60s child/600s recovery/
3GiBhost+5GiBdisk reserves. SDK2a975370 active104only; olda6dc1c88 preserved.
Host tools/sdk-d49, evidence/sdk-d49-*; controller7740done/idle/no armed guard.
104docs20261110-d49-final;105unaccessed/unchanged/D43docs. Boot2774d201r2 unchanged.
Exact-1 SDK import crash repaired104, historical Python CPU/KFD/105clocksource
causes remain OPEN and distinct from fixed VMheap/D05/D21. PLAN07 OPEN/four
packages remain; next D50 recursive Torch dependency/license/Gloo/build inputs.

## D48 native SDK sentinel candidate qualified, inactive (2026-11-10)

SDKsource143f1d8782/helpers67e682956; [report](bc250_sdk_d48_results/README.md).
-1terminatesSDKvisibilitylist beforeCIFATAL, earlierselectionsretained; CIchecksON.
Oneunitccachebuild/retainedlink/privateoutputs/normalizedRPATH, originalhostbuild
RO. R1missedlink.d outputcaughtRO; R2redirectsfixed/cachehit. Candidate2a975370
17,575,272B/source5aa899d0;1450symbolcontract/DT_NEEDED/SONAMEequaloldSDK.
23finalfreshdaily/RCCLprocesses pass6negativeimports/CPU0devices,14candidateGPU
ordinal/default/termination,2normaloldcontrols +1oldpreloaddiagnostic.227numeric
checks,32ordinaryGPUtraces,ALLcandidatemathhashesexactoldcontrols, loadedDSObound.
Oldpreloadalsoemptiestraces; standardLDresolutionworks, noLD_PRELOADdeployment.
HSAUUIDGPU-XX unsupported;6plannedUUIDcasesunavailable, notsyntheticallyqualified.
131tests/3newbuildrefusals/independentrawtrace-provider-ELFreplay/3mutatedarchives
pass. Allstrictkernellogs unchanged/ACK/finalidle/nojobs,104boot2774d201r2.
NativeactiveSDK/D44refusal unchanged; source143fcommittedonlylocalcomponent,
originalhostsource/build/prefix/kernel/app/105preserved,cacheordinaryupdatesnoreset.
NextD49versioned104activation/EXprefixlocks/provideridentities/publicchecks/rollback;
keep105unchanged/legacyguard, public-1onlywithqualifiedinstalledSDKbinding.
PLAN07source/dependency/licenseclosureOPEN/fourpackagesremain. Specificimportfix
isqualifiedcandidate, notactive; unknownhistoricalPythonCPU/KFD/clockcausesunfixed.

## D47 portable Torch parent source and SDK admission (2026-11-10)

Source0bd04bc50; [report](bc250_source_d47_results/README.md).
Offlineparentpackage66,703,360B +standaloneGit/stdlibreplayer verifies21,491blobs/
37gitlinkidentities/mainLICENSE through EXACTbaseline/native/HEADtreehashes, no
alternates/sourcehistory/network. PrivateunbornHEADverificationrepo only, notbuild
setup.37directpinsavailablelocal/53licenseidentities; depscontents/27nestedlinks/
Gloooverlay/buildversion/toolchain/wheelsstillclosurework. OriginalshallowGitkept.
Workstation+104r1/r2pass; finalGit-envhardeningtested/oldrunskept. CPUperprocess2GiBAS/
120CPUseconds/affinity0–3/180wall, host3GiB/disk5GiB100msstopownedPG. NoGPU/build.
R2hostmin15.20GB/disk24.44GB(userfreedspace),12.398stestcost,noperfclaim.
SDK181explicitlinkinputs178host+3glibcimageverified,noGPU/1CPU256MiBread-onlyprobe.
Structuredargvavailable; neverexecuteunredirectedretainedoutputs. Candidate4GiBdisk/
4GiBRAM/jobs1/link1/ccache20G resourceenvelopefits; recheckactualadmissionbeforebuild.
128tests/4newintegrity-envcases/actualupdated-SHAtamperrefusal/staticchecks pass.
Alljobsdone/finalidle/noarmedguard/104boot2774d201r2; nativeprefix/source/build/cache/
SDK/default/kernel/app/105unchanged. D44refusalretained, SDKcandidateunbuilt.
NextD48 nativeSDKcandidatebuild/negativeimport/devicecount +validGPUbackward/profiler
qualification bothprofiles withloadedDSObinding/locks/recovery; activationseparate
EXprefixprocedure afterpass. PLAN07recursiveclosureOPEN/fourpackagesremain.
UnknownPythonCPU/KFD/105clocksource causesstillseparate/unfixed.

## D46 source reconstruction and SDK identity (2026-11-10)

Source4ea1bd17b; [report](bc250_source_d46_results/README.md).
Old3fileSDPAgap reproduced; separatecompleteTorchseries fromc779020 yields EXACT
nativea524980tree3e3edf7f andsourceHEAD8d8b18dtree33a86273. Privatefulltreereplays
pass; wrongtreepatchrefused. Existingoldseries/Githistories preserved.104focused35
pathreplaypasses;30rawnativepaths matchbothprofiles,5post-HIPifyC++ files differ.
Retainedreferenceoverlay reproduces5observedgeneratedfiles, notindependentcodegen
orpatchtobuildtwice. Piponlybuild/codegenretained.8componentGitpintreesavailable;
Torchshallow23commits/sharedobjects, portablefullsource/dependencyclosurestillOPEN.
SDKbcbb20f9→activea6dc1c88 EXACTrpath-onlybyte replay(.dynstr offset129047), code/
allothersectionsunchanged. CI=ON/-DROCPROFILER_CI=1;4relevantSDKfiles matchnative
1f9609fd6. Identityuncertaintyresolved; knownwarning-to-fatal mechanism nowbound.
Minimal-1terminatorproposal ONLYsourceapplychecked/notbuilt/notqualified/notinstalled;
D44refusal unchanged104only.124localtests/4newELFtests/static/scopedpatch/source
checks pass. No native/GPU/build/prefix/kernel/app/105change; alljobsdone/no guard.
.104admissiononly41,287,680Bover5GiBreserve; SDKDSO17,575,368B doesnotaloneestablish
compiler/link/cache/stage/rollbackbudget. NextPLAN07/D47 portableTorchsourcepack/
submodule/license/buildinputs +SDKcandidatebuildadmission. PLAN07OPEN/fourpackages
remain; historicalPythonCPU/KFD/105clocksource causesunchanged/unknown.

## D45 bounded mixed stability qualified (2026-11-10)

Source0ef00f158; [report](bc250_mixed_d45_results/README.md).
Six fresh104 processes daily/RCCL alternating pass162fixedCPUdouble references:
FP32matmul/conv/mathSDPA/realFFT output/loss/ALLinputgrad and2denseAdamWupdates
ALLparams/moments/counters each. Fixedatol5e-5/rtol5e-4,12updates/6CIDs, noapp.
Allstrictkernellogs unchanged; GPUpeak67,128,320B, sampled cgrouppeak932,622,336B,
minhost13,893,718,016B; sampledcontainerswap/OOMkill0. EXboard+SHruntime/4GiBcaps/
3GiBhost+5GiBdisk/60child/600guard, ACK/idle/alljobsdone. Foreign-arrival ACKafter
ownexit prevents rebootingunrelatedwork; branchuntested. Independentmetadatagates/
3alteredarchives/staticlintpass; no fullCI/longrun/performance/pagingIOclaim.
No nativeprefix/library/source/default/kernel/105change;104boot2774d201r2.
HistoricalPythonCPUIP0x3b3e6/KFDqueue-mm/105clocksource causesstillUNKNOWN/OPEN.
Newlead: profilerSDK warning mapsFATAL inCImode; retainedbuildCI=ON butbuiltDSO
bcbb20f9 !=activea6dc1c88, exactprovenanceunresolved. D44refusal remains104only;
no SDKnativefixclaim. PLAN06boundedcomplete; nextPLAN07/D46 source/artifactaudit,
oldSDPAthreefilegap andSDKCI/build/packaging mapping before anyrepairbudget.
Fourimportantpackagesremain, optionalAPIexpansion separate.104~98MiB over5GiBdisk
reserve, do not begin nativebuild without explicitresource/storage plan.

## D44 specific HIP visibility safeguard deployed on .104 (2026-11-10)

Source9d03c5c90; [report](bc250_visibility_d44_results/README.md).
Old HIP_VISIBLE_DEVICES=-1 import reproduces139: profilerSDK agent.cpp invalid-token
fatal/abort then Python segfault/libc general-protection fault. Shared launcher now
refuses exactly that input before elevation/container; inherited/explicit/envfile/
context checks. Other filters unchanged, not blanket-qualified.12CPU-double GPU
loss/backward+CPU-first checks across daily default/0 andRCCL0,120localtests and
3alteredarchive refusals pass. Actual UID1000 public torch/train exit2, distributed
existing tracebackexit1 before doas/worker.104public/sharedD09 helperSHAa7b3a5b8;
oldc5167f17 saved evidence/visibility-d44-install-r1. EXboard+EXruntime install;
GPUtests EXboard+SHruntime/600guard/60child/4GiBcaps/3GiBhost+5GiBdiskreserves.
Allcandidate kernel logs unchanged; oldbaseline newlibcfault preserved separately.
R1 extensionless-loader testbug retained, authoritative r2 fixedexplicitloader.
Independent archive replay pass, allownedjobsdone/guardACK/idle. No native rebuild,
parserrepair/registry/profile/kernel/reboot/app/105change. HistoricalPythonCPUIP0x3b3e6,
KFDuncertainty/105clocksource timeout remainOPEN, distinct fixedVMheap/D05/D21.
PLAN06OPEN; next D45 predeclared sixprocess smallmixed output/backward/AdamW
stability pass on104 (report has exact bounds), no unknown-cause repair claim.

## D43 fresh1.024B training capacity completed (2026-11-10)

[Report](bc250_distributed_d43_results/README.md), sources8a590d02f/4f684e984/31aa4415d.
Two native FP32 FSDP ranks complete8updates at1,024,471,040params/seq128/global2,
without rawCPUreference reads/fullstate gathers in timed updates. D42 exact shape/
rank/source proof prerequisite; firstlossparity,8finitegradientnorms, ALL final local
GPU params/gradients/moments finite/counter8. Eachrank512,235,520localstateelements.
Two genuine second-update reserve stops drove forward and public backward cache
hooks (returnNone/gradients preserved); unused pages only/live bytecounts unchanged,
no lowered reserve/math bound or native optimizer change. Control8norms/5losses/
statecounts EXACT r1 BOTH. Large median53.365s; liveGPU8587.9/8585.4MiB, sampledhost
min3.620/3.282GiB; complete guardedjob~455s. Explicit capacity benefit, no speedup.
Container swap/OOM0;105hostswapbaseline2,756,608B pre-existing/unchanged (104zero),
pagingIO unmeasured. Baseline-aware replay refuses occupancy growth/container swap;
no claim no swapIO. Initial r2 control Docker-veth log refusal preserved/rechecked.
24groups/4finalACKs/116localtests/4archive mutations/lintpass. Finalr4metadata exports
(r3 retained), all oldattempts/sources kept; guards/ownership/resourcecaps unchanged.
Native prefixes/kernels/fallbacks/apps unchanged; exportidle/docs syncedBOTH.
Later handoff104idle,105 new unrelated Docker workload (preserved); check ownership
before next GPU task.105clock
remote-read warning stillOPEN; PythonCPUfault separate fixedVMheap/D05exit/D21load.
Single allresidentFP32statefloor16.392GB>RAM; no deliberateOOM or AMP/offload
impossibilityclaim. No longrun/restart/generalmodel claim. PLAN05 bounded capacity
scope COMPLETE, FIVE importantpackages remain. NextPLAN06/D44: inspect retained
fault/visibility evidence/current launcher; bounded invalid-selection reproduction
and prelaunch refusal/diagnostics repair, then declared stability pass. Older next
sentences below are historical; do not repeat completed kernel/capacity work.

## D42 1.024B source-bound GPU audit completed (2026-11-10)

[Report](bc250_distributed_d42_results/README.md), sources48f58f74c/f9f9ab563/4582251e3.
Two native gfx1013 FP32 FSDP ranks pass ALL244parameter loss/raw/norm/clip/updated/
localAdamW moment+step gates vsD41 FP64oracle at1,024,471,040params. Identicalrank
raw/clip/updatedhashes; fresh352.96Mcontrol nativehashesexactD39/D40. Oneupdate only.
Two safely stopped attempts exposed unused GPUcache and CPUoracle scratch pressure:
phase-boundary empty_cache preserves liveallocations, CPUAdamW reference1Mchunks/
1139chunks/ALLvalues/16MiBmomentpair, syntheticchunk/fullbitexact proof. No relaxed
reserves/bounds/nativeoptimizer/defaultchanges. GPUliveaudit~8265.7MiB, hostmin
3.249GiBguardian, swap/OOM0. 24groups/4ACKs/108localtests/4archive refusals/lintpass.
CPUready600s beforeGPUinit, nativePG120s unchanged; isolatedguard1800/child1400,
originalkernelguard600 unchanged.104boundedread-onlysource→105128MiBreader/no cache/
spool/fallback;49.175GBrawtransport includes4readseach, refs104only. Source/jobsgone.
104rebootaftercompletedtest, causeunknown; separateexport/testbootIDs verified.
105r4 clocksource remotereadtimeout remainsOPEN (pr_info_once/retry, quietlogs do
not prove recovery); distinct historicalPythonCPUfault/fixedVMheap/D05/D21. Native
prefixes/kernels/fallbacks/apps unchanged; currentdocs syncedBOTH. PLAN05stillOPEN.
NextD43: fresh training-only1.024B capacity+memory/shortmultiupdates+timings with
D42proofbound, nofullreference audit inworker; maintainownership/reserves/recovery.
Single all-residentFP32statefloor16.392GB>RAM; no deliberateOOM or broaderAMP/offload
impossibilityclaim. Earlier D41 next/status sentences below are historical.

## D411.024B CPU oracle completed (2026-11-09)

[Report](bc250_distributed_d41_results/README.md), producer3afa04930/finitef75cb361e.
104ONLY new tools/D41 CPU-only1,024,471,040-param FP64 GPT2/20blocks/2048width,
seq128/global2/dropout0/eager. D35 construction/math reused;352M freshcontrol ALL
loss/initial/gradient hashes exact. Native checkpoint false-reentrant/preserveRNG,
21modules/20→40blockentries;244hooks once/finite/128MiB roundtrip/0retainedgrads.
CPUloss9.416439180620552; initialFP64/RNGpreserved. NoGPUdevices/CUDAinit/PG/agent/
optimizer. CPU9.758GiB planned, soft10/hard11/combined12GiB, hostreserve3GiB/disk5GiB,
100ms owned-stop guard; actualcgroup9.051GiB/hostavailable5.365GiB, swap/OOM0.
488rawfiles11.449GiB now on104 evidence/distributed-d41-r1/large-oracle/reference,
manifest99b36b92; originalD35refs preserved/no105copy. Finaldisk5.424GiB. R2 final
metadata rehashes old/new and independently scans ALL initial/gradient finiteness
+SHA in1MiBchunks.16CPUgroups/2ACKs/100localtests/3archive refusals/lint pass,
logs/registry/boots unchanged/finalidle/docs syncedBOTH,105docs only. Native/defaults/
apps unchanged; PythonCPUfault distinct VMheap/D05/D21. GPU1.024B stillunqualified.
NextD42: GPU/source/RAM andstartup budget beforetwohostFSDPaudit; remote2×12.294GB
verification needs≥197s on1Gb/s, beyondold120sPGtimeout. Add readinessbarrier or
reviewedboundedtimeouts/guards, keep originalmathgates/oldrefs, no deliberateOOM.

## D40 bounded read-only reference transport completed (2026-11-09)

[Report](bc250_distributed_d40_results/README.md), code470c4046b; additive tools BOTH.
Same352.96M FSDP/D35 refs/D39 gates.104 guarded UID1000 source serves only manifest/
296 listed blobs to105,1MiBchunks/256MiBAS/64FD/no writes or directory listing.
Remote105 descriptor only68,123B manifest; allblobstatus/size/SHA checks,96MiB max
retained blob/1MiB verification buffer, zero diskcache/silentfallback. Source1185
requests/16,941,957,120B rawpayload, eachblob4reads/no failures; source stoppedbeforeACK.
ALL native raw/clipped/updated/localAdamW hashes EXACT localcontrol/D39/D36; bounds
unchanged.24groups/4ACKguards/93localtests/3archive refusals/lint pass. Worker207svs
121.5local includes network/checks, no training speed/largefit claim. GPU peaks same,
cgroup≤2515.4MiB/swap+OOM0/logs+registry+boots unchanged/final idle/docs synced BOTH.
Original D35 localrefs retained, no new fullcopy. Native defaults/apps unchanged.
PythonCPUfault distinct fixedVMheap/D05/D21. PLAN05OPEN. NextD41: source/config/RAM/
11.449GiBpayload+5GiBreserve admission then CPU-only1.024B checkpoint+stream oracle
on104; bounded headroom/abort/ALLfinite+hash+coverage checks, no GPU admission yet.

## D39 spool-free CPU-reference clipping completed (2026-11-09)

[Report](bc250_distributed_d39_results/README.md), codea82aca0ba; additive tools BOTH.
Same352.96M native FSDP fixture/D35 CPU refs. ALL clipped gradients checked directly
against FP64 CPU ref×checked coefficient at unchanged1e-8/5e-5; maxabs1.30094e-8
passes combined bound. CPU coefficient still uses checked actual-gradient norm.
ALL raw/loss/norm/AdamW gates retained; raw/clipped/updated/local-state hashes EXACT
fresh legacy control/D36. Legacy route/refs preserved. Raw-gradient spool1.315GiB→0
perhost,5GiB reserve retained; no training/default change. Live GPU audit peaks
unchanged2980/2974MiB, swap/OOM0/logs+registry+boots unchanged/final idle/docs synced.
24groups/4ACKguards/86localtests/3 archive refusals/lint pass; no speed/largefit claim.
PythonCPUfault distinct fixedVMheap/D05/D21. PLAN05OPEN. NextD40 qualifies bounded
read-only per-tensor reference transport104→105 at352.96M before1.024B oracle/RAM/
storage/GPU admission;11.449GiB fullrefs+5GiB reserve still cannot fit105.

## D38 CPU checkpointing+gradient streaming completed (2026-11-09)

[Report](bc250_distributed_d38_results/README.md), code22d702e47; additive D38 tools
BOTH. Native nonreentrant/preserveRNG CPU checkpoint partials/13 module guards,
12blocks reentered during backward (24entries vs12 control). Same352.96M FP64
oracle/full-global backward/dropout0/cacheOFF; ALL initial/loss/148 gradient bits
EXACT D35/D37/modes/hosts, one hook each/zero retained grads. Source checkpoint.py
eab49639 matches qualified Torch; CPU RNG unchanged in dropout0 fixture. CPU only,
no native GPU/default change. Cgroup5208.641→4179.109MiB104/5196.531→4249.711MiB105
(18.2–19.8%lower vsstream), hostavailable9.237→10.261/9.132→10.033GiB. Same96MiB
scratch+5GiB reserve/refs preserved.32groups/4ACKs/86localtests/3corruption refusals,
swap/OOM0/logs/registry/boots unchanged/final idle/docs synced BOTH. No speed/largefit/
stochastic/longrun claim. PythonCPUfault distinct VMheap/D05/D21. PLAN05OPEN;
nextD39 removes full actualGPU gradient spool via independent CPU clipped refs at
unchanged gates, qualify352.96M before1.024B oracle/storage/GPU admission.

## D37 CPU gradient streaming completed (2026-11-09)

[Report](bc250_distributed_d37_results/README.md), code2610b2b76; additive D37 tools
BOTH, final r2. Native post-accumulate CPU hooks validate ALL148 FP64 gradients
against D35 before clearing p.grad, exactly one hook per unique/tied parameter;
after-backward retained elements352957440→0. Initial FP64 values/loss/ALL gradient
bytes EXACT across control/stream/hosts/D35. One96MiB scratch file+5GiB reserve;
no full reference copy/spool, original raw references retained. CUDA uninitialized,
CPU checkpointOFF/no optimizer/PG/GPU tensor. Cgroup7556.621→5218.113MiB104 and
7481.609→5202.461MiB105 (~30.5–30.9%lower); hostavailable7.033→9.212/6.984→9.109GiB.
Streaming slower complete phase76vs57s due I/O/checks, no training-speed claim.
24finalgroups/4ACKs/83localtests/3corruption refusals/lint pass; preliminary r1
weight-comparison tightening retained, corrected controls rerun before streaming.
Logs/registry/boots unchanged/swap+OOM0/final idle/docs syncedBOTH; no native/app/
default change. PythonCPUfault distinct fixedVMheap/D05/D21. PLAN05OPEN. NextD38:
qualify native CPU nonreentrant checkpointing+streaming against existing352.96M
refs before1.024B oracle/storage/clipping-spool admission; larger shape still blocked.

## D36 per-unit FSDP qualification completed (2026-11-09)

[Report](bc250_distributed_d36_results/README.md), code22447a4a9. Additive D36 tools
BOTH, existing D35 352.96M math/oracle retained. Public per-wrapper recurse=False
summons/explicit whole-call refusal/other matrices stay sharded. ALL148 names in
root+12block units,39 contexts across3 passes/maxunit28331520 vswhole352957440.
CPU ALLgrad/loss/norm/clip/AdamW gates unchanged; raw/clipped/updated digests EXACT
across paths/hosts, local moment/counter digests EXACT perrank. Report-only CPU norm
sum-order delta3.6e-15 accounted separately, not a numerical-gate relaxation.
Audit peaklive5456.828→2980.129MiB104/5450.629→2973.930MiB105 (~45.4%lower);
hostavailable5.260→7.838/4.920→7.246GiB. Qualification peaks exclude initial model
construction; no training-memory/speed/large-fit claim.24groups/4ACKs/79localtests/
3metadata corruption refusals, logs/registry unchanged/swap+OOM0/final idle. No
native/default/app/kernel change; docs synced BOTH. CPUfault distinct VMheap/D05/
D21. PLAN05OPEN,1.024B blocked oracle/storage. NextD37: CPU gradient post-accumulate
streaming versus existing352.96M refs, exact coverage/hook count/peaks, only one
parameter of scratch before larger CPU/GPU admission.

## D35 staged CPU references /352.96M completed (2026-11-09)

[Report](bc250_distributed_d35_results/README.md), code02107159f. Additive tools/
distributed-d35 BOTH. Separate CPU-only FP64 oracle publishes3.9446GiB of hashed
raw FP32 initial/FP64 gradients per host, exact ALL148 roundtrip/identical manifests;
no CUDA initialization. GPU workers read references per parameter, temporary actual
FP32 gradients spool to disk for ALLclip checks then are removed after success;
original references retained. CPU FP64 ALLgrad/loss/norm/clip/streamedAdamW ALLvalues/
local moments/counters pass unchangedD32 gates.352957440 master/moment elements
single/176478720 each FSDP rank. Fresh bothorders loss parity≤1e-5;72groups/14ACKs/
76localtests/3archive corruption refusals. Single265.535–266.563ms vsFSDP17968.643–
17973.422ms (~68xslower), liveGPU5863.493vs3183.965–3189.670MiB (~45.6%lower).
Paired wholehost FSDP headroom+0.927–0.956GiB; both fit, no larger-than-single proof.
CPU cgrouppeak≤7550MiB/GPUqualification≤2504MiB, hostavailable≥4.932GiB/swap+OOM0.
No native/default/kernel/app changes; guards/logs/registry/fallbacks retained/final
idle/docs syncedBOTH. PythonCPUfault remains distinct VMheap/D05/D21. PLAN05OPEN.
1.024B CPU/GPU blocked:15.266GiB fullCPUoracle/singleFP32state floor exceeds RAM,
.105 lacks unchanged reference/spool+5GiB storage reserve. NextD36: qualify bounded
per-FSDP-unit ALL audit on existing352.96M refs, then streamedCPU/storage feasibility.

## D34 bounded159.8M GPT2 qualification completed (2026-11-09)

[Report](bc250_distributed_d34_results/README.md), implementation58e91cb5d.
Additive tools/distributed-d34 BOTH. Shape-specific CPUFP64 ALL148grad/loss/norm/
clip/streamedCPUAdamW ALLvalues/localmoments/counters pass unchangedD32 gates;
159,807,488 FP32 eager tiedGPT2 params/global2/micro1/seq128/positions256.
No fullCPUoptimizer/export; moment scratch≤128MiB, whole qualification cgrouppeak
≤4991.34MiB/hostavailable≥5.811GiB/swap+OOM0. Fresh digest/source/runtime-bound
workers omit CPU/full audits. Bothorders pass64groups/12ACKguards/71localtests+
3privatearchivecorruptionrefusals. Single150.288–151.092ms vsFSDP8088.982–8103.416ms
(~54xslower); liveGPU2737.507vs1493.502–1495.365MiB (~45.4%lower),master/moment
ownership159807488vs79903744/rank. Measuredcgroup~1082vs1648MiB,hostavailable~10.1GiB
both: no clearwholehostcapacitygain. Never sum overlappingcounters. Initial/later
losses match across modes/hosts/orders≤1e-5; not convergence/restart/longrunproof.
No nativebuild/reboot/prefix/default/app changes; guards/registry/logs/fallbacks
preserved/finalidle/docs syncedBOTH. HistoricalCPUfault stays separate fromfixed
VMheap/D05/D21. PLAN05OPEN;352.96M/1.024B not admitted. NextD35 stages CPUgradient
artifacts with source+storage/memory guards before352.96M GPUqualification.

## D33 fresh training-only memory controls completed (2026-11-09)

[Report](bc250_distributed_d33_results/README.md), implementation `199d3f2ab`.
Additive tools/distributed-d33 on BOTH hosts. Immutable host/mode proof JSON binds
published D32 archive/source/registry identities and prior CPU loss/ALL gradients/
clip/AdamW qualification; tampering refuses before allocation. Fresh containers/
UUIDs reuse identical D32 GPU build/data/backward/optimizer math but retain no CPU
base duplicate and forbid CPU oracle/full-state audit entry points. Same45.9M FP32
GPT2/global8/micro1/seq256, initial update+two warmups+five measured, both orders.
Initial/five measured losses match D32 within1e-5; prior CPU checks are reused,
not counted as new CPU executions. Master/AdamW ownership and traffic match D32.
Timing:single247.897–249.549ms,DDP1680.839–1681.057,FSDP6879.746–6882.628.
Live GPU peak940.725/1114.484/767.607–769.358MiB. Fresh measured cgroup maxima
1081/1872/1646MiB versus D32's2100–2332/2792–3048/3260–3448. This corrects the
measurement baseline; it is not a native Torch/application memory optimization.
New memory.stat/RSS/GPU stage snapshots identify accounting categories, not exact
library ownership. DDP shmem endpoints458→514→530→530→530MiB reproduce both orders/
hosts; final three plateau in this short run, no long-run/leak claim. FSDP stays
near304MiB shmem after initial GPU update. Whole-worker cgroup peak≤1874.44MiB;
host available≥10.625GiB, host/cgroup swap0/OOM0. Counters overlap; do not sum them.
CPU-only larger-shape planner records159.807M/352.957M/1.024B parameter/state floors
with exclusions. ALL candidates GPU_admitted=False; no fit/speed/convergence proof.
84 fresh groups/12 ACKed guards/68 local tests/lint pass; no failed GPU phase.
PLAN-01 locks/resources retained, logs/registry unchanged/final idle/no jobs/guards.
Boots2d86d523r2/e902e636r3 unchanged/native providers/defaults/fallbacks preserved;
no build/reboot/download/prefix mutation/apps/foreign stop. Docs synced BOTH hosts.
Historical Python CPU fault stays distinct from fixed VM heap/D05 exit/D21 load.
PLAN-05 OPEN. Next D34: memory-bounded initialization and shape-specific correctness
for159.8M, then separate fresh single/FSDP/global2/micro1/seq128/positions256 checks;
avoid a co-resident full CPU optimizer/moment export, keep larger shapes blocked.

## D32 representative capacity/timing baseline completed (2026-11-09)

[Report](bc250_distributed_d32_results/README.md), implementation `6af433cb8`.
Additive tools/distributed-d32 installed on BOTH hosts. New conservative admission
and owned eager GPT2 fixture: 45,871,104 all-trainable FP32 parameters, tied head,
six blocks/width768/12heads/vocab4096/sequence256/global8/micro1, dropout0/cacheFalse,
AdamW and root clip.5. DDP/FSDP accumulate four local micros with native no_sync;
single-board workers process eight. CPU FP64 global loss/ALL76 gradients and CPU
root norm/ALL clipping/ALL AdamW checks pass unchanged D05/D21/D22 bounds.
Two warmups+five timed updates per fresh worker; both mode orders pass. Later
finite losses match across modes within1e-5, not a convergence or replay proof.
Single .104:247.963/248.057ms; .105:249.278/249.128ms. DDP:1681.584/1682.027ms;
FSDP:6885.565/6885.373ms. Both physical eth0 links report1000Mb/s. DDP input is
183,484,416B/update; FSDP gathers707,186,688B plus reductions183,484,416B, not wire
bytes. FSDP master/AdamW ownership halves to22,935,552 elements/rank.
Live GPU peak:940.725MiB single,1114.484DDP,768.366–770.856FSDP. However, reserved
GPU memory and measured cgroup usage rise for FSDP. Lower GPU allocation does NOT
prove physical RAM saving or larger fit. The timed worker already performed CPU
validation; CPU/runtime/communication contributions are not separately attributed.
Whole-validation cgroup peak≤4745.49MiB, host available minimum8.635GiB including
CPU audit/9.27GiB during measured training; host/cgroup swap and OOM counters zero.
Seven-update segment:~1.85–1.98s single/11.87–12.89s DDP/48.19–48.49s FSDP;
guarded phase33–34/53–55/93–94s includes CPU/import/agent work, not production time.
72 groups/12 ACKed guards/64 local tests/lint pass; no failed GPU phase. Admission
requires host≥10GiB and device free≥4GiB before model allocation; no OOM search.
PLAN-01 locks/resources retained, logs/registry unchanged, final idle/no jobs or
armed guards. Entry boots2d86d523r2/e902e636r3 unchanged; providers/defaults/fallbacks
retained. No native build/reboot/download/prefix mutation/app inspection/foreign
stop; docs synced BOTH hosts. Historical Python CPU fault remains distinct from
fixed VM heap/D05/D21. PLAN-05 OPEN. Next D33: source-bound CPU qualification then
fresh training-only memory controls without preceding oracle/audit work; establish
physical headroom before larger FSDP initialization/admission. Larger fit, speed,
other-model accuracy, convergence and AMP/PEFT are not qualified by D32.

## D31 combined BF16 PEFT checkpointing completed (2026-11-09)

[Report](bc250_distributed_d31_results/README.md), code725e495c8, BOTH additive
tools/distributed-d31. New BC250BF16CheckpointPeftFSDPTrainer combines D29 native
nonreentrant/RNGpreserving recompute with D30 BF16working/FP32master-reduce-buffer/
nativeautocast/noscaler; namedPEFT/frozen/AdamW/rootclip/fullsave/export inherited.
DefaultD30precision/runtime predicates factored, checkpointOFFdefault remains;
oldinstalledhelpers untouched. PinnedplainLoRA/eagerGPT2/dropout0/cacheFalse/
FP32originalparams/checkpointinitiallyOFF; bothrecipes bound+runtimeguarded.
Same24352=21280frozen+3072trainable/19units/12176shards/1536AdamW each. CPUrounded
BF16+FP64 ALLadaptergrad/loss/CPUclip/rootnorm/ALLAdamW originalbounds pass,
maxglobalrelative.00760193/.00783804; ALLfrozenmasters EXACT. ActualBF16compute/
gathers+FP32reduce/master/optimizergrad; GPUblock14calls/7seedmicros,6/3replay,
offcontrol7/7 provesrecompute. Matchedcontrol ANDhashverifiedD30archives haveEXACT
same transcript/finaldigest. BF16gathers259→357seed/111→153replay;16reducecalls/
12288B/window retained. Tinycompleteharness peak459264vs499200B/rank(8%lower),
not capacity/speed/wholehostproof. FullCP2union6files480247B/newseed+UUID+containers
restoresALLmodel/AdamW/scheduler/fourRNG EXACT; windows3/4 ALLloss/grads/updates/
counts/digests exact. Publicadapterloads exact/RNGunchanged. Three rank1grad
mutations ANDpostclipLIVEcheckpointRNGoption drift BOTHrefusebeforeAdamW/zero/
unchangedstate; private rank1saveduse_reentrant corruption BOTHrefuseoptimizer
restore/originalbytesintact/discardabortedsession. RetainedD30BF16control passes.
163groups/18ACKguards/30CPUpatterns536obs/62localtests/lint pass/no failedGPUphase.
6c4758c18 pinsimmutableD30/D31archivehelperhashes instead ofmutablecurrentbytes;
botharchivereplays yieldbyteexactoriginalverifiedJSON/finalverifiersinGit+D31
namespace(aftermetadataexport), noGPU/mathchange/hashweakening.
Initiallocaltempquota +7devshmmounttesterrors resolvedwithownedworkspaceTMPDIR;
mountguardsunchanged/no cachesdeleted. PLAN01resources/locks retained,logs+registry
unchanged/finalidle/nojobs/guards. Entry2d86d523r2/e902e636r3 unchanged;nativeprovider/
package/defaults/fallbacks retained,no build/reboot/download/prefixmutation/apps/
foreignstop;docssyncedBOTH. HistoricalPythonCPUfault distinctfixedVMheap/D05/D21.
PLAN04 boundedmemoryfeatures COMPLETE. NextPLAN05/D32 conservativemeaningfulowned
model oneboardvsDDP/FSDP,matchedbatch/sequence/precision/RAMswapallocatorheadroom/
warmstep+wholejobtiming; no intentionalOOM. FP16/otherrecipes remain separate.

## D30 explicit BF16 PEFT FSDP completed (2026-11-09)

[Report](bc250_distributed_d30_results/README.md), codee8c6d120c, BOTH additive
tools/distributed-d30. New BC250BF16PeftFSDPTrainer requires plain pinnedPEFT/
eagerGPT2/dropout0/cacheFalse/checkpointOFF/BF16working+FP32master/reduce/buffer/
nativeautocast/noscaler. FrozenAMPrefusal factored; defaultoldhelpers stillrefuse.
Actual args/plugin/no-modelcheckpoint guards bind precision+modelingpins to full
checkpoint. D28 partitions/AdamW/rootclip/fullsave/publicadapterexport retained.
Sharedoracle now checks ALLtrainable adaptergradients for frozenAMP, preserving
old alltrainable/FP32 behavior. Same24352=21280frozen+3072trainable/19units/
12176shards/1536AdamW each. BF16actualweights/output/gathers,FP32reduce/master/
optimizergrad observed; ALLfrozenvalues EXACT. CPUroundedBF16+FP64 ALLadaptergrad/
loss pass oldD25bounds(.02/.03normgates+floor),maxglobalrelative.00760193/.00783804;
CPUALLclip/rootnorm/CPUdoubleALLAdamW pass originalD21/D22bounds. Seed259gathers/
identicalelementsequence toD28,input623168→311584B; reduction16calls12288B/window.
Not totalNIC/speed/memorycapacity. FullCP2union6files480247B/freshseed1773+newUUID/
containers restoresALLmodel/AdamW/scheduler/fourRNG EXACT; windows3/4 ALLloss/grads/
clip/updates/counts/digests EXACT. Publicadapterroundtrip exact/RNGunchanged.
Fourrank1gradient/LoRAmutations BOTHrefuse/zeroAdamW/unchangedstate; private rank1
savedreductionprecision corruption BOTHrefuseoptimizerrestore; originalbytes
intact/discardabortedsession. RetainedD28FP32control passes.116groups/16ACKguards/
28CPUpatterns444obs/62localtests/lint pass/no failedGPUphase. PLAN01locks/resources
retained/logs+registryunchanged/finalidle/nojobs/guards. Entry2d86d523r2/e902e636r3
unchanged; nativeproviders/packages/defaults/fallbacks retained/no build/reboot/
download/prefixmutation/apps/foreignstop. Docs syncedBOTH. HistoricalPythonCPUfault
separatefixedVMheap/D05/D21. PLAN04open; nextD31 explicitBF16+nonreentrantcheckpoint
combination/CPUALLgrad+AdamW+frozenpreservation+exactrestart beforeadmission; FP16
and capacity separate. No universalmodel/speed/wholememory/stochastic/elasticclaim.

## D29 explicit FP32 PEFT activation checkpointing completed (2026-11-09)

[Report](bc250_distributed_d29_results/README.md), code8e04fa9b4, BOTH
additive tools/distributed-d29. New BC250CheckpointPeftFSDPTrainer requires
explicit nonreentrant/RNGpreserving boolean kwargs, pinned native GPT2 modeling
sources/eager/cacheFalse/checkpointing initiallyoff. Native Trainer enable/embedding
output-grad hook/block calls retained; D28 fullsave/export/FP32frozen/AdamW/rootclip
contracts inherited. Actual checkpointfunctions/options guarded+checkpointbound.
Old helpers/frozenAMP/pluginactivation/FSDP2/compile/offload restrictions retained.
D28 owned24352=21280frozen+3072trainable/19units/12176shards/1536AdamW each;
CPU ALLadaptergrad/loss/rootnorm/ALLclip/CPUdoubleALLAdamW unchangedbounds pass;
ALLfrozenvalues exact. GPUblocks14calls/7micros(seed),6/3(replay),control7/7:
actual backward recomputation. Checkpointoff transcript/finaldigest EXACTsame.
FullCP2 union6files480247B/newseed+UUID+containers restores ALLstate/fourRNG EXACT,
windows3/4 ALLgrads/loss/updates/counts/digests exact. Publicadapterroundtrip exact.
Fourrank1gradient/LoRAmutations BOTHrefuse/zeroAdamW/unchangedstate; rank1private
saveduse_reentrant corruption BOTHrefuseoptimizerrestore; originalfiles preserved.
Tinycompleteharness peak491008B vs546304B/rank(10.12%lower),not isolatedactivation/
largefit/speed. Gathers259→357seed/111→153replay,16reductions12288B/window;
not totalNIC.163groups/18ACKguards/28CPUpatterns500obs/62localtests/lint pass;
no failedGPUphase. Initialpermission/prematurelaunch setupfail preserved beforeGPU.
PLAN01locks/resources retained; logs/registry unchanged/finalidle/no jobs/guards.
Entryboots2d86d523r2/e902e636r3 unchanged; nativeproviders/packages/defaults/fallbacks
unchanged/no build/reboot/download/prefixchange/apps/foreignstop. Docs syncedBOTH.
HistoricalPythonCPUfault distinctfixedVMheap/D05/D21. PLAN04open; nextD30 explicit
BF16 PEFT frozenbase with checkpointingOFFfirst/CPUALLgrad+AdamW+exactrestart;
retain frozenAMP refusal untilqualification. CombinedBF16checkpoint/FP16 separate.

## D28 actual PEFT LoRA FP32 FSDP completed (2026-11-09)

[Report](bc250_distributed_d28_results/README.md), e08c98d15, BOTH
tools/distributed-d28. New BC250PeftFSDPTrainer/peft_fp32_policy pins PEFT0.18.1/
Transformers5.2/Accelerate1.12+nine sources, single default plain unmerged causal
LoRA/FP32/no dropout/bias/variants. Inherits frozen policy/AdamW/clip guards;
checkpoint binds adapter/scaling/base config/attention. Native PEFT auto-wrap
19units retained; public Accelerate FULLmodel save/load replaces adapter-only
checkpoint selection for this explicit route. Native Trainer loop/RNG/state and
D21/D27 optimizer/policy retained. Public PEFT canonical adapter export CP2/4/
replayCP4 loads ALLparams/logits EXACT on matching CPUbase;14,216B safetensors,
RNG unchanged. R1 wrapper-prefixed export missingkeys caught by strict load;
R2 canonicalized/pass, R3 final recipe guards +selected reruns. No relaxedbounds.
Owned GPT2 tied frozen head/8 PEFTConv1D adapters r4alpha8/zeroB,24352unique=
21280frozen+3072trainable; shards12176each/AdamW1536each. D26 shiftedtargets/
localzero micro, CPUFP64 ALLadaptergrad/loss/clip/CPUdoubleALLAdamW at existing
FP32 limits pass; ALLfrozenvalues exact. Actual FP32gathers/reductions observed,
16reductions/window12288B input, no totalNIC/speed/memory claim. Fourrank1gradient/
scaling mutations BOTH refuse/zeroupdates/unchangedstate; rank1-only private
checkpoint LoRArecipe corruption BOTH refusesoptimizerrestore while generic
frozenhash unchanged. Originalfiles intact. Retained D27 native-adapter control
passes. Full CP2 union6files480247B; freshconstructor1773/container/UUID restores
ALLbase+adapter+AdamW/scheduler/fourRNG EXACT, windows3/4 ALLgrads/loss/state/counts/
digests EXACT.116groups/16ACKguards/26CPUpatterns408obs/62localtests/lint pass;
2failedR1+2passedR2 retained/all20ACK. Nativeproviders/prefix/packages/defaults/
apps/fallbacks unchanged; entry2d86d523r2/e902e636r3, no agent reboot/build/download/
foreignstop. Logs/registry unchanged/finalidle/no jobs/guards/docs synced.
HistoricalPythonCPU fault separatefixedVMheap/D05/D21. PLAN04 open; nextD29
explicit non-reentrant FP32 PEFT activation-checkpointing/CPUgrad+restart+bounded
allocator comparison. FrozenAMP/variants/elastic/capacity remain separate.

## D27 explicit frozen FP32 FSDP policy completed (2026-11-08)

[Report](bc250_distributed_d27_results/README.md), code80080e6a0, BOTH
tools/distributed-d27. New BC250FrozenFSDPTrainer/frozen_fp32_policy captures named
immutable FP32 trainable/frozen split, cross-rank signatures/shapes, positive root
clip, trainable-only AdamW/state and pre-update gradient/policy guard. Default
BC250FSDPTrainer still refuses frozen; prior installed helpers preserved. Native
checkpoint JSON binds policy/shapes; rank1-only private corruption makes BOTH
refuse before optimizer restore, zero updates. Owned eager GPT2/tied frozen head+
8native rank4 Conv1D adapters:24352unique=21280frozen+3072trainable. Fullshard/orig
root+2blocks,12176param/rank,1504/1568trainable+AdamW. D26 shifted49/57/57/21 and
localzero-targetdata; CPUFP64 ALLadaptergrad/loss/rootclip/CPUdoubleALLAdamW at
unchanged FP32 bounds pass; ALLfrozen VALUES exact. Frozen original sharded grad
None; native full-gradient inspection materializes flat-buffer views, not AdamW
membership. R1 probe assumption failsbeforeupdate, R2 contextfixed/pass; R3final
policyhardening/save+loadguards/fullselectedtests pass.80640B reduceinput/window
shows mixed-flat/padding cost, no proportional whole-memory/traffic/speed claim.
Three rank1 gradient mutations BOTH refused/zeroupdates/unchangedstate. Retained
alltrainableGPT2 FP32 control passes. CP2 DCPunion6files469453B/all originalhashes
preserved; freshconstructor1773/newprocess/container restoresALLstate/fourRNG,
windows3/4 ALLgradient/loss/clip/AdamW/RNG/counts/digests EXACT.97finalgroups/
14ACKguards/22CPUpatterns294observations/62localtests/lint pass,2failedR1+2passedR2
retained/all18ACK. No provider/build/prefix/default/package/app changes/reboot/
foreignstop. PLAN-02 native cohort retained; .104entryafcc7be4r2/.105entry570437edr3,
logs/registry unchanged/finalidle/nojobs/guards/docs synced. HistoricalCPUfault
separatefixedVMheap/D05/D21. ActualPEFT/frozenAMP/checkpointing/elastic/capacity
remain separate. NextPLAN04/D28 actual pinned PEFT FP32 LoRA/FSDP admission.

## D26 actual Trainer causal-LM FSDP completed (2026-11-08)

[Report](bc250_distributed_d26_results/README.md), implementation8e04668d7,
BOTH tools/distributed-d26. Shared D25 harness gains fixture parameterization;
original installed tools/policy helper preserved. Owned eager GPT2/tied embedding-
head/21280unique trainable params/root+2GPT2Blocks/world2/FULL_SHARD/origparams,
FP32/BF16working/FP16working-nativeShardedGradScaler pass. Per-rank parameters+
AdamW10640 each; aliases stay SAME Parameter before/after gradients/updates/load.
D16 causalcollator/shifted CE/global49/57/57/21, actuallocalzero-targetmicro;
CPUFP64+roundedCPUAMP ALLgrad/loss, nativeglobalclip/CPUdoubleALLAdamW at unchanged
D15/D21/D22/D25 bounds pass. Max BF16relative.006585226/.006604265, FP16.000825151/
.000917906. NativeFP16finite scaler512; rank1Infwindow3 both skipAdamW/scheduler/
EXACT unchangedstate/backoff256→128, nextfiniteupdate3. CP2 six-fileDCPunion653339B
finite/653347Boverflow, originalhashes retained; fresh process/container/UUID/
seed1773/scaler8 restores ALLstate/scaler256/fourRNG EXACT, windows3/4 ALLgrads/
loss/clip/AdamW/RNG/counts/digests EXACT for allpolicies+overflow. BERTFP32default
control passes.232GPUgroups/18ACKguards/15CPUpatterns256observations/62localtests/
Black/Ruff/diff pass. No failedGPUphase; archive verifier alias-call-count check
corrected to exact18/15/9/7, originalfailure retained; r2 metadata verified with
same GPU/handoff bytes. No source-provider/native-prefix/image/kernel/default/
registry/app changes/reboot/foreignstop; PLAN-02 cohort preserved. Logs clean/
finalidle/no jobs/guards, docs/memory synced, bootsafcc7be4/8c396bd1 unchanged.
HistoricalPythonCPU fault separatefixedVMheap/D05/D21; no speed/largefit/universal
causal-model/stochastic/elastic promise. NextPLAN-04/D27 explicit frozen/adapter
FSDP admission then activation checkpointing; retain frozen refusal until proof.

## PLAN-02 native rollout completed (2026-11-08)

[Results/rollback](bc250_plan02_results/README.md), implementation72c4276ad,
sourcee1d8d1ff920. Selected daily/RCCL on BOTH hosts use versioned r4 MIOpen
381eac05240e33acced9472fb7c3071afe8bc4f434407e67a29ce64423b935ac.
FP16 HIP-direct forward FP32 accumulation plus versioned cache option prevents
reusing old half-accumulation binaries. Fresh-cache tests alone proved insufficient:
r2 rolled back, old-cache reproduction retained. Both profiles/hosts preserve12
old logical records/add6 keys, cold/warm CPU references pass, no warm saves,
original owned cache hashes preserved. Full .104 consumer matrices98groups each,
.105 native matrices26groups each, live rollback/original-mismatch/re-promotion,
eight identity checks/two activation refusals/public two-host RCCL+DDP references
pass;49 final r4 ACKguards/62localtests. Metadata/archives in linked report.
Coherent new CK/SDK/ORT pins selected on .104; explicit CK stays FP32, B07.13
separate. No optional SDK deployed to .105. Its daily launcher is preserved;
D09 explicit RCCL follows new alias. Stable runtime lock inode preserved on each
host; new target profiles enable locking. Old releases/packages/caches and r2
fallback/production healthcheck/native controls preserved. Unchanged host-local
ROCm files are hardlinks: detach modified inodes in future updates. No native
Torch/image/kernel change, app access, reboot or broad performance claim.
Historical Python CPU fault remains distinct from fixed VM heap/D05/D21.
Next PLAN-03/D26 actual Trainer FSDP GPT2 tied weights/shifted loss/restart policies.

## PLAN-01 reservation update (2026-11-08)

Selected upcoming MIOpen/CK modules migrated to stack-only board_preflight and
cooperative board reservation/shared runtime lock. Default Docker admission and
post-test state now checked. Implementation2e24474d6; additive tools/plan01 both
hosts. Four CPU lock/admission controls per host and62local tests pass; no GPU or
new recovery reboot qualification. [Evidence](bc250_plan01_results/README.md).
Native providers/boot/registry unchanged. Next PLAN-02 identity/precision repair
qualification; use new overlay ahead of old helpers. Other old guardians and
custom daemons are not implicitly migrated.

## Remaining-work priority review (2026-11-08)

[Current prioritized checklist](../../../BC250_REMAINING_PLAN.md) reconciles the
full flow chronology through D25. Ten important work packages, plus optional
expansion; not a completion percentage. Next PLAN-01 reserves boards and checks
selected guardians, then PLAN-02 qualifies ordinary-runtime MIOpen FP16 precision
repair. D26 remains next model capability. Frozen/PEFT FSDP, activation
checkpointing, real capacity/performance, bounded crash triage, complete source
reconstruction and a coherent release remain important. B06.12 and K6 are done;
B07.13 deployment is still pending. Planning only: no new runtime/hardware proof,
no host changes, old fault distinctions and safe defaults preserved. Host copies
remain D25 until the next authorized sync.

## Evidence, precedence and audit coverage

Current host manifests, exact source snapshots, committed patches and executed
reports outrank prose from older milestones. The audit covered 143 BC-250 branch
commits after base `7a803357a6e0de10c780a683aa7492da39579938`, all 17 library,
three system, two Torch and four Triton patches, component histories, retained
validation/benchmark reports, `.104` manifests/kernel/sysfs and a read-only
`.105` legacy-source inventory. Twenty project threads were present in the local
Codex state index. Private transcripts are recovery sources, not support proof.
The unrelated untracked `src/` was preserved and excluded from modification.

Distinguish **implemented**, **built**, **executed/numerically qualified**,
**unqualified**, and **hardware unavailable**. Counts describe particular scopes;
counts from different milestones overlap and must not become a coverage percentage.
No claim below establishes every public operation, shape, type or concurrency mode.

The durable entry point is the root AGENTS.md rule plus flow.md; keep the detailed
inventory on demand instead of injecting its entire history into every turn.
OpenAI documents project instructions and workspace files as continuity tools:
[AGENTS.md discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[workspace memory and compaction](https://developers.openai.com/cookbook/examples/agents_sdk/building_reliable_agents_memory_compaction).
Files persist independently of conversational summaries; no hidden permanent
model memory or guaranteed perfect recall is claimed. A versioned documentation
copy under `WS/agent-memory/latest` is separate from qualified source snapshots.

This audit changed documentation only. A later documentation commit does not
invalidate the installed code revision or require a ROCm/Torch rebuild.

## Hardware: measured board versus background descriptions

Community reverse-engineering documentation describes BC-250 as an ASRock mining
APU board with Zen 2 CPU cores, shared GDDR6, DisplayPort, Ethernet and USB. Its
stock core counts and marketing architecture names describe a different
configuration from our unlocked board. No manufacturer datasheet was recovered;
use measurements and native compiler capabilities for implementation decisions.
See the [community hardware documentation](https://github.com/H6-Technologies/BC-250).

The compiler execution model is **gfx10.1.3**, not a substitute gfx1010/gfx1030
identity. LLVM lists gfx1013 as an APU and exposes wave64 as a target option;
our native qualification uses wave32/XNACK−. Target options alone do not prove
wave64 or fault-migration execution. See [LLVM's processor table](https://llvm.org/docs/AMDGPUUsage.html#processors).

| Property                    | Measured `.104` value / durable evidence                                                                                                                        |
| --------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| CPU                         | AMD BC-250, family 23/model 71, eight cores/16 threads, SMT active; lscpu and `/sys/devices/system/cpu/online`                                                  |
| CPU instruction/cache facts | AVX/AVX2/FMA; 256 KiB L1d, 256 KiB L1i, 4 MiB L2, 8 MiB L3; one NUMA node. Do not assume AVX512.                                                                |
| PCI identity                | `0000:01:00.0`, vendor/device `1002:13fe`, subsystem `1022:0000`; DRM device uevent/sysfs                                                                       |
| VBIOS string                | `113-AMDRBN-003`; not a complete BIOS image provenance claim                                                                                                    |
| Native GPU                  | KFD target 100103; HIP `gfx1013:xnack-`                                                                                                                         |
| Topology                    | 80 SIMDs, two SIMDs/CU → 40 CU, arranged as 20 WGP; HIP multiprocessor count 20 is not a lost-CU bug                                                            |
| Execution                   | Qualified wave32, 64 KiB LDS; KFD max 20 waves/SIMD; historical HIP L2 report 4 MiB                                                                             |
| Compute queues              | KFD 24 CP queues; DRM compute ring field is a mask, not a queue count                                                                                           |
| SDMA enumeration            | Two engines, eight queues/engine; zero XGMI SDMA engines and zero P2P links. Enumeration is not user-SDMA qualification.                                        |
| Memory                      | MemTotal 15,579,344 KiB ≈14.858 GiB; GTT 15,938,355,200 bytes =15,200 MiB ≈14.844 GiB; VRAM aperture 512 MiB; KFD local_mem_size 0                              |
| Memory implication          | CPU/GPU share physical board memory. GTT and system RAM are overlapping views, not separate ~16 GB budgets. OS/cache/CPU allocations consume the same capacity. |
| PCI link sysfs              | Current/max 16 GT/s, width 16. This is an advertised link property, not measured shared-memory bandwidth.                                                       |
| Firmware fields             | KFD fw_version 144, sdma_fw_version 44, num_xcc 1. Firmware-file presence does not independently prove which image loaded.                                      |

Hardware-relevant sources include `compiler/amd-llvm/llvm/lib/Target/AMDGPU/GCNProcessors.td`,
`AMDGPU.td`, ROCr `core/runtime/isa.cpp`, the native HIP probes and
`WS/logs/baseline-torch.log`. `WS` means
`/home/nik/bc250work/rocm-10.2-gfx1013` throughout this file.

| Capability                          | Current conclusion                                                                                                                                                               |
| ----------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| Scalar FP32/FP64, FP16/BF16 storage | Native numerical GPU paths qualified for documented operations; availability is not a throughput guarantee                                                                       |
| Low-precision atomics               | Existing HIP CAS fallback passed contention, returned-value and neighbor-guard tests; no newly implemented/native FP16/BF16 atomic instruction claim                             |
| Packed arithmetic versus dot        | Some scalar/packed-half arithmetic is usable; unsupported packed dot lowering is replaced by widened scalar operations. Do not equate all packed arithmetic with matrix support. |
| MFMA/WMMA                           | No usable native path. Never whitelist another architecture or enable matrix-only engines to bypass this. Scalar implementations of the mathematics remain possible.             |
| Wave64 / XNACK+                     | Compiler/runtime registration exists; our board execution/fault migration is unqualified                                                                                         |
| SDMA                                | Keep `HSA_ENABLE_SDMA=0`; ROCr also intentionally defaults gfx1013 user SDMA off. Explicit override is an experiment, not a supported default.                                   |
| VMM / async pools                   | APIs retained; K5 bounded r3 fixed-VA remap passed with the qualified override. Historical r2/original-r3 failures remain evidence; experimental VM heap stays disabled.                                                                                                     |
| GPU reset                           | Current boot and live parameter `amdgpu.gpu_recovery=0`; reset behavior unqualified                                                                                              |
| Media                               | DRM VCN decode/encode, JPEG, UVD and VCE queries return −EINVAL with zero exposed rings. Current driver exposure is not proof silicon lacks every media block.                   |
| Profiling                           | API/kernel/copy tracing works. Hardware counter tables, PC sampling, ATT and SPM are unqualified.                                                                                |
| Debugger                            | Native architecture/ELF registration patched; consumers and attach/step/wave suites not qualified                                                                                |

## L02 real LoRA checkpoint replay update (2026-11-07)

[Report](bc250_lora_resume_results/README.md),
[commands](../../../BC250_LORA_TWO_HOST.md#restore-a-sealed-l02-checkpoint).
Implementation284d3315b/guard-verifier4d5f60e43; additive L02 runner/profile on both
hosts, original L01/outputs/checkpoints retained. Native loaders plus sealed
common/rank-RNG file inventories and model/data/settings fingerprints; custom
restoration of TRL's otherwise-reset token counter. Offline peer preparation
preserves local RNG, validates before atomic rename, never unpickles payloads.
Actual Qwen3.5-2B/rank128/134553600 FP32 elements, native FP16 AdamW/global16/
dropout.05: checkpoint2→fresh windows3/4 after reboots matches ALL parameter/
gradient/AdamW/scheduler/scaler/RNG/data/per-step hashes exactly. Scale8→loaded512,
poisoned RNG restored, token10648→15745/19377. Test-only accelerated growth produces
native overflow skip4, exact preservation/backoff1024→512; production growth kept.
Five accepted/three retained rejected guards: two overstrict skip-probe failures,
one .105 R2 system rejection for Docker startup despite exact math. New Docker
pre/post check; final R3 pair clean kernel/registry/Docker/Podman. 54 local tests
and3 actual refusal/preservation controls pass. Payload11/1635023695B; all native
checkpoints preserved, retained .105 transport tar; only redundant new .104 tar
removed. No native install/prefix changes or agent reboot. Same two ranks/pinned
sources/FP16/AdamW/workers0/unpacked optimizer boundaries; no one-host conversion,
elastic/shared-FS/partial-window/crash-durability/full-epoch/performance proof.
Native aggregate train_loss caveat retained. Next D24 with Docker-aware guardian
before BF16 FSDP working-parameter/all-gather qualification; D23 guards unchanged.

## L01 authorized two-host LoRA application update (2026-11-05)

[Report](bc250_lora_two_host_results/README.md),
[usage](../../../BC250_LORA_TWO_HOST.md). User explicitly authorized
`bc250-projects/lora-smoke`, separate from excluded `finetune-n7`. Additive DDP
entry point on both boards, newer app-only Transformers5.22.1/TRL1.17.1/
PEFT0.21.2/Accelerate1.15/datasets5.1, native providers unchanged. Qwen3.5-2B,
rank128/alpha256/all-linear, 134553600 FP32 LoRA elements, FP16 base/AMP/native
scale128, global16 preserved through accumulation8 single/4 per rank. Fixed
rank-0-only final evaluation and explicit process-group shutdown. First-update
POST-CLIP full-gradient one-GPU comparison relative L2 .0034373344 <=.03 and all
parameter bounds pass; first loss/native PRE-CLIP norm also match. No independent
CPU Qwen proof. DDP final replicas identical; no skipped updates. Dropout.05 R2
matches all96 input+label hashes: warm five updates122.892s single/99.861s DDP,
1.231x throughput; whole short train_runtime239.404/248.044s, startup makes DDP
slower overall. No full-run speed/capacity/convergence claim. Production smoke
one update +two-rank eval8, ALL372 saved tensors exactly match live PEFT export.
Eight acknowledged guards, kernel/registry/boots unchanged, no native install or
prefix mutation. Recovered allocation retries and reference conv/delta paths
remain; no general8192-length fit/fast-package proof. Complete metadata exports
verify; initial non-root exports rejected for missing root-only guard directories,
retained separately. Originals/checkpoints untouched. Next application L02 is
full same-world checkpoint distribution/fresh-process optimizer/scaler/RNG/cursor
replay; old one-host continuation unqualified. Stack D24 remains separate.

## D25 actual Trainer classic-FSDP FP16 update (2026-11-08)

[Report](bc250_distributed_d25_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-fp16-working-parameters-and-scaling-d25).
Implementation/verifier36a215d8c, BOTHtools/distributed-d25. Explicit
fsdp_fp16_parameter_policy: nativeFP16workingparams+gathers/FP32masters+syncedgrads/
reduce+buffers+AdamW, positiveglobalclip/enabledworldnativeShardedGradScaler.
PublicunscaleONCE, finitepublicrootFSDPclip, nativerankoverflow synchronization/
skip/backoff/scheduler suppression; actualAdamWprehookrefuseslateinvalid/scaler
mismatch. OldF32/BF16/frozen/packageguards retained. PinnedAccelerateoptimizer/
modeling sources added. OwnedBertMLM22561/eager/world2/fullshard/root+2layers/
workers0/sequential/micro2/accum2/global8/8/8/4: roundedCPUFP16+FP64loss/ALLgrad,
clip/CPUdoubleALLAdamW atD15/D21/D22bounds pass,maxrelative.001115321/.000713187.
Master/AdamW11284/11277. Stress128/growth2finiteends512; rank1Infwindow3 AFTER
reduceflags[False,True] BOTHnative skipAdamW+scheduler/unchangedstate/scale256→128;
window4finiteupdate3. FP32/BF16/inactive controls +postclipNaN/scalemismatch
refusals zeroupdates pass. CP2/DCPunion6files864357Bfinite/864365Boverflow,
originalhashes retained; freshconstructor1773/scaler8 restoresALLstate/scaler256/
4RNG EXACT, finite+overflowwindows3/4 ALLloss/grad/clip/optimizer/scaler/RNG/data/
digests EXACT. Nativeproductiondefaults65536/2000 unchanged. No universal/speed/
capacity/stochastic/elastic proof. Final128groups/18ACKguards/8contracts144obs/
58localtests/Black/Ruff/syntax pass;4failedguards retained (R1probeexpectedscale,
R2bypassedoptimizerandschedulerhook). Fixedactualcreate_scheduler+rawoptimizer,
R3allrerun assertsgateidentity/calls;10intermediateR2passedguards/all32ACK retained.
Noprefix/package/kernel/app changes/reboot/foreignstop; allfinalkernel/registry
unchanged, Docker/Podmanidle/no jobs/armedguards; entrybootsafcc7be4/8c396bd1
retained, docs/memorysynced. HistoricalPythonCPU/KFD separatefixedVMheap/D05exit/
D21load. NextD26 plannedactualTrainerFSDP eagerGPT2causalLM/tiedparams/shiftedglobal
loss/F32+BF16+FP16/CPUALLgrad+clip+AdamW+exactrestart. Frozen/PEFT/FSDP2/offload/
compile/activationcheckpointing/elastic/realcapacity separate.

## D24 BF16 FSDP working parameters update (2026-11-08)

[Report](bc250_distributed_d24_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-bf16-working-parameters-d24).
Implementation/verifier dfe347376, BOTHtools/distributed-d24. New explicit
fsdp_bf16_parameter_policy/native BF16 workingparams+parameterallgathers,
FP32 mastershards/reductions/buffers/synchronizedgrads/AdamW, no inputcasts/
keptlowgrads/scaler. D23 compute-only/frozen/package guards retained. Docker-aware
pre/post guardian now required; defaultdaemon queried explicitly, no foreignstop.
Owned eager BertMLM22561/world2/fullshard/root+2layers/micro2/accum2/workers0/
sequential/global8/8/8/4: roundedCPU BF16localaccum+FP64global loss/ALLgrads,
nativeglobalclip/ALLclippedgrads/CPUdouble ALLAdamW pass unchangedD15/D21/D22bounds.
Maxrelative CPUAMP.008212774/FP64.005715182. Master/AdamW elements11284/11277
(BF16alignment), FP32control11282/11279. Inactiveclip/FP32/one-rankNaNcollective
refusal preserveparams+optimizer+scheduler/zeroupdates. CP2+offlineunion6/864357B:
initialALLshards/AdamW/scheduler/fourRNG EXACT, windows3/4 ALLgrad/loss/clip/state/
RNG/data/digests EXACT. Nativecheckpointhashes retained. BF16gather277592B/35calls
seed118968B/15replay; inactive79312BvsFP32control158608B/10calls,2Bvs4Belements,
paddingdiffers. FP32reduction361088B/12calls. No totalNIC/peakRAM/speed/largefit proof.
64GPUgroups/10ACKguards/6CPUcontracts60obs/58localtests/Black/Ruff/syntax pass;
no GPUfail/kernel-log/registrychange. Initial archive verifier fulltail overcount
corrected; original archives retained. No nativeprefix/package/kernel/app change,
no jobs/containers/armedguards; boots afcc7be4/8c396bd1 retained, bothdocs/memory
synced. Historical CPU Python/KFD faults distinct fromfixedVMheap/D05exit/D21load.
NextD25 planned actualTrainerFSDP FP16/nativeShardedGradScaler/CPUgrad/globalclip/
overflow-skip/exactrestart; keepFP16refusal untilproof. Largerfit/causal/frozen/
otherarchitectures/elastic separate.

## D23 actual Trainer FSDP BF16 computation update (2026-11-05)

[Report](bc250_distributed_d23_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-bf16-compute-policy-d23).
Helper/factoryc49d99b84/export-verifier0ecfd6258; BOTHtools/distributed-d23.
Explicit fsdp_bf16_compute_policy +bf16True admits nativeBF16autocast withALLFSDP
params/reductions/buffers/master+gradshards FP32/noscaler/noinputcasts. Default/
lowparameter/lowreduction policies refused, FP32 pathretained. GPU LinearoutputBF16/
weightsF32/logits+lossF32/reduceinput+outputF32 observed;3reductions/finalmicro,
0nonfinal;361024inputB/4updates,not totalNIC/paramcomm/capacity/speed proof.
OwnedBertMLM22561/D21dataset/padding/accum2/eager world2: matchedCPUBF16+FP64
loss/ALLgrad atD15 bounds, maxrelativeCPUAMP.0006256018/FP64.0048790555; everyparam
normfloor3e-6sqrt(n). Nativeglobalclip/ALLclippedgrad/CPUdoubleALLAdamW bounds
D22 unchanged. Nativecheckpoint2 +offlinepayloadunion EXACT initialshards/AdamW/
scheduler/fourRNG and2resumedwindows ALLgrad/norm/coefs/weights/moments/loss/RNG/
rows/digests. BF16 inactivecoef1, strictFP32compatibility andone-rankNaNcollective
refusal/zeroAdamW preserveparams+optimizer+scheduler. Accepted64groups/10ACKguards,
4CPU patterns40obs/42regressions, no GPUfailure. No native/default/package/app
changes/install/reboot; bootc1199fca/bad537ed retained, bothdocs/memory synced,
no jobs/containers/armedguards. No lowprecisionforwardparam/paramallgather/FP16/
scaler/FSDP2/generalmodel/speed/largefit proof. Historical faults distinct from
fixedVMheap/D05exit/D21load. NextD24 BF16forward/backwardparams+allgatherbuffers,
FP32masters/reductions/round-aware CPU/clipping/AdamW/restart readiness; keepD23
compute-only untilproof. FP16scaler/actualcapacity/elastic separately.

## D22 actual Trainer FSDP FP32 clipping update (2026-11-05)

[Report](bc250_distributed_d22_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#trainer-fsdp-fp32-gradient-clipping-d22).
Helper/fixture2a11969bc/export-verifier790357231, BOTHtools/distributed-d22-r2.
Positivefinite max_grad_norm admitted in separate BC250FSDPTrainer, native root
FSDPglobalL2clip retained; zero=D21noclip. Collective pregrad/postnorm finite
checks stop both live ranks beforeAdamW. OwnedBertFP32 rawloss/ALLgrad CPUFP64,
CPUfullglobalnorm/coefficient/ALLclippedgrad/ALLAdamW params+moments pass D21/D14
bounds; cap.05 active4updates/norms1.46937–2.26792, cap100coef1. Freshfixedworld
nativecheckpoint2+payloadunion exactinitialshards/AdamW/scheduler/4RNG then2windows
ALLgrad/norm/coefs/weights/moments/loss/RNG/rows/digests EXACT. Separate one-rank
Inf/NaN andfinite1e30normoverflow both-refuse/noAdamW/unchangedparams+optimizer+
scheduler pass; transientgrad/RNG notrollback, sessiondiscard/reloadrequired.
Accepted68groups/12ACKguards +2CPUlimitpatterns24obs/42regressions. Retained2failed
Infsetupguards(permission folder, correctedbeforecontrols); no numericalgate
relaxed. Runtime/native/defaults/apps unchanged, no install/reboot; c1199fca/bad537ed
retained. Bothdocs/memorysynced, no jobs/containers/armedguards. No speed/capacity/
AMP/scaler/FSDP2/generalmodel proof; latecallbackmutations/elastic unsupported.
Historical CPU/KFD/visibility separate fromfixedVMheap/D05exit/D21loadrepair.
NextD23 actualTrainerFSDP BF16 readiness/policy/CPUgrad+globalclip+AdamW/exactrestart;
keepAMPguard untilproof, FP16scaler/largefit separate.

## D21 actual Trainer FSDP FP32 and sharded restart update (2026-11-05)

[Report](bc250_distributed_d21_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#trainer-classic-fsdp-fp32-d21).
Helper/fixture/payloadutility5189348ba/export-verifier39918b800; opt-in BOTHtools/
distributed-d21-r3/BC250FSDPTrainer, original DDPhelper unchanged. Actual classic
FSDP FULL_SHARD/original FP32 params/root+2BertLayers; ownedBertMLM22561/28records/
unequalMLMmask/accum2/global8+final4. Four GPU forward/backward/ALLgrad/loss CPUFP64
andALL-parameter/AdamW moments/counters CPUdouble gates pass at D12/D07 bounds.
Native empty-optimizer DCP template lost ALLAdamWmoments onfreshresume:4failedACK
guards retained. Subclass public optimizer-aware DCP loader+FSDPconversion repair
restores EXACT modelshards/ALLAdamW/scheduler/fourRNG then2windowsfulltranscript.
Native shardedcheckpointrequires BOTHhosts' DCPfiles; offline bounded256MiB
hash/owner/conflict utility merges sixfiles864357B, localTrainer/RNG/scheduler
unchanged, originalhashespreserved. Samecompletedcheckpoint/world/rank/trusted
inputs only, notsharedFS/lineagevalidator/elastic. Param+AdamWelements11282/11279
sum22561; no peakRAM/speed/largefit proof. Accepted30groups/4ACKguards +4failed
separate,4localCPUcontracts/42regressions pass. Native/defaults/apps unchanged,
no install/reboot; .104returnedc1199fca r2/.105bad537ed r3. Bothdocs/memorysynced,
no jobs/containers/armedguards. No AMP/scaler/clipping/FSDP2/causal/generalresume
qualification; D12 aggregate train_loss caveat/native warnings remain. Historical
PythonCPU/KFD/visibility separate fromfixedVMheap/D05exit/thisloaddefect.
NextD22 actualTrainerFSDP FP32 globalnormclipping CPUallgrad/clip/AdamW+exactrestart,
extendcurrent max_grad_norm=0 guardonlyafterproof.

## D20 BF16 diagnosis and explicit evaluation precision update (2026-11-05)

[Report](bc250_distributed_d20_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#explicit-fp32-evaluation-head-d20).
Helpers/contracts/bytehash1a61687a7/export997b38bb7/finalscope570e60428.
Native wide Bert BF16 two FP64 outliers reproduced EXACTLY bymatched CPU BF16;
GPU-vsCPU-BF16 zero gate violations over30408704 values, maxabs.001953125.
Evidence supports BF16 rounding inthisfixture, not provennative arithmetic bug;
original.003/.003 FP64 gate remainsfailed fornativeBF16, no tolerance change.
New opt-in BOTH tools/distributed-d20-r3/bc250_eval_precision.py fp32_eval_head
keeps selected FP32-master eval head inFP32, BF16encoder retained. Actual BertMLM
head originalFP64gate0 violations/maxabs.00005873836, GPU-vsCPU policy.00000470877.
Per-instance scope restores onnormal/exception exit; eval+no_grad/singleinput/
singleoutput/FP32weights guards. Modelstate/IDs/keys/ties unchanged; training/
backward/generalmodel/speed/capacity unqualified. Actual Trainer.evaluate observes
encoderqueryBF16/headFP32 and native metrics EXACT afterexit. Full/compact metrics
parity1e-11/native loss unchanged WITHINpolicy. 36groups/6ACKguards, CPUdigest3/
12obs/head4/8obs,42localtests. New bounded versioned bytearraydigest replacesowned
D19 list/JSONfingerprint; native/runtime/defaults/apps unchanged. Guardsall exit0,
no jobs/containers/armedguards, bothdocs/memory synced; bootsd83af4e1/bad537ed.
Historical PythonCPU/KFD/visibility separate fromfixedVMheap/D05. NextD21 practical
distributed gap review +actual pinned TrainerFSDP FP32 readiness/CPUgradient/
optimizer/fixed-world restart integration ifsupported; do notrepeatqualifiedDDP.

## D19 compact validation and vocabulary crossover update (2026-11-05)

[Report](bc250_distributed_d19_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#compact-metric-validation-and-larger-vocabulary-d19).
Final helper/fixtures1a39b5254/export-verifierc99e46fe4, BOTH tools/distributed-d19-r3.
Active-token selection retained, scalar validation reads12→1 in GPU dispatch
probe, nonzero3 remains. Dense prototype rejected: wider78ms/extra~40MB versus
D18~59–61ms. Final176groups/16ACKguards, CPU5patterns/80obs,GPU2patterns/32obs,
42localtests pass: actual Trainer Bert/GPT2 smallFP32/BF16/FP16 +vocab32768/maxlen64
FP32,19records/unequal masks/tail dedup, full+D18+D19+NumPy64 metrics1e-11 andALL
real-position logits at ORIGINAL CPUFP64 bounds. Native loss unchanged, labels/
counts/order/opt-out exact, no state/gradient/optimizer mutation. Wide compact
57.7–61.3ms vsfull1.12–1.15s =18.6–19.6x THIS FP32 evaluation, D18 already similar;
final small pairedD18/D19 ratios1.014–1.034/twosamples, no universal speedclaim.
Wide arrays152MiB→456B/gatherinputs79167488→240B perrank; labels/localbatchlogits/
padding remain; tensorbytes notwire, ~75.5MB live allocator saving notlargefit/RSS.
Initial wideBertBF16 metricparity pass but CPUFP64 logitgate FAIL2/30408704,
maxabs.003654425659, fixed.003/.003 unchanged; retained2failedACKguards and16prior
successful prototypeguards separate fromfinal16. GPT2wideBF16 unrun, wideBF16
unqualified, not provennativefault. Native/D09/consumers/defaults/kernels unchanged,
no reboot/install/app access. Finalcontrollers8981/8659 exit0, nojobs/containers/
armedguards; bootsd83af4e1/bad537ed retained, bothdocs/memory synced. Historical
PythonCPU/KFD/visibility separate fromfixedVMheap/D05. NextD20 matched CPUautocast+
FP64 investigation ofwideBF16outliers, originalgates; replace expensive owned
fixture listfingerprint with bytes before newwide tests. Broad generalizations/
streaming/largecapacity/recovery remain separate.

## D18 compact distributed token statistics update (2026-11-05)

[Report](bc250_distributed_d18_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md#compact-distributed-evaluation-statistics-d18).
Helper/contracts/fixtures9a31ac663/export-verifier6bc35b079. Opt-in
CompactTokenMetrics supplies pre-gather FP64 GPU per-example NLL/count/correct
and stateless final token metrics after native tail dedup. Staged BOTH tools/
distributed-d18; no native/default/consumer changes. Actual Trainer Bert/GPT2
FP32/BF16/FP16,19rows/dynamicpadding/unequal counts/tail, full/compact ABBA plus
evaluate/full-logit label-free opt-out:108groups/12ACKguards pass. Full+manual
NumPy64 parity1e-11, CPU model D17 bounds, native loss unchanged. Four CPU
contractpatterns/48observations,42localtests. Prediction arrays88464→456B194x,
prediction gather inputs36472→240B~152x perrank, all loss/label/prediction inputs
37264→1032B~36.1x. Tensor payload only, not wire/totalcollectives. Live allocator
peak decreases12288–40448B in tiny fixtures, no largerfit/RSS claim. Compact
cold211–231ms/repeat43.5–49.3ms vs full mostly38.2–44.6ms(one90.5ms):NO speedup.
Full local logits/native padding/labels remain, not streaming/generation/general
models/recovery. GPUchunk7, CPU1/3/4096; conservative FP64/scalar-sync overhead
open. No jobs/guards/containers; qualification bootsdc9e9293/133b8dc2, later
syncbootsd83af4e1/bad537ed, no development reboot/requalification. Docs/memory
BOTH synced. NextD19 optimize launches/synchronizations and bounded larger-vocab
crossover with retained parity and explicit defaults.

## D17 distributed evaluation and token metrics update (2026-11-05)

[Report](bc250_distributed_d17_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md#distributed-evaluation-and-token-metrics-d17).
Metric/contracts/fixturesf8f0a475a/exportscf7eb27c2. New opt-in TokenWeightedMetrics
stateless dataset token_nll/perplexity/accuracy/count/example metrics from full
native gathered logits. Native eval_loss retained, different batch weighting:
BertF32 4.59509 vs token4.56861; GPT2F32 4.60964 vs4.58609. Actual native Trainer
evaluate/labeled+label-free predict, BertMLM+GPT2 inFP32/BF16/FP16 onbothnodes,
19records/dynamicpadding5–12/unequaltail. EXACT19roworder/labels, native tail dedup,
counts126masked/120causal. ALLreal-position logits vsCPU-FP64; explicit metrics
vsmanual NumPy64 at1e-11; tokenNLL vsCPUmodel precisionbounds. Label-free exact
same logits/no labels/loss/metrics; params unchanged/gradsNone/nooptimizer.
96groups/12ACKguards/42localtests+5CPUpatterns/60obs, first revisionpasses.
Native/D09 runtime/helpersD12–16/kernels/packages/apps unchanged, no build/install/
update/reboot/download. Both docs/memory synced, no jobs/guards; historicalfault
separation retained. Full logits STILL gathered/stored: FP64chunking not whole-
logit memory solution. No streaming/generation/largefit/general eval+trainresume
or speed/memory measurement claim. NextD18 compact per-example token stats BEFORE
gather with CPU parity and bounded payload/memory/timing comparison.

## D16 causal Transformer AMP and target accounting update (2026-11-05)

[Report](bc250_distributed_d16_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md#causal-transformer-training-and-target-counts-d16).
Helper/contracts/fixturesce09f6c24/exportsad6454602. New opt-in CausalLabelCollator
clones caller labels/masks position0+rightpadding BEFORE Trainer counts;
causal_token_loss shifted FP32 CE/global count, explicitD12 normalization/D14clip.
Localallignored-positiveglobal differentiable0 tested; emptyglobal/missingcount/
unmaskedfirst labels refused. Nativecount49/57/57/21 vsraw51/60/59/23. Actual
GPT2LMHeadModel21280/eagercausal/dropout0/cacheFalse/Conv1D, BF16/FP16 Trainer,
rightpadding/promptmasks/global8+final4/CPUautocast+FP64 ALLgrad +CPUclipping/AdamW
pass unchanged D15 limits. Maxgradrel vsFP64 BF16.00495313/FP16.000661749.
FixtureFP16scale128/growth2→256 checkpoint2→coordinated skip/backoff128→recovery,
no native defaults changed/default65536 claim. Nativefreshresume complete state/
losses/grads/RNG/rows/global+local counts exact,8rank-windows.96groups/8ACKguards/
42localtests+7CPUcontractpatterns/56observations, first revision passes.
Native/D09 runtime/helpersD12-14/kernels/images/packages/apps unchanged; no build/
install/prefixupdate/reboot/modeldownload. Both docs/memory synced, no jobs/guards.
Right-padded unpacked sequences only; no leftpad/packing/RoPE/GQA/SDPA/cache/
generation/FSDP/DeepSpeed/checkpointedgrads/arbitraryarch/loss/scale/eval/convergence/
capacity/performance/generalresume proof. Historical fault distinctions retained.
NextD17: actual Trainer distributed evaluation/prediction +explicit token-weighted
metrics/unequal final batches/CPU references. Release reconstruction separate.

## D15 actual Transformer AMP Trainer update (2026-11-05)

[Report](bc250_distributed_d15_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md#offline-transformer-amp-training-d15).
Fixturesdeed14f14/exports84039269e; D12/D13/D14 helpers/native runtime unchanged.
Actual BertForMaskedLM22561/eager/dropout0, variable attention padding/MLM counts,
BF16/FP16 Trainer token-normalized UNshifted CE/active clip.05/AdamW/StepLR.
CPU-autocast micro/rank and CPU-FP64 full-global graph references with manual
logsumexp CE: ALL param gradients +global budgets pass declared bounds, maxwhole-
grad relativeL2 vsFP64 BF16.00487906/FP16.000594423; vsCPUAMP .0000582494/.0000732985.
Per-param norm floors for nearzero cancellation explicit; not independent Bert
implementation. CPU-FP64 norm/ALL clipped grads/AdamW params+moments pass.28records/
sequential workers0/global8 plus final4. Fixture explicitFP16scale128/growth2,
native growth128→256, coordinated rank-local overflow skip/backoff128/finite
recovery; no default65536-Bert proof or native default change. Native per-host
checkpoint2→fresh replay ALL loss/gradient/state/RNG/rows exact,8rank-windows.
70groups/8ACKguards/42localtests, first revision passes; docs/memory synced BOTH.
No build/install/prefix mutation/reboot/app access/model download/performance
claim. Native tied-key notices pass exact restore, warnings/historical faults
remain as recorded. No causal LM/SDPA/Flash/FSDP/DeepSpeed/gradient checkpointing/
arbitrary model/scale/convergence/capacity/general resume qualification.
NextD16: owned causal Transformer AMP Trainer SHIFTED CE +valid-target counts,
clipping/CPU refs/native restart. Evaluation/release reconstruction separate.

## D14 Trainer AMP clipping and FP16 coordination update (2026-11-05)

[Report](bc250_distributed_d14_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md#coordinated-trainer-amp-clipping-d14).
Implementation625b840d7/exportsb3e3d8a30. Opt-in BC250ClippedTrainer decorates own
public Accelerator clipping method; native Trainer/unscale/L2 clip/scaler retained.
D12/D13 helpers unchanged. Pre-unscale scaler agreement/badflag makes rank1-only
AFTER-DDP FP16 Inf detectable by BOTH native scalers: skip/backoff/recovery
coherent. Final public gate stops BOTH on NEW post-clip callback nonfinite;
BF16 pre-clip bad input and scaler mismatch collectively refuse, checkpoint2
params/AdamW/StepLR/globalstep2 retained. Sixrank refusal observations, reload
required (RNG/grad/scaler transient state advances). Same D07 MLP/variable-mask MSE,
BF16/FP16 actual Trainer, clip.002 active (coeff.0365090–.107265): independent CPU
pregrad oracle +FP64 global L2/ALL clipped gradients +CPU AdamW params/moments
pass fixed limits. Eight exact fresh native resumed rank-windows incl FP16 skip,
nonfinite masks/finite components exact.118groups/14ACKguards/42localtests;
first revision passes, native/D09 runtime/registry/images/kernels/defaults/apps
unchanged. Both docs/memory synced, no jobs/guards. Dense FP32 master two-host
one-optimizer L2 only; no dynamic callback reorder/mutation after final gate,
peer-loss/arbitrary model/AMPBert/FP32/sparse/FSDP/performance claim. Finite extreme
norm overflow guard has no separate GPU injection proof. NextD15: actual owned
Transformer AMP Trainer token CE/clipping/reference/native restart integration.

## D13 actual Trainer AMP update (2026-11-05)

[Report](bc250_distributed_d13_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md#trainer-bf16fp16-and-finite-gradient-checks-d13).
Fixture/callback70bd075e7/exports62b35356c. D12 explicit BC250Trainer unchanged;
actual Trainer BF16/FP16 DDP MLP1051648, masked MSE/global active-window count,
D07 independent explicit-rounding CPU loss/ALL-gradient oracle and CPU-double
AdamW ALL-param/moment checks pass. MaxgradientrelativeL2 BF16.00276769<=.02,
FP16.000372784<=.004; fixed bounds, no correction.28records/global8 plus final4,
sequential/workers0,clip0. Native FP16 default65536 scaler skips third attempted
update with Inf BEFORE final DDP reduction, preserves params/AdamW/scheduler,
backs off32768 and recovers. Global_step4 vs actual Adam/StepLR3. Native per-host
checkpoint2+scaler/four RNG →fresh native Trainer resume exact through attempts3/4,
8rank-windows. Optional public FiniteGradientCheck collectively stops BF16 on
rank1-only AFTER-reduction nonfinite before ANY optimizer change, both retain
checkpoint2 weights/AdamW/scheduler/step2. Stop/reload, not automatic skip; callback
refuses FP16.68groups/10ACKguards/42localtests, all first-revision phases pass.
Native/D09 tools/registry/images/kernels/consumer packages unchanged, no build/
install/prefix update/reboot/app access; both docs/memory synced. No AMP Bert/
arbitrary loss/clipping/FSDP/DeepSpeed/eval/capacity/performance/elastic/general
resume proof. Historical Python fault remains separate from fixed VMheap/D05 exit.
NextD14: Trainer AMP clipping +coherent rank-local FP16 nonfinite policy.

## D12 actual Trainer and accumulation repair update (2026-11-05)

[Report](bc250_distributed_d12_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md#transformers-trainer-on-two-hosts-d12).
Fixture514ed8be2→acceptedf5edcaac1, exports3f2d1e391/CPUtestcc47efc09.
Stock Trainer5.2/Accelerate1.12 double-divides accumulation: two-micro gradients
0.5x CPU reference. Explicit BC250Trainer removes only the Accelerator division,
retains native Trainer loop/checkpoints, version+source guards, no user-script
or package mutation. Actual two-host FP32 Bert MLM22561 CPU loss/all-gradient/
AdamW params/moments pass4updates;28records/global8 plus finalshortglobal4.
Equal-token native mean uses accepts_loss_kwargsFalse; variable masks explicit
unshifted masked_token_loss hook/global count, no generic loss auto-inference.
Native per-host save_on_each_node checkpoint2→fresh train(resume...) restores
complete state, four RNG and rows exactly,8resumed rank-windows.78checkgroups
(6stock diagnosis+72corrected)/10GPUguards/18actualCPUchecks/42localtests.
Native/D09 tools/registry/kernels unchanged. r1 LR floating assertion corrected
without loss/gradient tolerance changes; logs/source/checkpoints retained.
CPU invalid HIP_VISIBLE_DEVICES=-1 import abort diagnostic retained; omit flags
for CPU-only assertions, invalid-environment parser behavior remains separate
from historical Python fault and fixed VMheap/D05 agent exit. GPU bootsold7344/
e1cb; current dc9e/133b after intervening reboot, no development reboot issued.
No actual Trainer AMP/FSDP/DeepSpeed/clipping/eval/convergence/capacity/speed or
arbitrary-data/resume claim. NextD13: actual Trainer BF16/FP16 bounded references.

## D11 native Accelerate integration update (2026-11-05)

[Report](bc250_distributed_d11_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md#accelerate-on-two-hosts-d11).
Fixture21766d81d/exportsf5329413b, existing Accelerate1.12.0. Actual two-host
DDP prepared model/loader/AdamW/StepLR +accumulate/backward in FP32/BF16/FP16:
CPU loss/all-gradient/optimizer references pass fixed D07 bounds. Global8/four
micros, zero reduction before last; update-only scheduler/native FP16 overflow
skip and explicit FP32/BF16 finite guard/recovery pass. Native save_state on both
hosts/save_on_each_node +registered cursor; fresh native load_state restores all
state including four RNG streams exactly. Iterator prefetch before restore avoids
CPU RNG advancement. Two replay windows/case/rank exact (12rank-windows), four
one-rank bad metadata cases refuse both without mutation (24observations).
78groups/12ACK guards/42localtests; native/D09 tools/registry/kernels unchanged,
no build/install/reboot/app access. Independent CPU low-precision rounding oracle,
not a new arithmetic implementation. No generic checkpoint API/untrusted-native-
load/multiworker/shuffle/partial-window/elastic/Accelerator FSDP/HF Trainer/
capacity/speed proof. NextD12 planned: actual offline Transformers Trainer
integration with bounded references. Historical Python fault remains distinct
from fixed VM heap/D05 exit repair.

## D10 fresh-process replay update (2026-11-05)

[Report](bc250_distributed_d10_results/README.md),
[recipe](../../../BC250_DISTRIBUTED_GUIDE.md). Fixture553eebce1→accepted0c0efa810,
exports97641e4b6. Separate launches of owned tiny FP32 Bert DDP and FP16 FSDP MLP
reproduce three windows exactly against uninterrupted GPU references: loss/all
local grads/params/AdamW/scaler/StepLR/Python+NumPy+TorchCPU/CUDA+cursor RNG and
shuffled cursor across epoch boundary and skipped optimizer update. Five updates/
six attempts,12 exact resumed rank-windows; strict deterministic test setting.
Collective local hash/config/cohort admission +weights_only load;11 one-rank bad
header/file/serialization/loaded-metadata cases refuse both before state applies.
48groups/44refusal observations/42localtests/8accepted guards. r1 false namespace-
inode uniqueness proof failed before restore, corrected using CID+UUID/start ticks;
failed source/logs/checkpoints and4ACKs retained, not a native arithmetic defect.
Native stack/D09 tools/registry/daily/kernels unchanged, no build/install/reboot/
app access. Not a generic API, DataLoader-worker/partial-window/elastic/peer-crash/
arbitrary model/precision migration or speed/capacity proof. NextD11 planned:
actual Accelerate two-host DDP accumulation/AMP integration, beyond D09 imports.

## D09 public distributed consumers update (2026-11-05)

[Report](bc250_distributed_d09_results/README.md),
[options/recovery](../../../BC250_DISTRIBUTED_GUIDE.md). Tools e1bbb8ec7 with
maintenance37067fdde/exportsbb4d672ea installed BOTH as distributed-d09. Ordered
env/env-file/mount/mount-ro/shm flags reuse ordinary parsing, private caller context
and fixed two-node/eth0/Ring/Simple restrictions revalidated. Agent/worker pin Torch
package/_C path/version and selected native/Python hashes before user script;
D05 exit repair/native runtime SHA/release unchanged, daily/default/images/kernels
unchanged. EX candidate qualification→activation→D05 live rollback→reactivation
and installed consumer pass; durable hashed payload/old metadata retained.
Owned offline Bert22,561 params with frozen Transformers5.2.0/peft.18.1/accelerate1.12,
FP32 global8/two micros/rank: two CPU-double losses/all-gradients/AdamW updates,
external ro/rw mounts/env order/spaced args/cwd/UID pass before/after installation.
D08 FP16 FSDP retained references/scaler/shard-nonfinite/exact replay pass new
bootstrap. Actual Nik/doas CPU Gloo3/context/three identity refusals per rank,
Torch CUDA flag false.46GPU+6CPU groups/24CLI+6identity refusals/37localtests/6ACKs;
clean agent exits/kernel logs/registry per phase. No speed/arbitrary package/model/
HF Trainer/mixed Bert/full capacity or fresh-process checkpoint claim. No native
build/package install/reboot/user-app access; existing loopback profile retained.
NextD10 newly planned: owned fresh-process two-host checkpoint/restart with
scheduler/RNG/data cursor and strict mismatch references, fixed-world boundaries.

## D08 classic FSDP AMP/shard update (2026-11-05)

[Full report](bc250_distributed_d08_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md). Numerical0815ac97b/helpers97e75946a
qualify owned1.05M residual MLP/global8/micro1/accum4/rank, classic FSDP FULL_SHARD/
use_orig_params with actual BF16/FP16 forward params, FP32 reduction/master/final
grad shards. Eight Linear wrappers/input casts off/outer autocast, not an arbitrary
model or alternative-policy claim. Each rank525,824 params/AdamW exp_avg elements;
two CPU rounding/FP64 oracle gradient/loss steps and separate actual-gradient CPU
AdamW owned-state comparisons pass. Max grad norm errors BF16.31388%/FP16.05349%,
within fixed2%/.4%. Rank1-only Inf/NaN AFTER RS leaves rank0 finite; native FP16
ShardedGradScaler and BF16 collective finite guard both skip exactly. Mismatched
scaler refuses; same-worker local optimizer/scaler checkpoint next update exact.
100groups/48checkpoint+8scaler refusals/8ACKs, clean agents/logs/registry. no_sync
retainsfull low-dtype grads vs FP32 shards, reduction-input4.21 vs16.83MB/rank per
update (all-gathers/control excluded). Isolated live allocator peak75.04 vs73.61MiB,
+1.43408; not whole-board/capacity/general scaling or speed proof. Native cohorts,
registry/profile/images/kernels unchanged; no install/build/reboot/app access.
Docs explicit opt-in. NextD09 newly planned: distributed env-file/env/mount and
owned tiny Transformers consumer integration, caller/identity/refusal/rollback.
Fresh-process checkpoint, larger-model capacity/other transports/release stay open.

## D07 distributed AMP/scaler update (2026-11-05)

[Full report](bc250_distributed_d07_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md). Tools ae52b7306/615caa65e qualify
BF16/FP16 autocast on an owned1.05M residual MLP, global8/micro1/accum4/rank,
no_sync and every-micro sync. Native cohorts/profiles/registries/images/kernels
and active D05 exit bridge unchanged; no backend defect or build/install/reboot.
Two CPU low-rounding/FP64-math oracle loss/all-param+input-gradient steps; largest
param relative-L2 errors0.22560% BF16/0.04558% FP16, below fixed2%/0.4% limits.
Separate CPU-double AdamW uses actual admitted AMP gradients, not an unrounded
whole-trajectory equivalence claim. Rank1 Inf/NaN before final reduction propagates,
both skip with exact params/AdamW unchanged; BF16 explicit finite guard, FP16
native scaler growth/backoff/recovery. Collective scaler mismatch refusals and
same-worker exact optimizer/scaler checkpoint replay cross a growth boundary.
92worker groups/48preload+8scaler-mismatch refusals/8ACKs; clean agents/logs/registry.
FP32 masters/gradients and4.21MB no_sync vs16.83MB every-micro reduction-input
bytes/rank: no automatic gradient compression. No speed/convergence/attention/
compiler/arbitrary model/FSDP AMP claim. Docs only add explicit recipes; apps
untouched. NextD08 newly planned: FSDP mixed precision/accumulation/shard-aware
nonfinite decisions/state replay and actual shard/allocator measurements.

## D06 distributed accumulation update (2026-11-05)

[Full report](bc250_distributed_d06_results/README.md),
[usage](../../../BC250_DISTRIBUTED_GUIDE.md). Numerical toolsbd3726047 and
sequencing/export539951ced reuse the active D05 launcher/agent repair. No native
runtime/default/profile/registry/image/kernel change. Equal global batches8/32,
micro1/2, FP32 transformer12.61M, two CPU-double loss/gradient/parameter/AdamW
state comparisons and exact same-world optimizer-boundary checkpoint replay pass.
192 worker groups/144 preload refusals/24 ACKs, clean agents/kernel logs/registries.
Counted steady reductions12→3 or24→3 buckets,201.75/403.51→50.44MB bucket input
bytes per rank; not physical wire counters. no_sync1738→462ms or3475→496–497ms,
3.76x/~7x versus synchronizing every micro; single63–65/123–128ms remains faster.
Peak live allocator rises51.10MiB/rank; no memory/capacity/general speed claim.
SH locks,4CPU/10GiB/12GiB/600s guards; no jobs remain. Owned checkpoints stay on
hosts, raw metadata/logs verified locally. Existing app loops unchanged; opt-in
no_sync pattern already supported by installed Torch. NextD07 newly planned:
bounded BF16/FP16 AMP DDP accumulation, finite/nonfinite optimizer/scaler and
checkpoint references. FSDP capacity/other transports/release audit remain open.

## D05 distributed lifecycle and measurement update (2026-11-04)

[Full report](bc250_distributed_d05_results/README.md),
[use/recovery](../../../BC250_DISTRIBUTED_GUIDE.md). Both explicit commands now use
versioned distributed-d05 tools; source-derived API-hash-checked agent exit method
from Torch8d8b18d repairs TCPStore owner lifetime. Native Torch a524980/ROCm/daily
profiles/images/kernels remain unchanged. Direct torchrun outside the explicit
command still uses the old installed API. CPU real-store reproduction and two
candidate repetitions/two units, strict GPU candidate/public gates, ordinary-user
CPU context and live rollback/reactivation pass. Historical old DDP-a .105
post-worker agent error is preserved; it is unrelated to VM heap/Python CPU fault.
Matched-global-batch FP32 transformer: single23.55–27.22ms, DDP438.25–438.41ms,
FSDP651.55–651.70ms; no speedup. Live allocated memory lower with state sharding,
reserved pool higher; larger-model fit not established. NextD06 accumulation/no_sync
communication amortization with independent optimizer/checkpoint references.

## Kernel, unlock, cooling and recovery

Current host105 userspace (D01–D04,2026-11-04): after 656 target checks/nine
guards and nine ordinary launcher checks, `.105` selects
`/opt/bc250-stack/releases/10.2-gfx1013-20261104-d01`, daily/RCCL Torch source
`a524980` and image1bb4ac0e matched to `.104`. All24,664 ROCm inventory entries
already matched; the new release explicitly shares retained immutable M105.1
ROCm. Do not retire that base, its old image506f or operational health tools/data.
No kernel/reboot/boot-policy change: `.105` signed r3/module0957f1, retained r2;
`.104` remains r2. Original M105.1 migration/boot recovery evidence remains in the
[migration report](bc250_host105_migration_results/README.md). Current update,
reviewed daily rollback and dependency rules: [HOST105_UPDATE.md](../../../HOST105_UPDATE.md).
Actual two-host RCCL/DDP and bounded FSDP state/checkpoint references are now
qualified below; performance/large-model capacity/elastic recovery remain open.
[Rollout evidence](bc250_distributed_d01_results/README.md),
[developer usage](../../../BC250_DISTRIBUTED_GUIDE.md).


Historical observation2026-10-19 (N2.18): .104 already running fallback
`7.1.5-bc250rocm-r2`, boot `d0532075-88fb-4a2f-974e-f61578d20d9f`.
No reboot or boot-policy change in N2.18; reason for the intervening reboot is
not established by this batch. The test launcher accepts an explicit r2/r3
acknowledgement-guard release, retains r3 default, refuses mismatch before GPU
launch and keeps the600s deadline/nonce/boot-ID/recovery checks.

Previously qualified experimental kernel: `7.2.10-bc250rocm-r3`, six guarded
one-shot boots2026-10-15. Qualified module0957f1aa4: bounded GPU/Torch/VMM and prior SVM
milestones pass; HPD init dummy errors and two IRQ ASSERT WARNs repaired below.
Physical display/hotplug and broader failure/recovery remain unqualified.
The retained r2 fallback is
`7.1.5-bc250rocm-r2 #2 SMP PREEMPT_DYNAMIC` (built2026-09-16). Its recovered
original configuration is `bc250stack/kernel/configs/r2.config`; r3 migrated
config/diff and evidence are in `bc250stack/kernel/r3/evidence/`. Do not infer
configuration from unrelated7.0.10 headers.

Retained r2 fallback boot files:
`/home/nik/bc250work/bc250-rocm-lab/{vmlinuz,initramfs}-7.1.5-bc250rocm-r2`.
New r3 artifacts are `/home/nik/bc250work/kernel-r3-7.2.10/artifacts/`; entry
`bc250-rocm-r3-7.2.10-trial` defers AMDGPU until base SSH/cooling ACK.
GRUB persistent default is `bc250-rocm-r2-kernel`, timeout three seconds.
`WS/recovery/` retains kernel/initramfs, grub.cfg, grubenv, grub and cmdline.txt.
Additional lts6.18.52/n17/stable7.0.10 boot files are present, but their presence
alone is not a verified fallback boot. Use one-shot GRUB selection for a new
kernel, preserving the known working persistent default until SSH gates pass.

Relevant current boot arguments (see `/proc/cmdline` for exact root UUID):

```text
amdgpu.sg_display=0 mitigations=off amdgpu.gttsize=15200
ttm.pages_limit=3891200 ttm.page_pool_size=3891200
```

The exact TTM arguments are `ttm.pages_limit=3891200` and
`ttm.page_pool_size=3891200`; additional arguments are
`amdgpu.gpu_recovery=0`, `amdgpu.bc250_cc_write_mode=3`,
`amdgpu.bc250_flush_pasid_kiq=0`, `amdgpu.bc250_flush_by_runlist=3`, and
`bc250.vulkan_v33_r1=1`. Preserve this known profile when investigating changes.

SP5100 TCO watchdog: OpenRC feeder10s/120s, `/dev/watchdog0`, config
`/etc/conf.d/watchdog`. r3 driver builtin/nowayout1, early initramfs arms180s
until root feeder handoff; candidate-only post-sysctl guard restores panic60
and oops/lockup panic. Actual r2 feeder-countdown/reset/SSH return observed
(reset-cause flag0, not a cause confirmation). r3 idle-GPU15s deadline expiry
emergency-restarted to r2; SSH/default/new boot ID verified, then r3 rechecked.
These bounded gates do not qualify pre-watchdog hangs or arbitrary GPU hangs,
AMD reset, SDMA or fault experiments. Persistent r2 default retained.

CPU unlock service `/etc/init.d/bc250-8core-unlock` calls
`/usr/local/bin/bc250-8core-unlock boot`. The script describes SMU mailbox 0x98,
mask register 0x0115A870, stock mask 0x77→0xff and warm-reset persistence;
these descriptions are not newly executed register experiments. Read-only
recheck2026-10-15: service enabled in OpenRC default/started; eight cores/16threads
online. Installed boot path never reboots (comments retain an earlier bootloop).
Cold-start mask reapplication needs a subsequent warm reboot for enumeration;
that sequence was not newly tested. Evidence/script/service snapshots:
`kernel/r3/evidence/core-unlock-audit/`. Requested upstream
`rw-r-r-0644/bc250-core-unlock` is volatile too; its permanent option is a BIOS DXE
mod. No firmware flash or duplicate unlock applied. BIOS-level persistence is
excluded from current scope by the user2026-10-15: leave BIOS untouched and
retain the existing service/warm-reboot arrangement;
r2/SSH fallback cannot recover a pre-OS firmware failure. DMI P3.00 alone is not
proof of unmodified firmware or mod compatibility. The 40-CU
patch is retained at
`/home/nik/bc250work/bc250-gfx1013-fix/patches/kernel/40cu-bc250-unlock.patch`,
with older work under `/home/nik/bc250-40cu-unlock`. Imported performance claims
in that patch are not measurements from this audit.

Important operational facts:

- No cpufreq policy or scaling_governor exists on `.104`; do not report a
  configured CPU performance governor.
- Kernel GPU power policy is auto, but an external SMU governor is active on
  .104 (observedPID5265). Its fix-metrics/fix-freq options bind tmpfs replacements
  over `gpu_metrics` and hwmon freq1_input. Those views are governor-supplied;
  DPM/DRM use the suspect original Current clock slot. True attained clocks
  remain unqualified; auto does not mean stock clocks. Source/provenance below.
- The SMU OC service requests CPU 3500 MHz, scale −19, max_temperature 90;
  these are settings, not verified operating frequency/voltage.
- `/usr/local/bin/bc250-fan-curve` reads CPU k10temp/Tctl despite comments saying
  GPU temperature. NCT6686 pwm2/CPU_FAN1 controls the fan. The existing curve
  uses PWM90 below60°C,175 below70°C,250 below80°C,255 above. Do not assume
  hwmon numbering survives reboot. AMDGPU hwmon has no PWM attributes here.
- Retain cyan_skillfish2 CE/ME/MEC/MEC2/PFP/RLC/SDMA firmware files under
  `/lib/firmware/amdgpu`; file presence is not verified runtime load identity.
- Retain `/home/nik/bc250work/{bc250-rocm-lab,bc250-gfx1013-fix,bc250-vulkan-v33,strata-port}`,
  `/home/nik/{bc250-40cu-unlock,bc250_smu_oc,rocm-systems}`. Older Vulkan7.0.10
  module/initramfs artifacts are distinct from retained r2 and the new r3 candidate.

K6 packaging/handoff complete2026-10-15: `bc250stack/kernel/r3/handoff/README.md`
consolidates current scope, exact configs, source/signature provenance, build/sign/
install and one-shot boot/recovery recipe. Image84995f1 and module0957f1 remain
separate identities. A disposable Git index reconstructs both exact trees from
signed base+six patches;21 source/fan/binary/config/firmware artifacts checksum-
verified locally and on .104. Current source archive and binary snapshots remain
in ignored `kernel/artifacts/k6-7.2.10/`; .104 KWS/handoff/k6 retains copies.
No generated module signing private keys exported; no runtime/build/reboot/GPU requalification.
Fresh read-only artifact and loaded identity checks pass; watchdog/SSH/cooling
active, same boot a287fd16, r2 default and empty next_entry. Historical failures
below are superseded only by their recorded later fixes. Telemetry fidelity,
Python CPU crash and MIOpen timing/workspace defect remain unresolved; bounded
kernel failure/recovery limits remain. Next N2.1 solver coverage audit.

Clean-kernel workstream K1–K6, bounded outcome2026-10-15: fresh signed stable
Linux7.2.10 base`7b20d05fd38de8e0be168956f6fd241d750c7d69`, Greg fingerprint
`647F28654894E3BD457199BE38DBBDC86092693E`; separate source
`bc250stack/kernel/linux-r3` revision`84995f1a85188cde56910729214b0a68a7e402da`.
Native .104 workspace`/home/nik/bc250work/kernel-r3-7.2.10`,16jobs/20GiB ccache;
image/modules/fan driver built/signed/installed,499-entry initramfs inspected.
Initial fan sign-file path failed, then repaired using O=build tool; logs/unsigned
module retained, no kernel rebuild. Image SHA
`4531223f35bd9f313a242c2aa5d7bece0b4ffd4161d28b41271914a9b54d4558`;
initramfs SHA`0e8444c328d070e0f6d422514116e19ccb13feb9e44d0bb4b289f0d1ea3eeb11`.
Loaded kernel/AMDGPU build IDs match build artifacts; signed NCT6687 works with
existing cooling service. r2/default/artifacts/.105/ROCm/Torch preserved.

Minimal series retains CU/PASID/TTM quirks, restores diagnostic probe registers
and propagates runlist errors through existing partial-operation paths. Some
upper SVM notifier/unmap callers still discard returns/pre-mark ranges unmapped;
end-to-end failed-flush containment and failure injection remain unqualified.
Full93,517-file r2 source comparison found7 active AMD changes+5 backups versus
archived7.1+7.1.5 (archive signatures not verified). Retained/installed r2 module
payload and loaded build ID match; not bit-reproducible proof. Actual r2 SDMA
firmware bytes are Navi12 under cyan names, differing from root cyan files;
r3 privately pins actual r2 bytes. Root firmware unchanged, ROCr SDMA off.

Same N1.8 runtime/image/DSO+kpack/harness compared on r2 and r3: Lt664 references,
SparseLt134 numerical+16 API, hipTensor304 API+12 GPU and19 eager Torch CPU
references passed; workload kernel-log deltas0. Two existing Display Core IRQ
WARNs persist at AMDGPU load. Malloc/free24 rounds+pinned GPU alias/streams/events
passed, including another24-round smoke after actual recovery/second r3 boot.
Historical pre-fix full gates failed explicit fixed-VA VMM on both kernels:
handle0 saw handle1 data before update; independent GPU copy confirmed it.
K5 isolation: HIP and direct ROCr each8 mismatches/26 checks, distinct-VA anchors
correct, ordinary KFD map/unmap churn restores26/26. This identifies stale compute
translations on DRM GEM_VA, which bypassed the existing KFD runlist workaround.

Patch0003/module source`8f3c0d352643378629e5a73f4b9df96b7d38d94a` waits for PT
completion, drops BO reservations before DQM locking, flushes through the existing
opt-in BC-250 runlist path and publishes timeline completion after success.
Only AMDGPU rebuilt/installed; kernel IMAGE remains84995f1, same signed image and
initramfs. New signed module is r3-only`updates/amdgpu.ko.gz`, build/installed/
loaded ID`12ad61569200a95f1d626396fca3692753231707`, compressed SHA
`451b8a3d05cdf494b67546f9074c86f4d931481a3efd7f7b64e7bad48611dadb`.
Original r3 module and r2 artifacts/default retained; .105 unchanged.

Original full gate now passes unchanged: library counts above, Torch19, two fresh
memory processes each24 allocation/pinned/streams/events rounds+16 explicit VMM
remaps. HIP/direct ROCr each26 checks pass without diagnostic churn; workload
kernel logs identical. New boot`e2a06828-208c-47f7-9838-8a5be9f7bfd5` base/cooling
and GPU guard acknowledged; no running jobs. Evidence`kernel/r3/evidence/k5-*.json`,
pre/post isolation logs, patch and source manifest; host`SOURCE_K5.json` separates
image/module provenance. Single .ko target failed modpost companion symbols;
incremental modules target succeeded with16jobs/20GiB ccache, preserved logs.

No universal VMM, concurrent raw DRM timeline ordering, injected-failure,
performance or sustained claim. Fixed CSR/VM-heap corruption and historical
unknown Python CPU segfault remain separate; experimental heap/SDMA remain off.
Next: upper SVM notifier/unmap failed-flush lifetime containment, then Display
Core warning analysis, telemetry and K6 integration; N2.1 library cursor retained.

SVM CPU-unmap follow-up/source`67d0f189570c831efa17b524c080de07bbec485a`
(patch0004) now bounded-qualifies one returned-error path: single-device BC-250
CPU unmap holds an independent queue eviction before PT invalidation and retains
it on error after successful quiescence; successful invalidation balances it.
Full-range mapped_to_gpu commits only after all invalidations succeed. Root-only
one-shot injection targets an exact host TGID and only fires while queues are
stopped; default/verified final value0. Initial scheduler-stop failure, retry-fault
migration and map-side BO/DQM ordering are NOT thereby qualified.

Current signed r3-only module SHA
`c1d0e5a887bc3d2019472b9758cf57c920e9ef9544b3b47f6ed1a2d620f5e868`,
loaded/build/installed ID`c99a2c6dc9074760efe06d074ec0248ce5e8739b`, image84995f1
unchanged. K5 module retained independently at`KWS/recovery/svm-before`, stock
r3/r2/default/.105 preserved. New boot`eb2cc9df-5df6-4250-a0ec-1af4de205f93` base/
cooling/GPU guards ACKed, no active jobs. SOURCE_SVM.json separates image/module.

Partial and full CPU-unmap injection each pass: five prior GPU references plus
normal write control; -EIO consumed once, exactly one expected kernel error line;
queued write unexecuted through200 one-millisecond host polling samples; process
exit0 and six-reference fresh-process SVM gates before/after. Baseline six SVM
checks had passed on K5 module. Unchanged full library/Torch/VMM regression gate
passes counts above, no workload kernel-log delta. No performance, hardware-hang,
scheduler-stop, retry-fault or universal SVM safety claim. Evidence
`kernel/r3/evidence/svm-cpu-unmap/`, helpers/source manifest/README. Next audit:
SVM map runlist flush under BO reservations; move with notifier/lifetime ordering
preserved before concurrency qualification. Display warnings/telemetry follow.

SVM validation locking follow-up/source`41db89c103b183c82c05bbe099ea96ad1ddee8d1`
(patch0005): validation map/unmap now capture PT fences under BO reservations,
release reservations, then wait and invalidate under notifier lock. Caller-held
range/mmap/SVM lifetime locks retained; prior updates drain despite later VMA
errors. Tracking-allocation failure falls back to an in-place fence wait; it does
not invoke DQM with caller BO reservations held. Validation PROT_NONE bounds
corrected from bytes/pages mismatch to current-VMA inclusive page indices.
Same probe:0 target-page unmaps before,2 after; no GPU access while PROT_NONE,
restoration references passed. Two concurrent processes each54 GPU references
(24 multi-VMA attribute rounds+24 CPU discard/restores) pass, no workload dmesg
change. Both prior partial/full error-containment gates and unchanged full
Lt664/SparseLt150/hipTensor316/Torch19+two memory processes each24 allocation/
16 VMM rounds pass again. Intentional2 trace+2 error lines retained separately.

Current r3-only module SHA
`d47974d6d7eef898b50a4fdf03edb716e8cd13dfd7f2373b495f273727e5f07f`,
loaded/build/installed ID`7ee5573fc124ceb0edad1bc662feb44ebdbf129f`; image84995f1
unchanged, previous module/source at`KWS/recovery/svm-map-before`. Boot
`84debfc4-5f83-413b-9656-6a5fd3140a3a` base/cooling/GPU guard ACKed; no active
jobs. Source/GIT_REVISION and SOURCE_SVM_MAP.json record41db89c10 module delta.
Evidence`kernel/r3/evidence/svm-map-deferred/`, helpers/README/patch0005.
No lockdep/universal deadlock, OOM/fence-error/partial-map failure-lifetime,
failed-stop/resume, XNACK/retry migration or multi-GPU qualification; no performance/
sustained claim. Next: analyze existing Display Core IRQ WARNs without blanket
suppression, then telemetry/K6. r2/default/.105/runtime/SDMA0/heap-off preserved;
unknown Python CPU segfault remains separate and unresolved.

HPD/Display Core follow-up/source`0957f1aa4717bb5ea19545bdcb01d6899e94bb9f`
(patch0006): bulk init iterated six HPD+six HPDRX enums despite DCN201 only
implementing two pairs. New active-table capability query rejects dummy/empty
entries before bulk clearing, preserving real set/ack operations, connector
enabling and invalid-call diagnostics. Baseline8 dummy-ack+8 dummy-set+8 failed
clear errors and2 IRQ ASSERT WARNs →0 each after guarded boot/load. Temporary
DRM KMS debug4 traced clears1,2,7,8 and restored0. DCN2.0.1 still active with
dc=-1; both DP connectors still enumerate disconnected; taint4160, WARN bit clear.
No physical hotplug/scanout/audio/resume or other-ASIC qualification. Existing
2 SDMA0 fence fallback-expiry and workqueue truncation messages remain; SDMA off.

Current r3-only module SHA
`6ebd8393a9c7d25cd810fd89d4ef5defa187980cae64da0d37ee0d975715e8d1`,
loaded/build/installed ID`b97bca56793aecea35967385d0c68d4186ecd121`; image84995f1
unchanged, previous module/source at`KWS/recovery/hpd-before`. Boot
`a287fd16-c97b-4d88-8055-b0553a2f7631` base/cooling/GPU guard ACKed, no jobs.
SOURCE_HPD.json/sourceGIT_REVISION0957f1aa4;16jobs/20GiB ccache dependency rebuild.
Unchanged Lt664/SparseLt150/hipTensor316/Torch19 plus two processes each24 ordinary
allocation+16 VMM rounds pass; workload dmesg delta0. Evidence
`kernel/r3/evidence/hpd-capability/`, source/patch/helpers/README. r2/default/.105,
userspace, SDMA0/heap-off and separate unknown Python CPU segfault preserved.
Next: read-only eight-core SMU/live-clock/k10temp audit against retained packaging
patch0005 before policy changes, then K6; N2.1 next library item. Prior SVM failure/
lockdep/XNACK/multi-GPU limitations remain.

Read-only telemetry audit2026-10-15 completed on0957f1aa4/same boot/prefix.
Original sysfs metrics read through governor-held fd6 bypassed overlays without
unmounting/stopping it. Device0:22 vs overlay0:25 verified; this is translated
v2.2 data, not raw PMFW.84 sequential samples~5Hz around an8-second gfx1013
integer workload;131,072 exact CPU references passed, kernel log identical,
config/executable/fan-script hashes and mounts unchanged. Existing governor
PID5265 preserved; no policy/source/prefix/boot changes. Guard ACKed, no own jobs.

Governor clock reports1000MHz idle→2000MHz median during own load; original
kernel/DPM/DRM mostly2–7MHz, max47; average kernel456–953MHz remains untrusted.
Overlay views agree84/84, shared kernel views74/84 sequential samples; neither
proves physical clock. Eight CPU core groups/16online threads confirmed, while
kernel copies6 core slots: temperatures0/2 report0.63–1.88/0.62–3.56°C,1/3 zero,
6/7 clock/temperature65535. GPU activity65535/DRM load errno22; overlay utilization
is governor-derived. Legacy hybrid-layout claims fit these symptoms but full
PMFW offsets/core identities were not qualified. No CPU-offline/raw-mailbox test.

Reported edge51–68°C versus original GFX51.25–68.25°C; hwmon truncation gives
−0.75..0°C difference in samples. CPU Tctl52.625–67.125°C; fan uses that CPU
sensor, existing PWM90→175/RPM~2k→3.5k. Absolute calibration/maximum thermal or
stability claim excluded. Installed governorSHA00cd65a6... and candidate built
SHAe8028ca2... differ; reference repo240b0df/Cargo0.4.0 lacks observed freq-overlay
code, so running executable/source identity unresolved. Local read-only archive
`kernel/artifacts/telemetry-readonly/governor-reference-240b0df.tar` (SHA in report)
preserves reference source, original repo/history untouched. Evidence/report:
`kernel/r3/evidence/telemetry-readonly/`, `kernel/r3/telemetry/README.md`.

Next pin active governor source/overlay behavior and kernel/daemon mailbox
coordination before a minimal guarded live-clock getter. Retained packaging0005
cannot be applied blindly: core count does not prove PMFW map; instantaneous
power is labeled average, and live-query failure falls back to the suspect slot.
Keep per-core/power semantics separate. K6/N2.1 retained; SDMA0/heap-off, prior
SVM failure/lockdep/physical-display limits and Python CPU segfault remain open.

Read-only kernel review2026-10-14:
[report](bc250_kernel_audit_20261014.md), [hash/file evidence](bc250_kernel_audit_20261014.json),
[active AMD delta](bc250_kernel_audit_20261014.patch). Reconstructed AMD subtree
from .105's archived7.1+7.1.5 patch:7 active changed files (+136/-3),2762 identical
and5 inactive `.orig` backups. Archive signatures/rest of Linux remain unaudited.
Installed AMDGPU modules match across hosts; loaded/initramfs identity not proved.
That initial identity gap was subsequently closed by the byte/build-ID comparison
above; retain this audit as a historical snapshot, not the current cursor.
Source concerns: ignored runlist-flush errors at all four custom call sites;
probe modes1/4 fail to restore RLC (current boot uses3); retained eight-core
telemetry patch0005 is absent. Existing boot log has two Display Core IRQ WARNs,
so unchanged workload logs are not evidence of a warning-free boot. No kernel
cause for Python segfault established; no live changes/build/reboot this review.
Before a new kernel: verify upstream base/full delta, review minimal patches and
prepare separately versioned candidate with one-shot boot/SSH fallback preserved.

## Installed stack and precise provenance

| Item                   | Current code / location                                                                                                                                      |
| ---------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------ |
| Branch                 | `users/Naster17/rocm-10.2-gfx1013`                                                                                                                           |
| Qualified TheRock code | `0115c01f06f8654b26b5a665286069caf89eda2d` (N1.8); newer docs commits need no runtime rebuild                                                                       |
| Libraries              | `08ba6f386ff661e720873779fc417b4246787d9c`                                                                                                                   |
| Systems                | `329eaf726632ab0136924804cbc1321ae09fc126`                                                                                                                   |
| LLVM                   | Upstream `7667f52dd687899256a3285ab0bce3c53bc73394`; no custom LLVM architecture patch                                                                       |
| Torch                  | `4ee65717d69e9f51021272e3f37d0f342f68fc08`; release/2.13 baseline `c779020750d7dc99a39ea8c104ef8df1fc9d0755`                                                 |
| Triton                 | `ee65ff3b85f99d52e678b252a99c8425fa46e329`; baseline `66b3f9ff575eb8db6adec658bbf0badd9a8d2e25`                                                              |
| Distribution/version   | TheRock 10.2.0; HIP component reports 7.17.0; Torch 2.13.0a0+gfx1013, qualified Triton 3.8 wheel. Different component version strings are not a failed port. |
| Active link            | `/opt/bc250-stack/current` → `/opt/bc250-stack/dev/priority1`; directory name retained, runtime profile `gfx1013-priority2-n18-dev`                              |
| ROCm / Torch           | `WS/build/dist/rocm`, editable `WS/pytorch-work`                                                                                                             |
| Runtime image ID       | `5df0a58392de13916819f6d84342b0338ae2c2bbb2f3c21a1f3ad38be2724a59`                                                                                           |
| Runtime tag            | `localhost/bc250-stack:10.2-gfx1013-priority1-candidate`; tag name predates P2, immutable ID above is authoritative                                          |
| Build image            | `localhost/bc250-rocm-build:10.2-ccache`; Ubuntu24.04/glibc inside Alpine host                                                                               |
| Triton wheel SHA256    | `2d65d6fefe359116521640e3cc2257bd03b4b24f6877fc8e28946ee9c57544d8`                                                                                           |
| Candidate              | `priority2-20261014` payload retired20261016; `runtime.retired.json` and validation retained as historical evidence, no longer executable                     |
| Recovery link          | `/opt/bc250-stack/before-dev-current` → retained `10.2-gfx1013-20261013-4b000d542` release                                                                   |
| Software fallback      | `previous` and `before-dev-current` → retained `10.2-gfx1013-20261013-4b000d542`; initial20261012 payload retired20261016                                       |

Remaining-folder review20261104 (.104 only, toold16294297): three obsolete
folders retired after full local archive/member/type verification and Git fsck:
bc250-gfx1013-fix clean5bb0dac duplicate, old .104 migration transfer staging,
and rocm-performance-20261013 old benchmark fixture. 113,943,255 logical bytes,
95,923,498B local archive SHA c50a20b1ff95a8528cab4d1b39d243eed2b30545b6ed857ab76a8a6c7e3833aa.
System/runtime/process dependencies absent; EX lock, native/GRUB hashes unchanged,
CPU launcher check passes. Seven actual folders plus required D04 alias retained;
private RADV has an absolute driver dependency, kernel/r2 folders serve development/
recovery, and active workspace/docs/tools/evidence remain. .105 not contacted.
Full retention map: [WORKSPACE_LAYOUT.md](../../../docs/WORKSPACE_LAYOUT.md).

Top-level follow-up20261104 supersedes the earlier keep-every-alias policy:
user explicitly authorized obsolete agent-file retirement. .104118/.105154 loose
entries retired (225 regular files,47 unused aliases),2,837,354,361 logical bytes.
Workstation archives verified both whole-file and every member SHA before deletion;
redundant host backup copies also retired. Unused installer/import/export staging
folders consolidated under tools/evidence. Top levels now12/17 entries, no regular
files. Keep D04 report alias on both and .105 qualification/recovery/promoter paths;
system service/tool/registry references protected. Native manifests/bootIDs unchanged,
CPU launcher/path checks pass; no GPU change or qualification. Logs/plans/recovery
archives: kernel/artifacts/workspace-cleanup-20261104/top-level. docs/README.md
remains the host entry; obsolete historical commands need grouped paths/new labels.

Storage organization20261104: .10496 exact workstation-backed transfer copies
(7,017,699,431B) and .105one duplicate import (1,490,746,029B) retired under host
EX locks after full hash/size preflight. 94 old .104 archives/two patches retained
locally in kernel/artifacts/workspace-cleanup-20261104, with literal deletion plans
and host journals; these ignored historical archives are not publication-approved.
No build/source/cache/checkpoint/image/release/native/boot/user-app retirement.
Distributed20/32 helper/evidence entries grouped on104/105 with all old paths
preserved as relative links. Runtime/registry/base hashes unchanged; CPU CLI/path
checks pass, no native changes or GPU requalification needed. Available space
observed18.1/45.0GiB. Both boards now expose bc250work/docs/README.md/versioned
snapshots; [workspace/publication map](../../../docs/WORKSPACE_LAYOUT.md).
Stale source export revision labels remain an explicit release-audit concern.

Storage refresh20261018 (code780e67ceb) recovered11.96GiB: free1.08->13.04GiB.
Explicit user cleanup authority supersedes retention only for inspected obsolete
compiled outputs:8,801 .o/.a/.so files in three inactive Torch build trees,
21 Triton bin/test compiled files,90 synthetic CLI fixture files and9 stopped
project build containers. All source/generated source/config/directory/link paths,
current A09 build/shared third_party source, ROCm build, caches, installed binaries,
images/releases, user data/recovery/compat/boot fallbacks/.105/src remain. Missing
retired generated targets must rebuild; deleted synthetic payloads require a new
fixture label.101 protected hashes unchanged; all images/user neural-p1/current/
previous/runtime metadata unchanged. Five8-group current/fallback/fresh-Triton GPU
gates pass,600s guards ACK, whole-batch dmesg identical. No native build/cache reset/
package/default/boot change or active job/container/guard. [Exact plan/journal and
report](bc250_storage_cleanup_20261018/README.md). Next N2.12 unchanged; CPU Python
fault remains unresolved/separate from fixed VM-heap/CSR.

Storage cleanup20261016 (implementation/evidence `a26b5621b666d08dc18257208a6a68d21b7bb21e`)
recovered29.13GiB on .104; available12.2->41.3GiB. User explicitly authorized
retiring inspected obsolete payloads,56 stopped containers,2 image records and
our synthetic test binaries/2 orphan contexts. Runtime hashes stayed unchanged;
the active runtime and retained frozen fallback passed bounded GPU gates; `previous` now selects that
newer fallback. All sources/incremental builds/caches, genuine user checkpoints,
user neural container, boot fallback, .105 and `src/` remain. Local Alpine/gfx1103
artifacts belong to a separate project and were excluded. Full deletion journal,
protected scope, limitations and test evidence:
[storage cleanup report](bc250_storage_cleanup_20261016/README.md).

`.104` exported source trees are gitless. Inspect `WS/source/GIT_REVISION`,
`WS/source/rocm-libraries/GIT_REVISION`, and both Torch GIT_REVISION files.
`SOURCE_PRIORITY2.json` records 51 transferred file hashes and code provenance;
workspace-root/dev/candidate copies exist. `SOURCE_PRIORITY1.json` and
`SOURCE_CANDIDATE.json` retain their older meanings. Source archives are baselines,
not magically updated by patching extracted trees. Generated torch.version's
Git label may retain the original export revision; use source manifests instead.

Reconstruction:27 library patches and four system patches in
`patches/amd-mainline/{rocm-libraries,rocm-systems}` reconstruct the exact A01
component trees. Two base Torch patches and four Triton patches remain in
`patches/{bc250-pytorch,bc250-triton}`. The legacy P2 helper does not include three
later SDPA file deltas in Torch4ee6571; its current overall mismatch is retained
in the A01 reconstruction report as a separate packaging gap. Use the complete
qualified patched Torch Git history/source, not that incomplete patch subset.
Apply each recorded component series in order. The P2 reconstruction helper verified
exact library/system/Torch trees using private indexes; it did not recreate the
entire compiler or claim a clean-room full rebuild. Nested committed branches
and `.smrev` metadata preserve the port independently from root submodule pins.
Do not reset sources or casually change pinned submodules. N1.1 adds
`SOURCE_N11.json` (7 verified changed files, qualified root/library revisions,
active hipTensor hash) on the P2 snapshot; P2 manifests/candidate stay unchanged.
N1.1 library reconstruction matched all18 ordered patches exactly. N1.2 adds
`SOURCE_N12.json` (8 verified changed files) on N1.1/P2 and exact19-patch
library reconstruction; retained earlier manifests keep their original meaning. N1.3 adds
`SOURCE_N13.json` (8 verified changed files), exact20-patch reconstruction.

## Library patch index

| Patch | Durable change / commit anchor                                                         |
| ----- | -------------------------------------------------------------------------------------- |
| 0001  | Native rocBLAS/Tensile target, capability/library predicates: `55e9abea68b`            |
| 0002  | MIOpen HIP-direct convolution applicability: `b6b37dcdf0d`                             |
| 0003  | Honest hipBLASLt HOST_API_ONLY build: `9ca9c61165f`                                    |
| 0004  | rocRAND native host/device dispatch: `039b79bcb43`                                     |
| 0005  | CK scalar DL/generic CK Tile architecture integration: `3f6e5c01b3c`                   |
| 0006  | Tensile CMake native target validation: `f4302e457a9`                                  |
| 0007  | Omit device-management discovery for host-only Lt: `2efc3717efe`                       |
| 0008  | hipTensor generic-only native profile: `a9e578151e5`                                   |
| 0009  | Execute generic arithmetic cases rather than silently skip: `b35368dd8f5`              |
| 0010  | RPP audio FFTS object dependency: `0a912025cc0`                                        |
| 0011  | Isolate generated headers/read-only sources and generic CK requirements: `e0eb4c93e7a` |
| 0012  | rocPRIM native dispatch in standalone and two hipCCL copies: `a8fc909de24`             |
| 0013  | RPP spectrogram destination row stride: `b9c8ad41e5b`                                  |
| 0014  | Native scalar Lt/tensor/sparse backends and generic CK operations: `a6656565253`       |
| 0015  | Wider types/epilogues/pruning, generic operators, decibel fixes: `7bc253117ee`         |
| 0016  | Lt C++/grouped consumers, CK Tile, hipDNN fusion/provider integration: `f12e43e3f9b`   |
| 0017  | Register-tiled Lt GEMM, tensor unary expansion, sparse activations: `01566e3144d`      |
| 0018  | hipTensor unilateral scalar contraction modes: `681d2f8a25d` |
| 0019  | hipTensor native C32F/C64F scalar contractions and identity/NEG/CONJ: `c6ac34f0ba3` |
| 0020  | hipTensor plan API validation and honest scalar preference contracts: `18983cf7497` |
| 0021  | Sparse Lt zero-stride broadcast/physical compression and span guards: `40e2c04af95` |
| 0022  | Sparse Lt native row bias pointer/stride/type and GPU epilogue: `11e165bd0bf` |
| 0023  | Sparse Lt device FP32 per-row alpha-vector flag/indexing and guards: `a9a1ace297c` |
| 0024  | Sparse Lt INT8 input/output, bounded INT32 accumulation, FP32 epilogue and saturating round-even conversion: `2fd5619d7c8` |
| 0025  | Lt 32-lane subgroup short small/skinny GEMMs and bounded dispatch: `48e1e564d6f` |
| 0026  | Preserve validated launch-grid bound for Lt subgroup dispatch: `a6f7d3d8ba7` |
| 0027 | A01 MIOpen dilated GEMM workspace query: `08ba6f386ff` |
| 0028 | A02 native gfx1013 bundled legacy CK scalar profile: `2093090048f` |
| 0029 | A02 BF16 reduction conversions/scaling and cache-key isolation: `d375d8a2ea7` |
| 0030 | A03 capability-aware optional CK plugin probing: `fcf7a7142af` |
| 0031 | A08 native dense INT8/INT32 Lt, modulo arithmetic and typed contracts: `319a235d6c6` |
| 0032 | rocRAND LFSR113 offset source-comment correction, binaries unchanged: `c29021b40e7` |
| 0033 | Tensile preloaded override code-object hydration, cold regression and exact-layout BF16 preset: `8389197c134` |

System patches: build-time ROCr hotswap default; native ROCdbgapi architecture
registration (`fa95eec5a2`); gfx1013 safe pool default (`dbdb7b8d1c`). The first
patch has a synthetic format-patch ID; its applied tree, not that ID, is evidence.
Torch patches: NHWC average-pool gradient padding (`b36877506d3`), qualified
native Lt backend selection (`4f13bf8647`). Triton patches: widened low-precision
dot (`c815e192`), signed/rounding tests (`63915ab7`), BF16 multiplication fallback
(`270bfb9d`), RDNA1 buffer flags (`ee65ff3b`).

## Component coverage: implemented versus existing native backend

**A09/N4 CPU-first profiler fix installed2026-10-18:** native Torchb823a1b,
final testsda45369, TheRock build669baa5d7/final controllers39f3ed14b. First ROCm
Kineto initialization retains available GPU capability, independent of first
capture selection; no visible GPU keeps CPU-only backend. CPU-first warm/cold,
GPU-first and hidden-device cases pass against candidate/installed. Each regular
GPU window13 kernels/two pinned copies, correlated positive-duration records;
exact profiled/unprofiled output/both gradients, CPU float64 references and
nondefault stream. CPU-selected GPU work omits GPU records; cold CPU capture
leaves Torch lazy init false (HIP availability/backend can initialize). API7/
core7 pass in both mounts, plus fresh installed reactivation. Only libtorch_cpu.so
changes; libtorch_hip.so identical. Independent pip-only candidate
WS/pytorch-profiler-order, ccache20GiB/jobs12/link1/native gfx1013, old full A09
build and shared immutable third_party preserved.104 original identities verified,
old runtime.json and CPU alias pinned via WS/recovery/profiler-order-before;
SOURCE_PROFILER_ORDER.json records active/recovery cohorts.21 guards ACK/600s,
18 successful process gates and three honest failed fixtures retained (small
copy expectation/two invalid -1 visibility values). First controller final hash
check auto-rolled back an old-hash-on-active-alias accounting mistake; corrected
old identity pin, qualified installed hash and fresh reactivation pass. Dmesg
unchanged whole batch. No ROCm DSO/packs/image/kernel/boot/default/.105 changes;
only development Torch link/runtime metadata. Free~10.13GiB/no active jobs.
[Report/results/recovery](bc250_profiler_order_results/README.md); patch0009 mbox
on4af4a15 includes nativeb823 plus two test-only fixture corrections; native
GIT_REVISION staysb823, final testsda453 separately identified. No performance/
LoRA gain or physical CPU-only host/counters/daemon/external concurrency/graphs/
long trace/cross-boot qualification. Python CPU segfault unresolved/separate
from fixed VM-heap/CSR; explicit SDK import aborts are separate. Next N2.13
generic sparse triangular vector/multi-RHS coverage, preserving legacy P2 and
keeping general sparse spsolve/factorization B02 separate.

**A08 completed2026-10-18: native dense INT8->INT32.** Libraries
`319a235d6c6496f0da37be44dfa60eaf31a29f82`, runtime helpers
`8f0dfb671e88b31bc3f5bb4a4958acefdafde556`; Torch4af4a15 unchanged.
The native gfx1013 Lt backend admits I8 A/B, I32 C/D/scalars, COMPUTE_32I,
modulo32 arithmetic, row/column/N/T/padded/offset/batched/broadcast layouts,
host/device scalar alpha/beta and DEFAULT/BIAS/RELU/RELU_BIAS. Existing
`torch._int_mm` works without Torch rebuild/launcher hooks.99 direct exact CPU
cases+11 refusals,16 public Torch configurations/out/custom streams+five
refusals pass; GPU trace confirms gemm_int8.1628 existing floating Lt cases,
core7/profiler7/sparse38/conv30/A0642/A0736/actual2B3 pass.12 guards ACK/600s/
dmesg unchanged,61 protected hashes unchanged. DSO+kpack paired recovery in
WS/recovery/dense-int8-a08-before, SOURCE_DENSE_INT8_A08.json in WS/current.
[Report/usage/timings](bc250_dense_int8_a08_results/README.md). Small eager timing
is mixed versus FP32 GEMM+cast; no universal or LoRA speedup. No quantization
recipe, packed low-bit/QLoRA, integer autograd or compile/graph qualification.
Other scales/compute modes (including public pedantic), pointer vectors and
nonlinear/aux epilogues remain refused. No active job/container/guard deadline;
disk~1.10GiB free, preserve existing sources/builds/caches. No kernel/.105/default
BLAS/SDMA0/heap-off/image change. Historical CPU Python fault remains separate
from fixed VM-heap/CSR. A01–A09 bounded scopes complete; next **N2.7**, existing
small Schur/device-side QR iteration roadmap with default host dispatch retained.

**A09 native Torch GPU profiler — 2026-10-18.** User selected A09 ahead of
unfinished A08. Torch `4af4a1508a6ec0b7b2b29f5e0115f8477c4d1bb5` adds tests on
unchanged native source71a0426; TheRock runtime helpers `4c2edb785` enable
USE_KINETO=1/KINETO_BACKEND=rocm with the pinned rocprofiler-sdk backend.
Seven profiler tests require actual GPU kernels/copies, positive durations,
CPU/GPU correlation, FP32/BF16 output/gradient references, allocation/free events,
scheduled capture, custom streams and CPU-only selection. Profiling is explicit;
ordinary training does not automatically collect traces or become faster.
Candidate short-window setup/teardown-inclusive wall overhead was~4.70x;
not application throughput. Existing convolution/sparse/training/A06/A07/model/
resume gates pass within their recorded scopes. [Evidence and usage](bc250_profiler_a09_results/README.md).
Hardware counters, broad simultaneous/external profiling, compile/graphs,
long traces and cross-boot absolute-clock calibration remain unqualified.
Final exclusive-lock switch to WS/pytorch-profiler-a09 passes four installed
gates;47 protected old/runtime/SDK/app files unchanged. Candidate CPU/Python
DSOs changed, GPU DSO byte-identical.14 guards ACK/600s/dmesg unchanged, no active
agent job/container/deadline. Recovery WS/recovery/profiler-a09-before and
SOURCE_PROFILER_A09.json retain provenance. Disk~2.03GiB free; preserve old build
and shared read-only third-party sources when rebuilding.
A08 remains next, then N2.7. Historical Python CPU fault stays unresolved and
separate from fixed VM-heap/CSR corruption. No .105/kernel/BIOS/image change.

**A06 correction and A07 head256 qualified on .104 — 2026-10-18.** Torch
`71a0426566bc9b952d9aa9b41f4a9d79ce46e0e8`, runtime helpers `e4f4b3854`.
Low-precision/autocast full gated-delta uses reference-ordered GPU preparation and
scan; non-autocast FP32/native standalone preparation remains. All372 nonzero
adapter gradients match exactly in BF16 actual2B65/1024 fixtures. A07 extends
explicit attention through D256: scalar tiled FP32/FP16/inference, BF16 D>128
training math-forward/native-LSE+tiled-backward hybrid.36 operator/dispatch tests,
42 A06 tests, four exact autocast cases, three actual-model train/checkpoint cases
and exact fresh/resume CLI checks within math and tiled pass. Actual-model A07
worst gradient relativeL2 is2.17%/1.31%, within unchanged4%;12 head256 calls prove
forward+checkpoint recomputation. The fixture now explicitly calls `.train()`;
earlier actual-model tests that only enabled checkpointing did not establish its
execution. BF16 hybrid lowers isolated peak memory but is slower; forward temporary
allocations remain quadratic. Math stays default; `--attention-backend tiled` is
an explicit new-run option with a distinct fingerprint. No automatic speedup claim.
Full FLA/dropout/compile/graphs/higher-order remain unqualified. [Report and evidence](bc250_a06_a07_results/README.md).
Final exclusive-lock deployment passes eight installed gates; five source/app
hashes match and39 protected runtime hashes remain unchanged. Recovery
`WS/recovery/a06-a07-before-r4`, provenance `WS/SOURCE_A06_A07.json`. At resume
r2 was active; documented one-shot r3 base/GPU/VMM recovery gates passed, with r2
still persistent fallback.32 GPU guards all acknowledged with unchanged per-run
dmesg; failed harness/numerical/legacy-caller attempts retained separately. No
active agent job/container/deadline. Next primary **A08**, then A09, then N2.7. Historical Python CPU segfault remains
unresolved and distinct from the fixed VM-heap/CSR corruption. Earlier failed A06
review and original timing evidence below remain historical and are superseded
only within this corrected scope.

**A06 recheck2026-10-17: BF16 LoRA qualification reopened; production unchanged.**
The original actual2B test initialized LoRA B=0, so its372 named gradients included
zero A gradients. New seed61718/std0.001 nonzero-B fixtures exercise372 nonzero
finite A/B gradients each. FP32/65 worst/global relativeL2~9.54e-6/3.81e-6 passes;
BF16/65 worst4.00217%, BF16/1024 worst5.42540% exceed the unchanged4% per-adapter
limit (global1.574%/0.870%). Do not call broad/resumed BF16 LoRA parity complete.
40 earlier independent operator tests remain bounded evidence, not invalidated.
Exact ATen preparation control in the same composition gives zero model-gradient
difference at65; native-forward/exact-backward and inverse swaps support FP32
preparation rounding/derivative-order amplification through BF16 as the next
investigation. Restoring original ATen scan structure still fails4.002%/5.573%;
candidate retained but NOT installed. Original ba6cbcd/module9ecb6f9 unchanged.
Forward phase1.341ms of27.384ms(~4.9%); backward19.836ms(~72.4%);196 batched-mm
calls unchanged. Native event-instrumented total26.594ms; coarse event envelopes,
not kernel-only time. Five actual nonzero-B1024 pairs median7.778/7.317s, pair
ratios0.9887–1.0756; diagnostic timing on a failed numerical gate, not deployment
or universal gain evidence. [Review](bc250_gated_delta_a06_review/README.md)
contains exact outcomes, counters, failed guards and source hashes.
Six review guards ACK/600s/dmesg unchanged53f83f7b; three retained numerical/
instrumentation failures.40/41 prior protected hashes match. Independent app edit
observed in finetune-n7/replica/bc250_ops.py at00:08:37UTC: eddfea40->d058b14a,
now enables A06 by default via BC250_GATED_DELTA=1. Origin not established; edit
preserved. Our test uses separate pinned old helper. For new ordinary training,
use inner bc250-train launcher option --env BC250_GATED_DELTA=0 before the script;
host shell assignment alone is not inherited. Current app integration is unqualified.
No prefix/module/DSO/kernel/default/boot change by this review; .105/fallback and
user src/data unchanged. Raw WS/logs/gated-delta-a06-review-* and ignored local
kernel/artifacts/gated-delta-a06-review/. No active job/container/guard deadline.
Exact next **A06 numerical parity**, before A07: reproduce seed61718 nonzero-B
65/1024, investigate reference-ordered Q/K normalization/reductions/derivatives
and decay scan rounding, keep4% model gate and independent operator refs. Full
native solve/scan fusion and FLA remain unqualified; A07–A09/N2.7 queued. Historical
Python CPU fault remains separate/unresolved; VM-heap/CSR corruption remains fixed.

**A06 completed2026-10-17 (first native substage):** Torch
ba6cbcd2fafac1a6503f1f580033fbf72ad2a4f9 / TheRock helperse61d609d3658ee984907841f4fe1dc8ecdf66baa
installed .104. Explicit torch.nn.bc250_gated_delta fuses Q/K normalization/scaling,
beta*K/V, padded/chunked layout and cumulative log-decay; native scalar wave32
forward/backward covers all Q/K/V/g/beta. ATen triangular solves, GPU dense matmuls
and sequential chunk scan remain. Not a complete FLA backend or automatic trainer
replacement.40 candidate/installed CPU-float64 stage and independent recurrent
reference tests pass, plus69 A05 and seven core groups. Tiny FP32/BF16 LoRA with/
without checkpointing and actual Qwen3.5-2B BF16/rank16/checkpointed65+1024tokens
pass all372 adapter-gradient comparisons; max relativeL2~3.23%, not bitwise equal.
Three paired synthetic1024-token forward/backward medians7.653->7.390s(~3.6% faster)
and peak11,976,548,352->11,745,457,152bytes(~220MiB lower). Not dataset/optimizer/
convergence/resume or universal speed evidence. Installed four operator fixtures
show prep1.34–1.88x/full1.02–1.12x ratios; exact protocol/sample data in
[report](bc250_gated_delta_a06_results/README.md).
Twelve guards all ACK/600s/dmesg unchanged53f83f7b, including three retained
model-adapter flag refusals.210 cached candidate/installed variants audit native
gfx1013/wave32/no-MFMA/WMMA. Spin nine targeted static/format checks and lintrunner
-a pass; no full-project typing/CI claim. Exclusive-lock additive module install,
41 protected hashes unchanged; DSOs, shared consumers and original trainer/helper
unchanged. Recovery WS/recovery/gated-delta-a06-before; SOURCE_GATED_DELTA_A06.json
in WS/current. Module SHA9ecb6f9014bc139e54cffe0fbb2c80f782035e57a3149e3c9a6b31fe445e8b2b.
Full local Git/incremental patch0006/bundle on6ed3d12 preserved; old SDPA3-file
reconstruction gap remains. No C++ build/image/kernel/reboot/default change;
existing Triton cache retained and ccache untouched. Raw WS/logs/gated-delta-a06-*
and local ignored kernel/artifacts/gated-delta-a06/. Disk8.4GiB free. No active
agent job/container/guard deadline. Existing LoRA command retains the reference
warning until explicitly integrated; packed/cache/higher-order/compile/graphs and
further native solve/scan fusion remain unqualified. Historical Python CPU fault
remains unresolved/distinct from fixed VM-heap/CSR. .105/BIOS/r2fallback/SDMA0/
heap-off/math/default dense BLAS/user src/data unchanged. Exact next **A07**:
inspect current tiled SDPA D<=128 guards and head256 Qwen3.5 geometry; implement
and qualify D256 output/QKV gradients with bounded GQA/mask/all-masked/LoRA refs,
keeping math default. Dropout/compile/graph contracts remain separate. A07–A09
before N2.7; K6 completed.

**Sparse/lint follow-up completed2026-10-16:** user-requested A04/A05 gap closure
before A06; Torch6ed3d12b184a27d21c0fa41d43269ff6bd17f1e0 / TheRock runtime
helpers4f478cc1d6acd9aa901f011a17d7c79856dccd36 installed .104. Raw native
FP32/FP16/BF16 compress/mm first-order gradients now cover retained slots, dense
input, bias and scalar/vector alpha. Public FP16/BF16 conversion preserves dense
source-weight edge; transposed row-sparse weights reconstruct on GPU for dense
GPU backward, avoiding incorrect transpose pruning. Retained zeros and absent/
undefined optional tensors handled. Public wrapper initialized lazily to preserve
Torch import order. Candidate/installed38 tests pass CPU references:18 raw,
16 public/linear rectangular/frozen/trainable,3 gradient-subset groups(5 each),
retained-zero group. Installed20 A04/120 matmuls,69 A05, seven core and30 conv
regressions pass.12 forward +12 training fixtures also pass CPU references.
15 guards including retained import/optional failures ACK/600s/dmesg identical
53f83f7b. Reversible exclusive-lock current/torch switch to WS/pytorch-sparse-followup;
full original A05 WS/pytorch-a04 and old WS/pytorch-work preserved.31 protected
runtime/DB/kpack/original/frozen/trainer/consumer hashes unchanged. Recovery
WS/recovery/sparse-followup-before; SOURCE_SPARSE_FOLLOWUP.json in WS/current.
libtorch_hip SHA8ffd32eeaa6548ef896a5f56773255a26a985398a4a7bf811463bcb1dc043b90.
Verified lint tools initialize locally in isolated uv caches;12 targeted configured
static/format checks and required lintrunner -a pass. Spin0.18/lintrunner0.12.7;
PYTORCH_LINT_SKIP_CODEGEN=1 avoids separate codegen. Not full-project type/CI claim.
A05 function/class ASTs unchanged by formatting; import order canonicalized.
Performance measured, not a blanket speedup: prepacked eager forward dense/sparse
0.279–1.585 across12 fixtures, compression separate; forward+backward sparse
1.92–3.36x slower than dense in12 small fixtures. Keep ordinary dense BLAS/training
unchanged and structured sparse explicit. GPU backward uses dense intermediates
and metadata sync; no hardware sparse accelerator or application speedup claim.
Raw valid floating first-order scope only; integer/higher-order/forward AD/compile/
export/graphs/broad shapes/concurrency/stability remain unqualified. Sources,
exact lint/timing protocol, errors and evidence in [follow-up report](bc250_sparse_followup_results/README.md).
Raw WS/logs/sparse-followup-* and local ignored kernel/artifacts/sparse-followup/.
Build r1 concrete-size codegen and r2 C++ operator/ADL failures retained; r3/r4
success, final bc250-sparse-followup-build-r4/image28ad36e6/pip-only/native/noGPU/
ccache20GiB/jobs12/link1. Log WS/logs/sparse-followup-build-r4.log. No active agent
build/test/container/guard deadline. Disk8.7GiB free; avoid unnecessary clones.
Full local Git, patch0005/reverse-apply/verified bundle on6181d00 preserved; older
SDPA3-file reconstruction gap remains. Kernel/image/boot/BIOS/.105/user src/data/
SDMA0/heap-off/math/default rocBLAS/r2fallback unchanged. Historical Python CPU
fault remains unresolved/distinct from fixed VM-heap/CSR. Exact next **A06**:
inspect existing gated-delta/FLA reference/model shapes, profile bounded forward/
backward substages, implement one scalar/wave32 stage with independent output and
Q/K/V/gate gradients. A06–A09 before N2.7; K6 completed.


**A05 completed2026-10-16:** Torch6181d00b5beb8cc9fd6800da7a19c8abf756b30d,
TheRock helpers869463155599a1c836bef10fb6089cbaa511058c installed .104.
New opt-in torch.nn.bc250_training_ops exports causal_conv and chunked_causal_loss,
without model/version hooks. Native gfx1013 K1–4 FP32/FP16/BF16 forward/dx plus
new deterministic FP32 partial-reduction dw/db; strided/custom-stream and autocast
handling; unsupported valid conv falls back to ATen. Chunked loss recomputes FP32
probabilities, validates labels/counts/kwargs; original low-precision logits remain.
Final candidate/installed69 TestCase tests pass:48 all-gradient conv,3 no-bias,
4 AMP/checkpoint/scaled-update,12 loss, dispatch/8-refusal group and tiny-model
CPU loss/all-gradient plus exact AdamW/parameter/GPU-RNG serialized resume.
Installed seven core groups and A04 sparse20/120-matmul regressions pass. Six
GPU guards all ACK/600s/dmesg identical53f83f7b;30 protected hashes unchanged.
Only one Python module atomically added to active WS/pytorch-a04 under exclusive
lock; Torch/ROCm DSOs and shared/frozen consumer bundle unchanged. Recovery
WS/recovery/training-a05-before; SOURCE_TRAINING_A05.json in WS/current.
Old finetune-n7 trainer/bc250_ops.py remain unchanged; new API requires explicit
import/calls. No automatic general conv/loss replacement, FLA, higher-order,
compile/graphs/export, large-model or performance/stability claim. Fixed8x AMP
scaling test does not qualify dynamic GradScaler overflow/growth. Existing real
Qwen/CLI51-gate evidence belongs to the old helper, not this module. Source/hash/
usage/evidence in [A05 report](bc250_training_a05_results/README.md), raw logs
WS/logs/training-a05-* and local ignored kernel/artifacts/training-a05/.
Full Torch Git/incremental patch0004/verified bundle preserved; prior SDPA3-file
bundle gap remains. Lint initialization still blocked, attempts retained; syntax/
whitespace pass. No active jobs/container/guard deadline. Disk15.9GiB free.
Kernel/image/boot/BIOS/.105/user src/data/default dense BLAS/SDMA0/heap-off/math/r2
fallback unchanged. Historical Python CPU fault distinct from fixed VM-heap/CSR.
Exact next **A06**: inspect recorded Qwen3.5 gated-delta reference/model shapes,
profile bounded forward/backward substages; implement one scalar/wave32 substage
with independent output and Q/K/V/gate gradient refs before deployment.
A06–A09 before N2.7; K6 completed.

**A04 completed2026-10-16:** Torch8cdfcc8e0feb3689a1ec1d292539af083b13ea94,
TheRock helpers b5879aefa7990928270645676e66cc40d972de6b installed on .104.
Fresh candidate configure finds the existing hipSPARSELt package; no discovery
source workaround needed. Native gfx1013 gate/FP32, HIP Python version/search,
truthful compiled availability, initialized search fields and pack/layout/device/
bias/alpha guards implemented. Candidate and installed20 tests pass:120 small
CPU-reference matmuls, independent pack decoding, FP16/BF16 public wrappers,
search and14 refusals. Installed30 convolution cases, seven core groups and one
tiled SDPA output/QKV-gradient smoke pass. Eight guards including retained test
failures ACK/600s, dmesg unchanged53f83f7b.18 protected ROCm/DB/kpack/original-Torch/
frozen-fallback hashes unchanged. Exclusive-lock atomic current/torch switch to
WS/pytorch-a04; original WS/pytorch-work fully preserved. Recovery metadata in
WS/recovery/torch-a04-before. SOURCE_TORCH_A04.json in WS/current; source/hash/log
scope in [A04 report](bc250_torch_a04_results/README.md). Native scalar same-dtype
FP32/FP16/BF16/INT8; valid1:2/2:4 inputs, compression is not pruning; GPU scalar/
row-vector alpha, typed/FP32 bias. Default dense rocBLAS unchanged. No sparse
hardware speedup, sparse-autograd/compile/graphs/large-shape/stability claim.
Retained path/CPU-alpha/version assumptions and stale-HIPify/live-script build
failures; final r4 pip build succeeded with ccache20GiB/jobs12/link1. Lint blocked
by missing offline initialization; attempts logged, no lint pass. Full Torch Git,
incremental A04 patch/bundle retained; old P2 SDPA3-file bundle gap remains.
Historical Python CPU fault distinct from fixed VM-heap/CSR. No active jobs or
guard deadline; retained build containers stopped. Disk16.1GiB free. Kernel/image/
boot/SDMA0/heap-off/math defaults/r2fallback/BIOS/.105/user src/data unchanged.
Exact next **A05**: inspect existing Qwen3.5 causal-conv/loss helpers and51 prior
operator/model/resume gates; reusable opt-in API, trainable conv gradients and
CPU/checkpoint/AMP references. A05–A09 before N2.7; K6 completed.


**A03 completed20261016:** libraries fcf7a7142af8c258164e106e66bf893c7f563d15,
source/helpers7d9176a45 (initial452ac72e7); installed .104. Optional CK probe now
respects build flag/existing solver whitelist, explicit path retains diagnostics.
No modern plugin kernels added; B07 remains optional. Genuine native gfx1013_20
system FindDb16 keys/58 CPU-checked immediate solutions, one searched HIP-direct
PerfDb record and41 compiled kernel records installed with provenance. Normal
unforced alternatives checked after the isolated search. Read-only installed
replay uses empty user FindDb/PerfDb and zero new user kernels; searched config
load logged. Unseen geometry fwd/dx/dw falls back and writes three user records.
30 Torch convolutions, A02 smoke3 and seven core groups pass. All16 GPU guards
ACK/600s and dmesg identical. Initial WAL-mode package failed readonly replay;
consistent SQLite backup now uses DELETE journal mode; prior cache/package kept.
Initial deployment precheck mistook expected Info-level empty-user notices for
system warnings and stopped before mutation; severity check corrected. No global
warning suppression. Source/DSO and three native DBs plus manifest installed under
exclusive runtime lock; rollback WS/recovery/miopen-a03-before prepared/not used.
DSO SHA3a751adc149778f061198014d6f086a75c33e7640b9473dd75bf14464652afce,
11 protected runtime/fallback/shared-heuristic hashes unchanged. Exact payloads:
[report/manifest](bc250_miopen_a03_results/README.md); raw WS/logs/miopen-a03-* and
local ignored kernel/artifacts/miopen-a03/. Binary bundle retained locally/.104;
text assets and hashes committed.30 library patches reconstruct exactly. Existing
historical private caches preserved, not promoted. No universal tuning/speedup/
coverage or cross-board/compiler qualification; compiled search candidates are
not all numerically checked. Python CPU fault separate from fixed VM-heap/CSR;
A01 clock/reboot/trace and Torch SDPA patch-bundle limits retained. No active agent
build/test/container/deadline. Exact next A04: inspect patched Torch hipSPARSELt
CMake discovery/version gates and qualify real compress/sparse-mm integration.
A04–A09 before N2.7; K6 completed. Kernel/BIOS/.105/user files unchanged.

**A02 completed20261016:** libraries d375d8a2ea7280170455d133058c3cf8e16f58b6,
TheRock source/helpers d17427bfeff6ab7204855a753a883c9a66fbf51b; installed .104.
Bundled legacy CK accepts the native gfx1013 target, uses RDNA1 descriptor0x31014000
and scalar FMA, disables unsupported packed-dot asm defaults and rejects XDLOPS.
Wave32 assembly checked. BF16 enum/raw-buffer/load conversions and compute-
precision alpha/beta epilogues corrected. Reduction cache keys versioned because
embedded header changes otherwise reuse older compiled kernels; caches preserved.
Installed54 numerical FP32/FP16/BF16 reductions with FP32 compute,3 unsupported
BF16 compute refusals and12 FP32/FP16 LSTM/GRU/tanh-RNN output/all-gradient cases
pass. Thread/wave/block/multiblock, scales/beta, guards, one strided custom stream
and FP32 min/max indices covered.30 convolution,32 stable event pairs and seven
core reference groups pass. All14 diagnostic/GPU guards ACK/600s, dmesg identical,
including retained baseline/intermediate failures. No general speedup/coverage
claim; BF16 RNN, wider NaN/index/compute/recurrent contracts unqualified.
Only versioned MIOpen DSO replaced under exclusive lock, installed SHA f4b1a10ce20b;
eight protected HSA/HIP/Torch/BLAS/kpack/frozen-fallback hashes unchanged. Recovery
WS/recovery/miopen-a02-before; installed WS/SOURCE_MIOPEN_A02.json.29 library
patches reconstruct exactly. [Report and evidence](bc250_miopen_a02_results/README.md).
No active agent build/test/container/deadline. Python CPU fault remains distinct
from fixed VM-heap/CSR; prior Torch SDPA patch packaging gap retained. A01 clock
limits and remeasurement caveat remain. Exact next A03: inspect actual gfx1013_20
DB/cache records, bounded tuning/replay/package provenance and capability-aware
optional plugin probing; missing system DB/plugin warnings remain A03/B07. No
fabricated DBs/architecture renaming. A03–A09 before N2.7, K6 completed.

**A01 completed20261016:** source/helper380a403a1, systems329eaf7266,
libraries08ba6f386ff installed in current development prefix. Direct C API and
retained events isolated an approximately14,360s shader/KFD counter epoch offset,
not a Torch dispatch fault. ROCr's existing epoch correction now applies to native
gfx1013, with Linux detection requiring a completed timestamp >1s in the future.
MIOpen's obsolete dilated-GEMM workspace exclusion removed; cold query480 bytes,
warm hybrid zero-workspace selection remains valid. Installed15 direct CPU fwd/
dx/dw configurations,30 Torch convolutions,32 stable event pairs/six raw-counter
references and seven core groups pass;21 total diagnostic/GPU guards ACKed and
dmesg identical. Failed attempts retained. No kernel/boot change; old DSOs and
metadata in WS/recovery/miopen-a01-before, active SOURCE_MIOPEN_A01.json records
exact hashes/revisions. Frozen fallback/HIP/Torch/BLAS/kpack hashes unchanged.
[Report, source, failures and limits](bc250_miopen_a01_results/README.md).
Absolute CPU/GPU trace correlation, opposite epoch/reset/reboot unqualified;
earlier GPU-event speed ratios (including N1.8) require remeasurement. Numerical
qualification and host-wall-time results remain separate. A02 legacy CK reduction/
LSTM backward is corrected above; A03/B07 plugin/DB gaps and Python CPU fault remain
open, distinct from fixed VM-heap/CSR. Source export component GIT_REVISION labels
updated; the unchanged superbuild baseline label remains7f73 with effective port
source380a403a1 recorded separately. No active agent job/deadline; N2.7 queued.

**Baseline Torch dependency/dispatch audit2026-10-16**, source7e39ed699
(A01–A04 updates above supersede its resolved failure/availability status):
[support matrix and ordered A01–A09/B01–B07 queue](bc250_torch_support_matrix.md),
[raw evidence](bc250_torch_support_audit/README.md).21 DSOs examined,19 load;
RCCL/gfx1013 CK grouped-conv plugin absent. All3520 dispatcher names recorded,
not a coverage percentage.51 unique targeted configurations:35 bounded passes,
6 reproduced MIOpen failure variants across two families,10 unavailable/refused.
Group1 convolution needs grad: missing workspace/elapsed<=0/no suitable solver;
GEMM-off diagnostic still fails. LSTM backward separately fails HIPRTC because
MIOpen's bundled legacy CK reduction config has no gfx1013 macro. This is not the
standalone CK Tile port or optional CK grouped-conv plugin. No gfx1013 system
FindDb/PerfDb/KDB; only three shared heuristic files, user ufdb/ukdb exist.
HIP20 multiprocessors reflects40-CU WGP mode; gfx1013_20 db key is correct.
Torch hipsparselt_DIR-NOTFOUND despite installed CMake package/DSO; direct N1
qualification does not imply Torch integration. Valid dense INT8 reaches Lt but
is unsupported; BF16 FFT/general sparse solve/efficient-flash/head256 attention/
distributed/Kineto gaps retained. FP16 power-of-two FFT, CSC/BSR, fused AdamW,
AMP/compile/graph and ordinary operators pass bounded probes. Existing Qwen
conv/loss/resume changes remain app scoped. No runtime/source-library/kernel
change or native rebuild; guards ACKed/dmesg identical/no active agent job.
Baseline reproductions were followed by A01–A04 corrections above. These baseline
counts remain historical; exact next A05. Selected Torch/library queue precedes N2.7.

| Component                                   | What exists and was executed                                                                                                                                                                                 | Boundary / where to continue                                                                                                                                |
| ------------------------------------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------- |
| LLVM/Clang/device libs/COMGR/HIPRTC         | Existing upstream gfx1013 support; native compile/runtime compile, builtin probes, Triton instruction-selection checks and GPU kernels                                                                       | Not a newly written compiler target; wider builtin/debug/offload suites remain                                                                              |
| ROCr/HIP/CLR                                | Native allocation/copy/stream/churn tests; safe pool default; VMM and async APIs retained                                                                                                                    | SDMA/reset/VM heap/fault migration are separate controlled experiments                                                                                      |
| Low-precision atomic APIs                   | Existing HIP CAS fallback qualified by `hip_atomic_low_precision.cpp`, contention/returned values/neighbor guards                                                                                            | Do not patch working atomic headers or claim native low-precision atomic instructions                                                                       |
| rocBLAS/Tensile + hipBLAS                   | Native identity/scalar predicates, 570 fallback kernels; P2 BLAS18; N3.2 cold override loader repair and exact-layout BF16 NT solution-490, full CPU reference/372 model gradients/CPU AdamW/paired timing                                                                              | No complete tuned database/BLAS suite; N3.2 tunes one1024-token model layout only, changes reduction order. hipBLAS uses this backend, no new arithmetic kernel.                                                                   |
| hipBLASLt | Prior964 C/C++ cases +N1.8 new664,403/561 regressions; f32/f16/bf16 same/mixed output, layouts/transposes/broadcast batches, host/device scalars, bias/activation/AUX/DGELU, grouped lifetime/events; tiled/register and short subgroup paths.36 paired graph timing fixtures | Subgroup K<=64 and min(M,N)<=4 or both<=32/area<=512; FP32 tree order differs. Selected graph kernel ratios1.14–1.86x, controls~flat; no whole-Torch/uncaptured latency claim. FP8/INT8/FP64/device groups/tuning unsupported. Default Torch BLAS remains rocBLAS. |
| hipTensor generic                           | Historical154 rank-two and23 higher-rank reduction/permutation/binary/trinary checks in generic-only profile                                                                                                 | Wider rank/layout/operators remain; these counts are separate from contractions                                                                             |
| hipTensor contraction | N1.1 real273 API cases (253 numerical +20 descriptor); N1.2 complex450 (426 numerical +24 descriptor) and273 real regressions. Same-type C32F/C64F, identity/NEG/CONJ, public complex scalars, shared/unilateral sums, padded/permuted layouts, zero/imaginary scalars, span/alias/guard bytes, in-place conjugated C/D,16-mode union with unit axes, plan lifetime and3algorithms×3workspace preferences/type with JIT NONE | N1.3:304 plan API checks +12 numerical executions across four real/complex types; pointer/size/enum guards, atomic failed setters, owning plans, zero required workspace. Scalar cache NONE/JIT NONE/autotune NONE/count0/rank0; alternative requests explicitly unsupported. Broader cache/JIT implementations remain optional. Mixed/widened complex and other complex unary ops unsupported; small-fixture timing is not general performance. |
| hipSPARSELt | f32/f16/bf16 retained; N1.7 INT8→INT8/COMPUTE_32I:134 numerical +16 API;234 scalar/609 broadcast/567 bias/438 alpha-vector regressions. Sparse A/B, layouts/transposes/broadcasts, strip/TILE/partial tiles, FP32 bias/alpha vector and plan/alias/guard checks | INT8 K multiple4 <=262140, FP32 alpha/beta/bias, nearest-even saturation. INT8 sigmoid/tanh enable and widened outputs unsupported. Matching logical batch counts; beta vector/split-K/multi-stream unimplemented. Borrowed vectors/bias live until completion; separate sparse pack, no hardware sparse speedup claim. |
| CK scalar DL | CPU-checked FP32/FP16/signedINT8 GEMM examples; N2.18 FP32 grouped forward128 CPU-reference cases +8 refusals, plain/Add+ReLU, valid strided layouts/guards/private streams | Direct header consumer only; backward, low-precision convolution, other layouts/configurations/performance and automatic Torch/MIOpen routing unqualified. XDL/WMMA/modern CK plugin off. |
| CK Tile                                     | Prior19 generic runs; P1 numerical247: top-k60, LayerNorm18, scalarGEMM160, Pool3D9;400 argument rejections; installed public ScalarGemmKernel                                                               | Explicit generic API, not automatic matrix-kernel replacement; more ops/types/layouts remain                                                                |
| MIOpen | CPU-reference convolutions, A02 native legacy reductions/recurrent gradients; A03 genuine seed FindDb16 keys/58 checked solutions, PerfDb1/KDB41 and normal replay/fallback | Small recorded scope; wider tuning/types/layouts/NaN/index/recurrent contracts unqualified; native-default CK plugin absent; explicit bc250-ck installed B07.8; B07.3 private contracts qualified; B07.4 default DL slower; B07.5 per-ID block256 wins two of six NCHW fixtures1.09–1.14x; candidate private; B07.6 genuine native Search/PerfDb and fresh public replay1.085–1.135x on two fixtures qualified; B07.8 explicit bc250-ck installed/rollback and daily+RCCL/plain consumers qualified; B07.9 isolated FP16/FP32-accum factories pass656 references; group-major missing installed symbol supplemented privately, B07.10 isolated FP16 facade passes724 half references/typed contracts and retained gradients,66 checks; installed option stays FP32-only; B07.11 public FP16 tuning/replay qualified, one of six shapes1.17x; original HIP-direct legacy half accumulation misses strict bound, source repaired and PLAN-02 ordinary rollout qualified on both hosts; B07.12 coherent FP16/FP32 plugin+private repaired MIOpen archive qualified in two build-free extracts/daily+RCCL;78 top checks/48 nested, explicit installation next; ordinary native precision repair active via PLAN-02 |
| hipDNN providers                            | Historical pass-by-value5 +HIP9 +MIOpen9, scalar-expansion14; P1 Lt48 +HIP18 +3D18 +fusion16; six host parameter tests                                                                                       | Normalization/training/backward/provider/fusion subsets only. Mixed FP32 virtual/low-precision output rejected; no automatic Torch provider routing claim   |
| RPP/FFTS                                    | Native image layouts16 plus audio26; decibel zero/reference/reduction fixes, spectrogram row-stride and FFTS build/link fixes                                                                                | Broader image/audio/volume operations, types and layouts remain                                                                                             |
| rocPRIM/hipCUB/rocThrust                    | P2 INT32 42/48/48; N2.16 reductions/scans576/480/480; N2.17 FP32/64 stable sorting/selection656/656/648 with exact bits/indices/counts, explicit special-value policies, guards/streams; Torch62 output/backward consumers | No production defect found; wider types/operators/iterators/determinism/graphs/concurrency/tuning remain. hipCCL is header packaging, not RCCL. |
| rocFFT/hipFFT                               | Existing native backend/AOT cache; P2 FFT30 plus N2.14 public complex96/real72 output+weighted gradients+guarded out, three independent checks and16 direct advanced plans/32 executions | No production defect found; native in-place/planar/callbacks/giant plans and broader performance remain; B01.1 explicit FP32 promotion helper qualified; native BF16 FFT still unavailable |
| rocSOLVER/hipSOLVER                         | Existing native/BLAS backend; P2 solver32 plus N2.1 complex Hermitian eigenvectors22: batch2/17–257, triangles/strides/streams, repeated subspaces and CPU gradients                                                                                                          | N2.1 native eigh22; N2.2 nonsymmetric eig16 host-LAPACK compatibility only. Native nonsymmetric eig, extreme scaling/conditioning, larger/two-stage and performance remain                                                                                         |
| rocSPARSE/hipSPARSE | Existing native backend: P2 Torch25+legacy16; N2.13 public triangular433 and direct generic96 fixtures/192 solves, exact pivot NaN policy in Torch | B02.1 small explicit GPU dense solve/first gradients separate; native general factorization/more formats remain; BSR64 bounded downcast, triangular autograd/performance unqualified; old corruption fix was CLR |
| rocALUTION                                  | CG40 ordinary cases andfive fresh-process200 passes after pool-default fix                                                                                                                                   | Larger systems, other solvers/preconditioners/concurrency remain                                                                                            |
| rocRAND/hipRAND                             | Native dispatch; P2 RNG28; N2.15 Torch state80/statistics22/contracts20/fresh continuation1 and direct native68 incl CPU Philox, offsets/streams and LFSR113 source-comment repair | No arithmetic defect reproduced; broader generators/orderings, large statistical suites, graphs/concurrent mutation/cross-version replay remain |
| OpenCL                                      | Native CL1.2 kernels, LDS/atomics/images/events/profiling checked                                                                                                                                            | OpenCL2 completeness and other feature combinations remain                                                                                                  |
| OpenMP/Flang                                | Mandatory GPU SAXPY/reduction/task dependency checks; bundled Flang compiled                                                                                                                                 | Fortran device execution and hipThreads remain unqualified                                                                                                  |
| HIPIFY                                      | hipify-perl CUDA translation → native compilation → exact GPU check                                                                                                                                          | hipify-clang requires parser/SDK input qualification                                                                                                        |
| hipFile                                     | Five synchronous/asynchronous compatibility I/O roundtrips                                                                                                                                                   | Not a direct-storage/DMA fast-path claim; registration/concurrency/ordering remain                                                                          |
| rocprofiler/ROCTX                           | Native marker/API/kernel/copy tracing with 3628 APIs,263 dispatches,814 copies, CPU workload correlations                                                                                                    | Counter mapping/advanced profiling/debug consumers separate                                                                                                 |
| AMD SMI / RDC                               | AMD SMI built without EPYC ESMI; disposable generated-header source copy                                                                                                                                     | Read-only telemetry remains incomplete/untrusted; RDC not built/qualified                                                                                   |
| Torch distributed / Gloo | B03.1 CPU31/B03.2 GPU41 local baseline retained. D01 two-physical-host Gloo four-dtype GPU sums and three CPU-reference global-batch DDP/SGD steps:16 checks. D02/D03 same16 each through matched RCCL/installed torchrun command. | Gloo is host-staged. D04 classic FSDP/use_orig_params+AdamW shards804/1608 parameters per rank and same-world/rank checkpoint resume:14 checks. Explicit bc250-distributed on both hosts; D05 12.61M FP32 transformer two-host reference/AdamW/checkpoint gates pass; modes measured in both orders, distributed slower in this fixture. No large-model capacity/general scaling/elastic/FSDP2/HF Trainer proof. Source8d8b18d agent exit lifetime repair active through pinned bc250-distributed bridge; native cohorts unchanged. D06 equal global8/32 FP32 DDP accumulation/no_sync CPU loss/all-gradient/AdamW state/checkpoint references pass; reduction payload4x/8x lower and3.76x/~7x faster than every-micro sync, but still slower than one host. Peak live allocator +51.10MiB/rank. D07 bounded 1.05M BF16/FP16 residual MLP autocast gradients/native scaler/nonfinite skip/state replay qualify within report limits;92 groups/eight guards. D08 classic FSDP BF16/FP16 Linear-wrapped MLP/shard-aware overflow and optimizer/scaler replay pass:100 groups/eight guards,525,824 owned parameter/AdamW elements/rank. Isolated live peak no_sync75.04 vs sync73.61MiB; D09 public env/mount/shm options and pinned agent/worker Torch identity are installed BOTH, with owned tiny Transformers5.2 FP32 DDP CPU-reference training and caller/recovery gates. D10 owned FP32 Bert DDP/FP16 FSDP fresh-process fixed-world replay now matches loss/gradients/optimizer/scaler/scheduler/RNG/cursor exactly across three windows;48 groups/eight accepted guards. D11 actual Accelerate FP32/BF16/FP16 prepared DDP/loader/accumulation/optimizer/scheduler/native save-load pass CPU and exact fresh-process references:78 groups/12 guards. D12 actual Trainer FP32 DDP finds/fixes double loss division through explicit source-pinned BC250Trainer, equal/variable-token CPU references and native fresh resume:78 groups/10 guards. D13 actual Trainer BF16/FP16 MLP CPU math/AdamW/native scaler+fresh replay and BF16 collective finite refusal:68groups/10guards. D14 native Trainer AMP clipping/ALL-gradient CPU predictions, rank-local FP16 overflow coordination, late-invalid/scaler refusal and exact replay pass:118groups/14guards. D15 actual eager Bert MLM BF16/FP16 Trainer token CE/padding/CPU-AMP+FP64/all-gradient+clipping/AdamW/native growth-skip-restart pass:70groups/8guards. D16 actual eager GPT2 causal Trainer AMP SHIFTED CE +native target-count collator/global49/57/57/21, CPU gradients/clipping/AdamW/localzero/restart pass:96groups/8guards. D17 actual native Trainer eval/predict Bert/GPT2 FP32/BF16/FP16 dynamic padding/19-row tail dedup/CPU logits/explicit dataset token metrics:96groups/12guards. Native eval_loss semantics retained. D18 opt-in pre-gather per-example FP64 token statistics passes108groups/12guards andCPU parity; prediction arrays194x/gather inputs~152x smaller in tiny fixtures, slower latency; local logits/label storage/streaming remain. D19 final176groups/16guards: scalar reads12→1 (nonzero retained), strict FP32 vocab32768 evaluation compact18.6–19.6x faster than full gathering in boundedfixtures, D18 similar. D20 reproduces nativeBF16 outliers exactly on CPU; explicit FP32 eval head passes ORIGINAL gate (maxabs.00005874),36groups/6guards. NativeBF16 strictFP64 gate still fails; dense prototype rejected. D21 actual Trainer classicFSDP FP32 Bert CPU ALLgradient/AdamW+fresh EXACT replay30groups/4guards passes; source-pinned empty AdamW restart repaired via public optimizer-aware DCP reader. Parameters/AdamW sharded11282/11279, separate-host checkpoint payload union required. D22 native FSDP globalFP32clip/CPUALLgrad+norm+AdamW/exactrestart andcollectiveInf/NaN/normoverflowrefusals pass68groups/12guards. D23 explicit BF16compute/FP32 FSDP params+reduce+buffers passes64groups/10guards with CPU BF16+FP64 ALLgradient/clip/AdamW/exactrestart, inactive/FP32/NaN controls. D24 explicit BF16 workingparams+allgathers/FP32 masters+reductions passes64groups/10guards with roundedCPU ALLgrad/clip/AdamW and exactrestart; parameter padding11284/11277, no speed/capacity proof. D25 explicit FP16workingparams/nativeShardedGradScaler passes128groups/18guards with CPUALLgrad/clip/AdamW, rank1overflow native skip/backoff, lateinvalid/scaler refusals and exactfinite+overflowrestart. D26 tied GPT2 causal FSDP FP32/BF16/FP16 shifted targets/CPU ALLgradient+clip+AdamW/native overflow/exact restart pass232groups/18guards. D27 explicit frozen FP32 native-adapter policy/CPU ALLtrainable-grad+AdamW/frozen values/exact restart and collective refusals pass97groups/14guards. D28 actual pinned PEFT FP32 LoRA/full base checkpoint/public adapter roundtrip and exact restart/refusals pass116groups/16guards. D29 explicit FP32 nonreentrant checkpointing/ALLgradient+AdamW/frozenvalues/exact restart and checkpointoff control pass163groups/18guards; tiny allocator491008vs546304B. D30 explicit BF16 PEFT workingparams/FP32frozenmasters/CPUALLadaptergrad+AdamW/exactrestart passes116groups/16guards. D31 combined BF16 PEFT/nonreentrant checkpointing/ALLgrad+AdamW/frozenpreservation/exactrestart passes163groups/18guards; tiny allocator459264vs499200B/rank. D32 representative45.9MGPT2 FP32 single/DDP/FSDP refs/timing/headroom pass72groups/12guards; singlefastest,FSDPlowerGPUbuthighercgroup,capacityOPEN. D33 clean training-only memory controls with hash-bound D32 CPU proof pass84groups/12guards. D34 separately qualifies159.8M. D35 staged CPU references qualify352.96M ALLgrad/clip/AdamW and fresh single/FSDP:72groups/14guards; FSDP45.6%lower liveGPU/~0.94GiB more host headroom but~68xslower. D36 per-unit audit passes24groups/4guards with exact paired ALL-state digests and~45.4%lower qualification GPU peak; no native training change. D37 CPU-only post-accumulate streaming passes24groups/4guards with exact FP64 refs/~31%lower CPU cgroup peak. D38 native CPU checkpoint+stream passes32groups/4guards with exact FP64 refs/actual recomputation and18–20%lower CPU peak vsstream. D39 independent CPU-reference clipping passes24groups/4guards with exact native state and removes1.315GiB raw spool/host; native training unchanged. D40 bounded remote reference transport passes24groups/4guards with exact native state, zero target rawcache and96MiB max reader payload. D41 CPU-only1.024B oracle passes16groups/2guards, full finite/roundtrip/hash coverage at9.05GiB cgroup; GPU fit remains unqualified. Larger-than-single GPU fit/1.024B GPU remain unqualified. Frozen FP16/PEFT variants/FSDP2/SDPA/other architectures/largerfit/elastic remain separate. [Usage/evidence](bc250_distributed_d01_results/README.md). |
| MIGraphX | B06.9 source134ee25f6 repairs handwritten wrapper C++20 parsing; private overlay C++17/20 and native async boundary qualified. Installed B06.8 opt-in SDK remains4797a6112; explicit bc250-migraphx launcher verifies identities/read-only mount and both runtime cohorts.165 checks/ten guards/68 GPU graph evaluations; live unregister/register recovery | CPython3.12/glibc/static FP32 fixture scope. Ordinary Torch defaults unchanged; no automatic routing, ORT GPU EP, graph autograd, all-model/dtype/opset/MXR portability or performance claim. [Use/results/recovery](bc250_migraphx_b068_results/README.md). |
| ONNX export/runtime | B06.1 pinned ONNX/Script/IR/ORT CPU installed on .104;181 candidate+181 installed checks plus early33: modern CPU/GPU-origin FP32 graphs, CPU-double/NumPy references, explicit host transfer and downstream HIP gradients | CPUExecutionProvider only qualified; Azure advertised/unqualified, no AMD GPU EP/ONNX autograd/training/W8 export claim. ROCm EP removed upstream1.23; B06.5 isolated native C++ ONNX→gfx1013 FP32 graphs qualified; B06.6 Python bindings/B06.7 relocatable SDK and B06.8 registered opt-in SDK installed in recorded static FP32 scope; B06.11 versioned ORT1.31+gfx1013 package installed via explicit bc250-ort; relocation/metadata/identity/recovery/caller-context174 checks/48 GPU evals pass. Ordinary CPU package/defaults preserved. B06.12 fresh modern static CPU/GPU-origin exports pass112 checks/24 public GPU evaluations in daily/RCCL with four reference routes and12 GPU-only node profiles. Broader models/dtypes/dynamic shapes/graph-autograd/performance remain unqualified. [Use/recovery](bc250_onnxruntime_b0611_results/README.md). |
| Weight-only quantization | B05.6 public opt-in `bc250_quant 0.1.0` installed on .104 daily/RCCL imagee2821cf8:817 candidate+817 installed checks, CPU-reference first gradients/state/AdamW resume and actual B05.5 checkpoint compatibility | Explicit frozen W8 only; native layouts fixed in both a524 cohorts, module first input/bias gradients via unchanged helper. TorchAO/bitsandbytes/NF4/packed INT4/full QLoRA remain absent/unqualified; native direct backward refuses. No model-quality/speed/peak-memory claim. [Use/recovery](bc250_quant_b056_results/README.md). |
| RCCL | B03.7/B05.3 native a524 rccl-single retained. D02/D03 real two-host eth0 Socket Ring/Simple four-dtype GPU sums, CPU-reference DDP/SGD, and D04 FSDP/AdamW state sharding/checkpoint pass. Matched .105 release installed; explicit bc250-distributed registered on both. | Ordinary daily NCCL remains off; old rccl-single loopback profile unchanged. P2P/IB/SDMA/symmetric off; D05 1/8/32MiB full-buffer references, DDP/FSDP numerical/restart and timing/memory gates pass;140 worker groups/18 guards. FSDP halves stored state but reserved pool higher; two-host fixture slower. D06 counted DDP reduction-input payload4x/8x lower with no_sync;192 worker groups/24 guards pass. D07 DDP and D08 FSDP AMP MLP sync/no_sync/scaler pass; gradients/reduction input remain FP32, no automatic compression. D08 native ShardedGradScaler synchronizes rank1-only post-RS nonfinite status so both skip; eight RS calls/4.21MB per backward, no all-gather/total-wire/speed claim. D09 explicit public consumer-env DDP and retained FSDP qualify; native provider/transport unchanged. No numeric FP8, other protocol/transport, offload payload, large-model/general scaling or two-host-over-one-host speedup proof. |
| rocSHMEM/MPI | Not built in current profile | Independent software-transport applicability remains; two-host RCCL works without MPI and is not rocSHMEM/MPI qualification. |
| rocDecode/rocJPEG                           | Current driver does not expose media rings                                                                                                                                                                   | Qualify feasible host decoding/HIP interoperability separately; do not invent an upstream JPEG software fallback                                            |
| rocWMMA / matrix-only engines               | Not applicable as native MFMA/WMMA execution                                                                                                                                                                 | Distinct scalar emulation/math paths could be work; never enable them by architecture spoofing                                                              |
| rocRoller/emulation/other optional projects | No blanket qualification                                                                                                                                                                                     | Audit pinned API/instruction applicability before enablement; CPU fallback must be identified                                                               |

Code roots for actual port backends:

- `rocm-libraries/projects/hipblaslt/library/src/amd_detail/gfx1013_scalar.{cpp,hpp}`
- `rocm-libraries/projects/hiptensor/library/src/contraction/contraction_scalar.{cpp,hpp}`
- `rocm-libraries/projects/hipsparselt/library/src/hcc_detail/gfx1013_scalar.hip`
- `rocm-libraries/projects/composablekernel/include/ck_tile/`
- `rocm-libraries/dnn-providers/miopen-provider/engines/plans/`
- `rocm-libraries/projects/rpp/src/modules/hip/`
- `rocm-systems/projects/clr/rocclr/device/rocm/rocdevice.cpp`
- `sources/bc250-pytorch/aten/src/ATen/{native/cuda/AveragePool2d.cu,cuda/detail/CUDAHooks.cpp}`
- `sources/bc250-triton/third_party/amd/` (verify repository path before edits)

Supported scalar contraction type combinations (A/B share the input type; C/D
share the output type): f16→f16, bf16→bf16, f32→f32, f16→f32 and bf16→f32 use
f32 computation; f32→f64 and f64→f64 use f64 computation. Reduced-precision
accumulation, f16→f64 and narrowed outputs are not supported in this profile.
The contraction source currently accepts 23 real unary operators (19 added in P2).
Low-precision Lt inputs may likewise use f32 outputs; sparse Lt uses its documented
backend-specific FP32 1:2 / FP16-BF16 2:4 compression, not a portable hardware format.

## Torch, training, files and resource utilization

Current application architecture/environment workflow:
[workspace developer guide](../../../BC250_DEVELOPER_GUIDE.md), reviewed2026-10-29.
CPU-only .104 tests verify explicit runtime-site venv bridging (a naive nested
system-site-packages venv cannot find Torch), offline package installation,
overlay/venv persistence and ordinary UID1000 runner execution.11 environment
checks/16 example syntax checks+links pass; no native/GPU qualification change.
Use bc250-train for frozen consumers, bc250-torch for custom environments; preserve
native Torch/ROCm and inspect package origins. Evidence kernel/artifacts/developer-guide.


**B07.12 coherent FP32/FP16 artifact completed2026-11-04:** toolsc64debffb;
  source96e5a7df/Torcha524980/installed native8389197 unchanged. Packages B07.10
  plugin+B07.11 repaired MIOpen, no rebuild/install/register/default change.
  Private runtime/ vs plugin lib/ separates loader search; fresh worker validates
  payload+27 actual native library pins/gfx1013/API5 before same-process runpy;
  parent kept alive/no exec-after-HIP. Eight copied ELFs normalized to relative
  RUNPATH, DT_NEEDED unchanged; source/config/license/resource recipes included.
  Two build-free extracts (spaces) x daily/RCCL, original sources/builds/candidates
  absent:78topchecks/48nested/10finalACK;1496half+1176float refs plus586installedfloat,
  24Torch output/all-first-gradient fixtures.24identity/four seed refusals;
  128half+128float absent-path,736halfAPI/1908runtime,488floatAPI/984runtime pass.
  Genuine mixed dtype DB8keys/profile; FindDb source had extra post-gradient
  entries, select six authoritative half shape keys and copy existing lines,
  preserving originals. Dedicated reference DB keeps seeded replay immutable.
  R1 packaging DB count/R2 metadata lookup failures before GPU; R3 accepted.
  Qualr1 identity passes then cross-bind hardlink fixture fails; host-created
  immutable payload fixtures+independent manifests fix harness, same package r3
  passes qualr2.29localtests/lint/syntax/independent archive+ELF inspection pass.
  CPU2/2GiB packaging/no build GPU/network/cache reset; SH runtime lock/600sgates.
  3965hashes/36links/fourAOTDBs/136backing files unchanged. New pre-existing
  d10514df boot/r2kernel throughout, logs unchanged/no KFD recurrence. Historical
  Python CPU fault/KFD uncertainty distinct from fixed VM heap; no root-cause claim.
  User authorizes stopping lucid_wiles/6fcfde0398e9; stopped/preserved/not removed.
  No .105/user-app access/deletion/reboot; no jobs/containers/armed guards;11totalACK.
  PackagePIDs8781+9525exit1/10202exit0; qual12590exit1/13470+export17389exit0.
  WS/tmp/ck-plugin-b0712/package-r3 and export-r2; WS/logs/
  ck-plugin-b0712-qualification-r2/final.json and ck-plugin-b0712-r2-controller.log.
  Release66651729B SHAa0d47ed4ba4df354cc7ea898a285eefa051f49693667386567ea392f66ddd91e;
  evidence7079195B SHAadb8ec1f0f84ec3bddc122c25ccae89969b982a6c6c5123b794b1c765c13a650;39payloadfiles/505424918B.
  Local kernel/artifacts/ck-plugin-b0712; free12,879,323,136→10,253,754,368B.
  Installed bc250-ck still FP32; ordinary native precision rollout pending.
  [Artifact/use/evidence](bc250_ck_plugin_b0712_results/README.md).
  **Next B07.13:** versioned explicit .104 install/registration and updated
  bc250-ck format2 worker activation, exact identity/refusal/live rollback and
  ordinary daily/RCCL caller-context gates under EX update lock. Retain old FP32
  artifact/tool/version records, ordinary native defaults/.105/user apps.

**B07.11 FP16 tuning and HIP-direct precision repair completed2026-10-30:**
  tools2d8f1dae0/libraries source96e5a7dfc43. Original public218 fails fixed FP16
  bound: H9/W11/C4/K8/g1/f3/index1 actual0.8427734375 vs0.8424274250864983;
  error0.0003460124>bound0.000306262. Source intentionally retains legacy half
  accumulator. Scope is precision, not memory corruption/all old training invalid.
  Native gfx1013 forward now retains float_types FP32 accumulation; other devices/
  backward unchanged. Isolated native-fix-r2 executes exact existing generator/
  one-resource-object/link recipes on fresh original-build copy; no interposition.
  Full patched source byte-identical in embedded library; mapped path/hash proved.
  Public137 vs repaired218, six NCHW shapes/two opposite orders/daily+RCCL:
  tuned CK wins g3f1/C=K16/H32W34 by1.1706–1.1782x; g1f3 moderate parity;
  four others favor repaired direct. No global/model/legacy-vs-repaired speed claim.
  Native Search5iterations/1000ms yields12PerfDb+12FindDb records; exact-ID fresh
  replay/no mutation, invalid-ID copy fallback.444CPU checkpoints/9516explicit
  calls (search internal excluded), maxerror0.00195155665 under fixed B07.9 bound.
  80top-levelchecks/10final ACKguards +24nested mapped-rebuilt-MIOpen checks;
  original/new MIOpen candidate refs724half+580float each, installed586float,
  24Torch output/all-first-gradient fixtures; plugin still forward-only.
  29localtests/lint/syntax pass. Buildr1memoryrefusal; qualr1/r2 original218
  precision fail/repro; native-fix-r1 stopped after hidden396-edge timestamp
  rebuild, retained. Native-fix-r2 resource-only relink passes; qualr3 missing
  sysdeps loader directory refuses before GPU, r4 supplies existing SDK path/pass.
  FourCPU/OMP4/job1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/ro SDK+source/
  no build GPU/network/600s; SH lock.3864hashes/36links/fourAOTDBs/136backing files
  unchanged. Same e327c5de boot/r2kernel/logs clean/no KFD recurrence; historical
  Python CPU fault/KFD uncertainty distinct from fixed VM heap, no root-cause claim.
  Installed native8389197/Torcha524980/FP32 bc250-ck/defaults/images/.105/user apps
  unchanged; precision repair is NOT active for ordinary launches. No reboot/delete.
  BuildPIDs23839exit1/24088+25122exit0; native26573stopped/29211exit0;
  qual24520+25960+29708exit1/30214exit0/export846exit0; no jobs/containers/armed guards.
  13totalACK; WS/tmp/ck-plugin-b0711/{inputs-r1..r3,build-r2,r3,native-fix-r1,r2,
  export-r4}; WS/logs/ck-plugin-b0711-qualification-r4/final.json/r4-controller.log.
  Evidence66256969B SHA0ce6d10f1c369e3ed68389b48562c76f79b633b4dc6474ec025d31102c7ed8d2.
  Free20,477,861,888→12,493,869,056B; copied builds retained.
  [Method/evidence](bc250_ck_plugin_b0711_results/README.md).
  **Next B07.12 (planned):** isolated versioned FP32/FP16 CK+matching repaired
  MIOpen artifact, exact identity/licenses/example DB and two build-free relocated
  extracts/daily+RCCL before installation. Preserve installed FP32/defaults/.105;
  ordinary native precision-repair rollout remains separate compatibility work.

**B07.10 isolated FP16 MIOpen facade completed2026-10-29:** toolsaf4704f69;
  libraries source742ee1e2c0e, installed native8389197/Torcha524980 unchanged.
  Explicit private compile flag reuses matching upstream half dispatcher/FP32
  accumulator and unchanged installed NHWGC archive. No extra factory object.
  Typed exact scalar/descriptors/packed2D geometry/int32-span contracts, half16B
  alignment, pointer overflow/output/scratch overlap and workspace checks; BF16/
  mixed/casts/TF32/determinism/bias/transpose refused. Global9/per-type6float+9half
  IDs;66 required exports plus forward-list API, helpers/GetCastType hidden.
  Daily/RCCL724half refs (656per-ID/64public/four mixed-handle),580candidatefloat;
  368halfAPI/954runtime/64absent-path refusals. Max half error0.00186229 under fixed
  B07.9 quantization+N2.18 residual bound; dyadic final-half exact, guards/inputs pass.
  Installed FP32 adds586native refs; eight half/eight installedfloat Torch output/
  all-first-gradient fixtures pass; plugin still forward-only, native backward.
  Private24+installed20+core7+CPU ONNX4+daily5+RCCL6=66checks/eight ACK guards;
  29localtests/lint/syntax pass. R1 test-only deterministic-wrapper assignment
  compile failure fixed with public setter; R2 build/first GPU qualification pass.
  Local audit initially assumed66 total instead of66required+forward-list export;
  exact set verified, no source/test-contract relaxation. Failed artifacts retained.
  FourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/ro inputs+SDK/
  no build GPU/network/600s.3849hashes/36links/fourAOTDBs/136backing files unchanged.
  Same e327c5de boot/r2kernel/log unchanged; no KFD recurrence/root-cause fix claim.
  Historical Python CPU fault/KFD uncertainty remain distinct from fixed VM heap.
  Installed bc250-ck FP32/defaults/images/.105/user apps unchanged; no reboot/delete.
  Build1PID19855exit1/build2PID20834/qualPID21771/exportPID23540exit0; no jobs/
  containers/armed guards. WS/tmp/ck-plugin-b0710/{inputs-r1,r2;build-r1,r2;export-r1};
  WS/logs/ck-plugin-b0710-qualification-r1/final.json, r1-controller.log.
  Evidence2837154B SHAf54cb396f04ead909da37448521017d77186e90058b3c39c6cf3a925ba5a741a.
  Ordinary free12,592,922,624→12,572,831,744B. [Scope/evidence](bc250_ck_plugin_b0710_results/README.md).
  **Next B07.11 (planned):** bounded paired FP16 public137/HIP-direct timing and
  native Search/PerfDb replay on owned fixtures before versioned opt-in packaging;
  preserve installed FP32 artifact/defaults/.105/user apps. No performance claim yet.

**B07.9 isolated FP16 DL factory proof completed2026-10-29:** tools0e8a28146;
  existing source has half inputs/output with float accumulator. Native shader
  dot2=0/mac_f32=1/wave32, no XDL; nine distinct IDs/layout. Installed archive has
  interleaved FP16 but lacks group-major definition: actual link+symbol audit prove
  gap. R4 compiles only unchanged upstream group-major source into isolated object,
  links unchanged archive. No installed/native admission or arithmetic change.
  Two layouts/streams/groups1+3, five geometries/two patterns plus C512 accumulation
  case:88fixtures/profile,328 GPUcalls+464 refusals each daily/RCCL=656/928.
  144 accumulation calls subset return256.25 vs sequential-half256. Dyadic final
  half-round exact; non-dyadic half-quantization+fixed N2.18 residual bound, max
  abs0.0018510521/max rounded error0.0009765625; immutable inputs/256-byte guards.
  Eight existing Torch FP16 output/all-first-gradient fixtures; installed FP32
  references586+eight gradient fixtures retained. Factory14+installed20+core7+
  CPU ONNX4/daily5/RCCL6=56 checks/eight ACK guards;29 local tests/lint/syntax pass.
  R1 probe name/include, R2 include order, R3 missing symbol fail before GPU;
  all inputs/logs retained, R4/first GPU batch pass. FourCPU/OMP4/jobs1/RAM10GiB/
  mem+swap12GiB/ccache20GiB no reset/ro source+SDK/no build GPU/network/600s.
  3832 hashes/36 links/four AOT DBs/136 backing files unchanged. Same e327c5de boot/
  r2kernel/logs clean; no KFD recurrence/root-cause fix claim, no owned jobs/guards.
  BuildPID13558/14091/14381 exit1; r4PID15015/qualPID15696/exportPID17462 exit0.
  WS/tmp/ck-plugin-b079/build-r4 and export-r1; source/archive audit JSONs;
  WS/logs/ck-plugin-b079-qualification-r1/final.json and r1-controller.log.
  Local evidence3683360B SHA8bfc6d1a4ce85413684170bf7c0c39ddd13982379901f6619f5cdc5531809351.
  Source3616b2f/installed8389197/Torcha524980/bc250-ck FP32/defaults/.105/user apps
  unchanged; ordinary free12,625,784,832→12,595,290,112B, nothing deleted. Historical
  Python CPU fault/KFD uncertainty remain distinct from fixed VM heap. [Scope/evidence](bc250_ck_plugin_b079_results/README.md).
  **Next B07.10 (planned):** isolated matching-MIOpen FP16 forward facade using
  qualified NHWGC factory, dtype-aware scalar/descriptor/alignment/overlap/scratch
  contracts and explicit public solver references. Preserve installed FP32 option.

**B07.8 explicit bc250-ck installed2026-10-29:** tools36da14fb2; accepted B07.7r2
  at /opt/bc250-stack/sdks/bc250-miopen-dl-gfx1013-b077, miopen-dl.json plus retained
  version record; tools/ck-b078 and /usr/local/bin/bc250-ck. Install/register/
  unregister use EX lock, launch holds SH; archive/manifest/payload/runtime hashes
  and image pinned, artifact verifies eight mapped ABI libraries/API5/66 exports.
  Existing caller-context handoff retained; check/info/seed-db/Python scripts,
  mounts/env/envfile supported; no solver forcing or default/native whitelist change.
  Live unregister/refusal/re-register, identity/byte/extra-file/CLI/lock tests pass.
  Main60+ordinary UID1000 env/argv/importdir/backward2+host13=75 checks/10 ACK guards.
  Two same-process public137 CPU-double calls plus accepted probes=586 native
  conv references;244API/492runtime per-ID refusals/304 retained contracts.16 normal
  Torch output/all-first-gradient fixtures across installed/plain daily/RCCL pass;
  backward uses retained solvers. Unchanged tolerance/guards;29 local tests/lint/
  pycompile/diff pass. Plain launches have no plugin env; core/CPU ONNX/consumer
  gates pass.3804 hashes/36 links/four AOT DBs/136 backing paths preserved. Same
  e327c5de boot/r2kernel/logs unchanged; no KFD warning recurrence or root-cause
  fix claim. Historical Python CPU fault stays distinct from fixed VM heap.
  No build/cache reset/deletion/reboot/.105/user-app access. InstallPID7929/
  qualPID8319/exportPID10985 exit0; no jobs/containers/armed guards. WS/logs/
  ck-plugin-b078-qualification-r1/final.json and matching r1-controller.log;
  user-r1/install-r1, WS/tmp/ck-plugin-b078/export-r1. Local evidence979892B
  SHA22e958b6f3ed827fd1500ded77288cfc71d983b365466808b2a924997bca7bfb.
  Native libraries8389197/source3616b2f/Torcha524980/images/defaults unchanged;
  ordinary free12,650,000,384→12,628,176,896B. [Use/recovery](bc250_ck_plugin_b078_usage.md),
  [scope/evidence](bc250_ck_plugin_b078_results/README.md).
  **Next B07.9 (planned):** inspect existing scalar FP16 DL factory/accumulator/
  type admission and N2.18/B07 evidence, then bounded isolated CPU-reference
  factory proof if feasible. Preserve installed FP32 option/defaults/.105.

**B07.7 opt-in relocatable artifact completed2026-10-29:** toolsfd8c3fe8c;
  accepted package-r2/qualification-r3, no install/native rebuild.23 files6568985B
  include unchanged plugin/probes, eight mapped ABI-library pins/API5/66 symbols,
  focused source/config/license snapshot and genuine two-key examples/profile.
  Explicit check/seed-db/run; no merge into existing DBs. Two extracts (spaces
  included) x daily/RCCL with original candidate/source/build paths absent pass
  50 checks/eight guards;1168 convolution references,488API/984runtime per-ID
  refusals/608 retained/128 absent/20 entry+seed checks. Unchanged tolerance,
  max abs2.35537e-6/256-byte guards;29 local tests/lint/pycompile/diff pass.
  First package's numerics pass but kernel gate rejects Failed to resume KFD.
  Entry exec-after-HIP replaced by same-process runpy; final r3 logs unchanged.
  Underlying mm/queue return untraced: no kernel-root-cause fix claim; preserve
  this N6 input, distinct from historical Python CPU fault/fixed VM heap.
  Qualr2 profile-context setup refusal before RCCL launch corrected inr3;
  initial CPU identity TTY attach error retained, noninteractive capture passes.
  No failed numerical test/relaxed gate/reboot/log clearing.3772 hashes/35 links/
  four AOT DBs/136 backing files unchanged. Same e327c5de boot/r2kernel; no jobs/
  containers/armed guards;10 total ACK. Shared lock/no compilation/ccache reset.
  Package1PID1940/package2PID3233 exit0; qual1PID2297/qual2PID3406 exit1;
  qual3PID3971/exportPID6306 exit0. WS/tmp/ck-plugin-b077/package-r2 and export-r3;
  WS/logs/ck-plugin-b077-qualification-r3/final.json and r3-controller.log.
  Accepted archive5031322B SHA90f0d323d0bb4ffd3a8ae04f1acd4e17805a10d7dcb62d577840cf2120e06984;
  local evidence6819295B SHAeebcf955438a9a5655bc69ca84334d14da83b61c54b629a5a0a55f7576fd73b4.
  Source3616b2f/installed8389197/Torcha524980/defaults/prefix/.105/user apps unchanged;
  ordinary free12,888,330,240→12,653,342,720B, nothing deleted. [Use/evidence](bc250_ck_plugin_b077_results/README.md).
  **Next B07.8:** versioned explicit .104 install/identity/registration/refusal/
  rollback and ordinary daily/RCCL activation under exclusive update lock.
  Preserve native payloads/default whitelist/routing/fallbacks/.105; update usage.

**B07.6 native Search/PerfDb public replay completed2026-10-29:** tools1eac70aa1;
  private libraries3616b2f/installed8389197/Torcha524980 unchanged. Existing native
  Search/GenericSearch and kernel-ID serialization/load validation work; no source
  repair/override. Public Find only137 genuinely generates two PerfDb+two FindDb
  records/profile for batch2/32x34/C=K16/g1f3+g3f1. Native5iterations/1000ms bound,
  private DB/cache. Both select block256 Default3x3; daily1x1Stride1Pad0 vs RCCL
  1x1Pad0. Four fresh opposite-order public137-vs218 processes retain1.085–1.092x
  and1.119–1.135x advantages; no whole-model/general/default claim. Untimed verbose
  replay proves exact-ID DB load/no search; DB hashes unchanged. Invalid-ID copies
  fall back to block8 Default with CPU references, original records preserved.
  148 CPU checkpoints/max abs2.35537e-6/unchanged N2.18 tolerance/256-byte guards;
  3172 explicit Immediate calls excludes internal search evaluations. Private7+7/
  core7/CPU ONNX4/daily5/RCCL6=36 checks/six ACK guards;29 local tests/lint/syntax
  pass. Probe-only fourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/
  ro sources+prefix/no build GPU/network/600s.3701 hashes/35 links/four AOT DBs/
  136 backing files unchanged. Same e327c5de boot/r2kernel/logs; no jobs/containers/
  guards. BuildPID30906/qualPID31175/exportPID32467 exit0. WS/tmp/ck-plugin-b076/
  {inputs-r1,build-r1,export-r1}; WS/logs/ck-plugin-b076-qualification-r1/final.json
  and WS/logs/ck-plugin-b076-r1-controller.log. Evidence1143590B,
  SHA13d5aabfaf5dcf053af67f8d3f7c62514a09b90104798e8c4d9cf4c6dffaa2e6.
  No failed GPU batch/install/default/prefix/.105/user-app changes/deletion;
  ordinary free12,898,476,032→12,890,648,576B. Historical Python CPU fault stays
  distinct from fixed VM heap. [Method/results](bc250_ck_plugin_b076_results/README.md).
  **Next B07.7:** isolated relocatable opt-in artifact with exact native MIOpen/
  ABI/source/license identity and genuine example records; identity refusals/two
  build-free extracts (spaces included) before installation. Preserve defaults/.105.

**B07.5 six private DL configurations completed2026-10-29:** tools45bac0baa,
  private libraries3616b2f/installed8389197/Torcha524980 unchanged. Probe-only build;
  six retained shapes/two layouts, matched cached218 vs explicit API5 per-ID
  invokers. Four opposite-order daily/RCCL timed processes; all six IDs observed.
  Full NCHW includes transforms; prepacked NHWC excludes per-call transforms,
  not pure kernel or an end-to-end packing gain. Fixed first-pass-selected
  block256 Default wins32x34/g1/3x3 by1.091–1.097x; block256 Filter1x1Pad0 wins
  32x34/g3/1x1 by1.138–1.144x. Other four shapes still favor HIP-direct. No public
  default/performance heuristic/install change; unpaired public anchor outlier
  retained, public paired replay remains next. Two diagnostic runs prove native
  IDs/MIOpenConvUni.2184 CPU-double checkpoints/52008 convolution calls,
  max abs2.35537e-6/unchanged tolerance/256-byte guards/immutable inputs pass.
  Private4+4/core7/CPU ONNX4/daily5/RCCL6=30 checks/six ACK guards;29 local tests/
  lint/syntax pass. FourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/
  ro source+prefix/no build GPU/network/600s; accepted B07.3 plugin reused.
  3692 hashes/35 links/four AOT DBs/136 backing files unchanged. Same e327c5de
  boot/r2kernel/logs unchanged; no jobs/containers/armed guards. BuildPID27923/
  qualPID28197/exportPID29498 exit0. WS/logs/ck-plugin-b075-qualification-r1/
  final.json and WS/logs/ck-plugin-b075-r1-controller.log; WS/tmp/ck-plugin-b075/
  {inputs-r1,build-r1,export-r1}. Local evidence1497520B,
  SHA4e2d4d39fd075519ff852c2a740caf6a80bbbf69f5e164b374a99777f6c0e703.
  No install/default/prefix/.105/user-app changes/deletion; ordinary free
  12,911,759,360→12,901,400,576B. Historical Python CPU fault remains distinct
  from fixed VM heap. [Method/results](bc250_ck_plugin_b075_results/README.md).
  **Next B07.6:** audit/use existing solver137 Search/PerfDb in isolated DBs,
  fresh-process public-Immediate selected-config replay with paired HIP-direct
  references/timings. Keep candidate private and defaults/.105/user apps unchanged.

**B07.4 paired public DL/HIP-direct timing completed 2026-10-29:** tools
  e11969041; candidate libraries3616b2f/installed8389197/Torcha524980 unchanged.
  Forced public Immediate137 vs218, six NCHW FP32 shapes/batch2/groups1+3/
  filters1+3/small+moderate inputs. Two fresh opposite-order processes/profile,
  daily+RCCL;32-call windows/three ABBA or BAAB cycles, separate first-use/query/
  prepare/wall intervals. Additional verbose runs prove Default block8 DL plus
  transposes vs MIOpenConvUni; diagnostic timing excluded. HIP-direct wins all24
  timed comparisons,1.13–2.57x; stream intervals include layout/launch/idle costs,
  not pure kernel time. No general speed or whole-model claim; cold-user caches
  are not machine-cold.576 CPU-double correctness checkpoints incl diagnostic
  passes;14,184 calls, only window-final outputs copied. Unchanged N2.18 tolerance,
  max abs2.35537e-6; immutable inputs/256-byte guards pass. Private4+4/core7/CPU
  ONNX4/daily5/RCCL6=30 checks/six ACK600s guards;29 local tests/lints/syntax pass.
  Probe-only isolated build uses fourCPU/OMP4/jobs1/RAM10GiB/mem+swap12GiB/
  ccache20GiB no reset/ro sources+prefix/no GPU/network/600s; existing plugin reused.
  3684 protected hashes/35 links/four AOT DBs/136 backing files unchanged. Same
  e327c5de boot/r2kernel; kernel logs unchanged; no owned jobs/containers/guards.
  Initial staging permission failure corrected on new owned directory before
  build; no failed GPU test/tolerance relaxation. BuildPID25065/qualPID25392/
  exportPID26699 exit0. WS/tmp/ck-plugin-b074/{inputs-r1,build-r1,export-r1};
  WS/logs/ck-plugin-b074-qualification-r1/final.json and matching
  WS/logs/ck-plugin-b074-r1-controller.log. No install/default/prefix/profile/
  .105/user-app changes or cleanup. Ordinary free12,926,078,976→12,914,167,808B.
  Local kernel/artifacts/ck-plugin-b074 evidence1,432,289B,
  SHA630a0525288253423c46e976bc3fe9c2b6009ccead3bb103be1d155c39ebd32b.
  Historical Python CPU fault stays unresolved/distinct from fixed VM heap.
  [Method/results](bc250_ck_plugin_b074_results/README.md).
  **Next B07.5 (planned):** compare all six private per-ID variants on retained
  fixtures; separate invoker-only/full-layout costs and see whether a qualified
  config beats HIP-direct before packaging. Keep candidate private/defaults/.105.

**B07.3 private DL scalar/argument/per-kernel contracts completed 2026-10-29:**
  source libraries3616b2f9f09 over2976e7d; tools7519e73b8. Private facade now
  refuses non-default exact double alpha/beta (including near1/NaN/Inf), bias,
  transpose mode, invalid grouped geometry/output extent and int32-overflow spans.
  get_solution checks the requested ID before allocating an owning solution.
  Captured invoker validates runtime scalars/descriptors/casts/pointers/alignment,
  output/input overlap and required scratch size/alignment/overlap before upstream
  dispatch. Existing upstream math and production admission/factory unchanged.
  Per-ID API solutions release temporary problem/solution before invocation;
  all six IDs run in packed NCHW/NHWC, groups1/3, five1x1/3x3 geometries incl
  stride2/dilation2/C3/K5/K132 tails, default/private streams/two value patterns.
  80 fixtures/profile,256 direct GPU calls each daily/RCCL=512 CPU-double checks;
  dyadic exact, unchanged N2.18 tolerance otherwise, max abs1.07042e-6.244 metadata/
  scalar/transpose/extent/unknown-ID refusals +492 runtime refusals verify output/
  scratch unchanged and inputs/256-byte guards intact. Six IDs each observed;
  shapes are bounded, not general lifetime/concurrency/scalar support proof.
  Retained public solver13764 GPU calls/64 absence refusals/304 prior contracts,
  eight Torch conv output/all-first-gradient fixtures plus native backward pass.
  Private9+9/core7/CPU ONNX4/daily5/RCCL6=40 checks/six final600s guards; initial
  r1 also40 passes, then transpose admission hardened;12 total ACK. No failed GPU
  batch or tolerance relaxation. Source inputs-r1/r2 and r1 validator saved;
  old B07.2 helpers/binaries/inputs unchanged. Two isolated builds use fourCPU/
  OMP4/jobs1/RAM10GiB/mem+swap12GiB/ccache20GiB no reset/read-only source+prefix/
  no GPU/network/600s bound.3676 protected hashes/35 links/four AOT DBs unchanged;
  136 backing paths match. Entry new boot e327c5de-46b8-4e05-94ad-d999a12a06e7,
  kernel still7.1.5-bc250rocm-r2; no agent reboot/cross-boot continuity claim.
  Accepted kernel logs unchanged through cleanup; no owned jobs/containers/guards.
  No install/prefix/profile/default change/.105 or user-app access. Entry ordinary
  free13,329,420,288B; cleanup12,928,606,208B; nothing removed. Buildr1PID11617/
  qualr1PID12209/buildr2PID13602/qualr2PID14039/export exit0. WS/tmp/ck-plugin-b073/
  build-r2; WS/logs/ck-plugin-b073-qualification-r2/final.json and matching
  WS/logs/ck-plugin-b073-r2-controller.log; export-r1. Local kernel/artifacts/
  ck-plugin-b073 evidence3284485B SHAe0038bf797e4846a91baa4efb81fc8b9c912badb2ba0152151bedc4565725b2b,
  211 rawfiles16842346B; matching sources/binaries/66 exports/hidden-helper proof.
  29 local tests/critical Ruff/Black/new-source clang-format/pycompile/shell/diff
  pass; full upstream CI not run. No broader shape/type/performance/scaling or
  non-default scalar arithmetic support claim. Historical Python CPU fault stays
  unresolved/distinct from fixed VM heap. [Results/contracts](docs/development/bc250_ck_plugin_b073_results/README.md).
  **Next B07.4 (planned):** bounded paired private DL vs existing MIOpen HIP-direct
  timing with retained CPU references, explicit solver attribution and warm/cold
  separation; classify wins/losses before deciding opt-in packaging/admission.
  Preserve installed defaults, private candidate, .105 and user applications.

**B07.2 isolated MIOpen FP32 DL forward plugin completed 2026-10-29:** source
  libraries2976e7d30bfdccb2fb79809b8c58a6b5722a0822 over installed8389197;
  tools528088c26/controller608ac7b86. Private API5 facade exports all66 required
  symbols; unsupported11 groups return false/empty lists and explicit workspace/
  solution INVALID_VALUE, nulls BAD_PARAM. Packed2D FP32 forward only, casts/TF32/
  mixed types/determinism refused. Existing upstream forward/invoker math reused.
  Copied config enables CK templates only in candidate; explicit DL macro/C++20/
  beta API/existing includes/archive. Host-hidden GetCastType gets its exact
  matching one-line definition hidden locally, verified not exported. Nine raw
  factory instances have duplicate IDs (transfer parameters omitted from names);
  private-only guarded filter retains first per ID, exposes six unambiguous IDs.
  No production factory/admission/default changes. Forced public Immediate solver
  137 (historical Xdlops name, actual DL kernels), packed NCHW+NHWC/groups1+3/
  filters1+3/batch2/5x7/C=K=4 per group/default+private streams/two input patterns:
  32 GPU calls each daily/RCCL=64; CPU double, dyadic exact/unchanged N2.18 tolerance,
  max abs6.07623e-7,256-byte input/output/workspace guards/immutable inputs pass.
  Fresh no-private-path subprocess32 refusals/profile=64;304 recorded contract
  checks include all unsupported groups/cast/FP16/TF32/nulls. Eight Torch conv
  output/all-first-gradient fixtures plus native arithmetic backward pass; ordinary
  Torch may choose retained solvers, not proof of automatic private dispatch.
  Private8+8/core7/CPU ONNX4/daily5/RCCL6=38 checks/six accepted600s guards;12 total
  ACK including earlier failures. Buildr1 C++17/includes, r2 beta/sqlite, r3+r4
  hidden accessor (r4 unchanged retry), r5+r6 built; r7 adds private ID filtering,
  r8 probe workspace-guard correction. Qualr1 duplicate IDs, r2+r3 test wrongly
  checked mutable workspace scratch (r3 retains native logs), r4 private gates
  pass then wrong launcher lacks transformers; r5 full pass after bc250-train fix.
  No numerical tolerance/reference relaxation. FourCPU/OMP4/jobs1/RAM10GiB/
  mem+swap12GiB/ccache20GiB no reset/read-only source+prefix/no GPU/network build,
  timeout600s.3663 protected hashes/35 links/four AOT DBs unchanged;136 retained
  backing paths match. Same r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63/kernel logs
  unchanged. No install/prefix update/default admission change/.105/user app access.
  Buildr8PID29981/qualr5PID31501/export exit0; no owned jobs/containers/armed guards.
  WS/tmp/ck-plugin-b072/build-r8; WS/logs/ck-plugin-b072-qualification-r5/final.json
  and WS/logs/ck-plugin-b072-r5-controller.log; export-r1. Local kernel/artifacts/
  ck-plugin-b072 evidence7298658B SHA9e2aaabae96bd8386cfb2ba65e3de62408a1bd7fd5cd121461baf64627e91ecc,
  283 rawfiles34976217B, matching source/binary/export checks and private helper
  visibility proof.29 local tests/critical Ruff/Black/new-source clang-format/
  pycompile/shell/diff pass; no full upstream CI. Free30,153,154,560B. Default
  Immediate alpha1/beta0 only qualified; non-default scalars/per-kernel selection/
  stride+dilation+tails/performance/other types/plugin backward remain open.
  Historical Python CPU fault remains unresolved/distinct from fixed VM heap.
  [Use/scope/evidence](docs/development/bc250_ck_plugin_b072_results/README.md).
  **Next B07.3 (planned):** private forward scalar/argument admission and per-kernel
  stride/dilation/tail qualification; resolve non-default alpha/beta before broader
  use, then select paired HIP-direct timing/opt-in packaging. Keep defaults/.105.

**B07.1 CK grouped-convolution plugin admission completed 2026-10-29:** tools
  a4bf75b8033b484b364be84a6233efe681203875. Nine local/staged source files match;
  API5 loader requires66 entry points across12 solver groups, using internal C++
  types despite extern-C entry names. gfx1013 excluded by plugin CMake/default
  loader whitelist; explicit MIOPEN_CK_LIB_PATH bypass remains. Existing scalar DL
  FP32/FP16 2D NHWGC/GKYXC/NHWGK factory matches plugin forward interface. Factory
  uses DL_KERNELS, distinct from configured CK_ENABLE_DL_KERNELS; MIOpen compile
  helper does not explicitly propagate the former. Actual plugin flags still
  require checking. Native no-DL probe enumerates0; -DDL_KERNELS probe links
  installed archive and enumerates9 FP32 instances.16 new interleaved grouped
  fixtures/profile,1x1/3x3/groups1+3/default+private streams/dyadic+non-dyadic,
  execute96 admitted instances each daily/RCCL=192 CPU-double comparisons.
  Dyadic exact; unchanged N2.18 tolerance otherwise, max abs6.07623e-7;256-byte
  guards/immutable inputs and native Torch HIP backward pass. Factory gates3+3,
  retained core7/daily5/RCCL6=24 checks/five accepted guards; earlier r1 parser
  rejects diagnostic-prefixed JSON after native exit0, preserved; six total ACK.
  Test parser now retains stdout and parses final JSON, no numerical relaxation.
  Two isolated native link builds, ccache20GiB/no reset/OMP4/fourCPU/RAM10GiB/
  mem+swap12GiB/read-only sources+prefix/no GPU/network/build timeout600s. No
  plugin/library/Torch build, install, admission/default/prefix change.3649
  protected hashes/35 links/four AOT DBs unchanged;136 backing paths verified.
  Same r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63/kernel logs unchanged; .105/user
  apps untouched. Qualification r1PID10147 exits1(parser), r2PID10922 exits0;
  evidence export exits0. WS/logs/ck-plugin-b071-qualification-r2/final.json,
  WS/logs/ck-plugin-b071-r2-controller.log; WS/tmp/ck-plugin-b071/build-r{1,2},
  export-r1. Local kernel/artifacts/ck-plugin-b071 evidence1560256B
  SHAbdd63ea9687bfe26f96ea253845fabfe9f906c91da2be4427ed360e7653c3319;
  109 rawfiles7008402B.29 local tests/critical Ruff/Black/clang-format/pycompile/
  shell/diff pass; upstream full CI not run. Free30,202,425,344B; no owned jobs/
  containers/armed guards. No MIOpen routing/backward-plugin/type/performance
  qualification; historical Python CPU fault remains distinct from fixed VM heap.
  [Audit/decision/evidence](docs/development/bc250_ck_plugin_b071_results/README.md).
  **Next B07.2 (planned):** isolated matching-MIOpen forward-only FP32 DL plugin,
  explicit DL macro, all66 required exports with unsupported operations refusing;
  private-path discovery and forced real solver CPU-reference/guard checks before
  any production admission. Keep installed defaults/.105/user apps untouched.

**B06.12 modern Torch export→public ORT GPU completed 2026-10-29:** tools
  75da6725bd2566828ace1074c5d24c7e0e6347ad, evidence e9e6c29a8. Fresh owned
  dense/CNN/gated-LayerNorm models, CPU+GPU export origins, static batch3,
  dynamo=True/opset18; all six exporter-produced graphs are IR10/opset18 without
  IR rewriting. Two changed inputs each match Torch CPU FP64, eager HIP FP32,
  retained CPU ORT1.31 and ONNX NumPy references at unchanged rtol3e-4/atol2e-5.
  Public bc250-ort daily/RCCL each30, generation30 and retained core7/CPU ONNX4/
  daily5/RCCL6=112 checks; seven ACK600s guards.24 consumer GPU graph evaluations,
  12 actual profiles assign every recorded node to MIGraphX with CPU fallback
  disabled;12 generation eager GPU forwards. Invalid inputs, native gfx1013,
  package/library origins, explicit CPU after GPU and exact Torch HIP backward
  pass.3642 protected hashes/35 links/four AOT DBs unchanged;136 retained workstation
  backing paths verified. No numerical/library/default/prefix changes, native
  rebuild or cache reset. Same r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63; kernel
  logs unchanged. Qualification PID5491 exited0, export exited0; no owned jobs/
  containers/armed guards. WS/logs/onnxruntime-b0612-qualification-r1/final.json,
  matching -controller.log; fixtures WS/tmp/onnxruntime-b0612/generation-r1/models;
  export WS/tmp/onnxruntime-b0612/export-r1. Local kernel/artifacts/onnxruntime-b0612:
  evidence813709B SHAf5a882a28504f7e0058fd9ae4982f7bf13ad307a58da5ad7e36d92b208f967b1,
  179 rawfiles3670773B.29 local tests/critical Ruff/Black/pycompile/diff pass;
  full upstream CI not run. Free30,209,069,056B. .105 not contacted; user apps
  untouched. Inference-only bounded coverage; dynamic shapes/types/operators/
  concurrency/graph-autograd/performance remain unqualified. Historical Python
  CPU fault remains unresolved and distinct from fixed VM-heap corruption.
  Practical B06 review finds no mandatory defect in this tested path; park optional
  expansion until a concrete supported-model requirement. [Results/gap review](docs/development/bc250_onnxruntime_b0612_results/README.md).
  **Next B07.1 (planned):** audit MIOpen CK grouped-conv plugin ABI/instance factories
  and native scalar/DL applicability versus retained N2.18/HIP-direct alternatives;
  choose a bounded useful implementation before building or changing admission.
  Preserve safe defaults, installed packages, .105 and user applications.

**B06.11 installed explicit ORT package completed 2026-10-29:** source remains
  ORTb81a7866e5a6eff964b9e411535b4c9198651a9f, registered SDK4797 unchanged. Tools
  packagebc0f0b0b9/context7870a6ee8/exportfddb14f61. Versioned archive
  bc250-ort-migraphx-1.31.0-gfx1013-cp312-b0611 at /opt/bc250-stack/sdks/;
  registration ort-migraphx.json plus retained version record; /usr/local/bin/
  bc250-ort delegates through existing SDK/Torch launchers with full shared-lock
  identity checking/read-only package. Install/register/unregister/context tool
  update use EX lock. New Python/distribution identity1.31.0+gfx1013.b0611 fixes
  missing build_and_package_info warning without suppressing it. Five copied ELF
  RUNPATHs normalized to $ORIGIN; DT_NEEDED and .text bytes match originals.
 504 files/two contained links include core/Python/providers/headers/licenses,
  original source+repair/hash pins/13 dependency archives/build settings. External
  qualified glibc/ROCm image/registered SDK required; not standalone/cross-device.
  Two private extracts (spaces included), original build and /sdk absent, each34
  pass. Installed daily/RCCL each34 plus core7/CPU ONNX4/daily5/RCCL6 bring158
  guarded/eight accepted guards, host14/user2=174 checks/48 GPU evaluations/24
  accepted actual model profiles. Earlier ordinary acceptance90/24 evals retained;
  all14 guard records ACK. Same FP32 six IR10/opset18 fixtures/CPU-double+CPU ORT
  references/tolerances/no CPU node fallback, wrong-input refusals, native gfx1013,
  exact Torch HIP backward/loaded origins/package identity/ro mount. Actual normal
  UID1000 exposed bare --env loss after elevation; fixed by caller-side options+
  existing private-context validation. Before/after proof covers bare env and
  inherited HF_*; old wrapper/records at WS/recovery/ort-b0611-before-context.
  Final acceptance-r2 passes; archive/numerical code unchanged by wrapper fix.
  Live unregister/refusal/re-register, identity drift, lock exclusion and plain
  daily/RCCL/SDK CPU ORT preservation pass.3553 native hashes/35 links/four AOT DBs/
  136 retained backing paths match, full package+SDK checked separately. Same
  r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63; kernel logs unchanged through cleanup.
  No owned jobs/containers/armed guards; no compilation/cache reset/source/build/
  artifact deletion. Initial ordinary30.1GiB; cleanup30,217,469,952B. Active core
  image/ROCm/Torch/SDK/defaults/boot/fallback/user apps unchanged; .105 not contacted.
  Package PID14494/relocation15155/install16120/final acceptance19704/export21375
  exited0. WS/tmp/onnxruntime-b0611/{package-r1,export-r1}; WS/logs/onnxruntime-b0611-
  acceptance-r2/final.json and matching -controller.log. Local kernel/artifacts/
  onnxruntime-b0611 package350,607,989B SHAcdac3eab84dd0803ae85eac9e46d657ea67bd267db0bbbaf2ac53c2ec21feae3;
  evidence1,872,893B SHA8bffd0ce5f114ff5e08ee51c32f8a3b9bd5919f59644b5448045b4b65fa22269,
  256 rawfiles8,720,980B plus package/ELF/verification/preservation/cleanup.29 local
  tests/selected critical Ruff/Black/pycompile/diff pass, full upstream CI not run.
  Explicit ONNX inference/provider choice only; no automatic Torch routing,
  graph autograd, arbitrary export/model/dtype/concurrency or performance claim.
  Historical Python CPU fault stays unresolved/distinct from fixed VM-heap damage.
  [Use/session/recovery](bc250_onnxruntime_b0611_results/README.md).
  **Next B06.12 (planned):** bounded modern Torch exporter→public bc250-ort GPU
  integration with independent CPU/eager-GPU references, then practical B06 gap
  review before further expansion. Preserve installed defaults and .105.

**B06.10 isolated ORT/MIGraphX candidate completed 2026-10-29:** matching ORT1.31.0
  core/Python3.12/provider built and actually executed on .104. Sourceb81a7866e5a6eff964b9e411535b4c9198651a9f
  over402979ac: explicit existing uint32 cache conversion, explicit this capture,
  redundant final-class virtual destructor removed (base destructor protected/
  nonvirtual). No calibration/quantization/lifetime/math qualification from these
  compiler fixes. Tools9791d4134;13 archives upstream-SHA1/SHA256 verified/offline
  mirror. Nested build venv explicitly uses retained NumPy2.5.1; no image package
  replacement. Private repaired SDK header only; registered SDK4797 remains intact.
  Clang24 Abseil diagnostic requires SYSTEM option plus CMake hook since ORT copies
  include paths without SYSTEM. -Werror retained; failing target smoke-r4 passes.
  Original source-r1/failed configs/builds/caches preserved; source-r2 original+three
  repair files all11379 hashes match. No native ROCm/Torch/GPU backend rebuild.
  FourCPU/jobs4/link1/RAM10GiB/mem+swap12GiB/private2GiB ccache/no reset/read-only
  source/root/no build GPU/network.24GiB admission, monitored20GiB loss/4GiB floor/
  two-hour phase bound. Entry~32.6GiB/export32,344,141,824B; no cleanup/reserve change.
  Three fresh import orders each32 (daily ORT-first/Torch-first, RCCL MIGraphX-first)
  plus core7/installed CPU ONNX4/daily5/RCCL6=118/seven ACK guards/36 GPU graph evals.
  Six retained FP32 IR10/opset18 graphs/two changed inputs, CPU-double/CPU ORT refs,
  unchanged tolerances/host inputs, dtype/shape/missing-input refusals, candidate
  CPU after GPU and exact Torch HIP backward pass.18 actual model profiles assign
  all nodes to MIGraphX with session CPU fallback disabled; actual candidate+SDK
  library paths/nativegfx1013 checked. Candidate ORT ELF has no exported protobuf/
  absl definitions. No broader loader/model/dtype/dynamic/concurrency/autograd/
  performance claim.2028 native hashes/29 links/four AOT DBs/136 retained backing
  paths match; installed SDK/source checked separately. Same r2boot83f6cb1b-4540-
  4149-839e-a6cf7cb80b63, kernel logs unchanged. No jobs/containers/armed guards.
  Accepted build-r5 PID8939, qualification-r1 PID9659, export-r1 PID11602 exited0;
  WS/tmp/onnxruntime-b0610/task-r2/build; WS/logs/onnxruntime-b0610-qualification-r1/
  final.json and -controller.log; WS/tmp/onnxruntime-b0610/export-r1. Failed configure-r1
  NumPy, build-r2/r3 Abseil, build-r4 provider diagnostics retained; no GPU failure.
  Local kernel/artifacts/onnxruntime-b0610: evidence3,223,004B
  SHAb7ff9d388611f0db8be2f66133dd42e9f4f702276e43064afe191ce1554a7d25;
  224 rawfiles17,064,377B; raw candidate29,115,179B/325 verified files (not relocation
  qualified), source repair/bundle/shallow reconstruction/fsck and ELF/cleanup.
  Initialized pinned lintrunner0.13.1/adapters0.14.1/Ruff0.12.12/clang-format20.1.8
  source lint passes;29 local tests/selected critical Ruff/Black/pycompile/shell/
  diff pass. Full upstream CI not run. Unpackaged candidate warns missing
  build_and_package_info; resolve identity in packaging, do not suppress in user
  scripts. Installed CPU ORT/image/SDK/ROCm/Torch/defaults/fallbacks/user apps
  unchanged; .105 not contacted. Historical Python CPU fault remains unresolved/
  distinct from fixed VM-heap corruption. [Evidence/scope](bc250_onnxruntime_b0610_results/README.md).
  **Next B06.11 (planned):** relocatable matching candidate package with sources/
  licenses/identity/loader metadata; fix package metadata, qualify extraction and
  explicit versioned activation/rollback plus retained daily/RCCL gates. Preserve
  installed CPU ORT/SDK/defaults/.105; no automatic provider/default switch.

**B06.9 ORT/MIGraphX admission completed 2026-10-28:** pinned unmodified ORT1.31.0
  402979ac1bef89dcd55add41eaa7277c407599eb at sources/bc250-onnxruntime. Shallow
  checkout/fsck and exact snapshot11,379 files verified locally and on .104;
  three gitlinks recorded/uninitialized, no signed-tag/full-history claim.
  Source279,492,314B SHAe4243f26f7b6fd6be09fabd08a77161da719cdded1cbbc696458e3e2aeb3e51b.
  Twelve static assertions establish CMake3.28/C++20/shared migraphx::c+hip::host,
  parser/native stream/output-copy/FP32-default/fallback/visibility controls.
  Real Clang C++20 blocker reproduced in installed SDK: handwritten C++ wrapper's
  two leading unqualified module return types misparse as module directives.
  Source144f6db5c qualifies both as migraphx::module; final134ee25f67007523bd03374c4b723fe075a3b77f
  also gates new regression on compiler C++20 support. ABI/math unchanged;
  registered SDK remains4797a6112 bytewise, candidate uses private header overlay.
  C++17+20 CPU header regressions pass;168 discovered C API declarations link.
  Eight guarded ORT-style device-pointer run_async cases each daily/RCCL: two
  models/two changed inputs/default+private streams, explicit output allocation/
  D2D copy, CPU-double/CPU ORT/host+device guards. Retained core7/ONNX4/daily5/RCCL6
  bring38/six accepted guards/16 GPU graph evaluations. Earlier fixture passes
  same16 plus core/ONNX, then log-label collision before remaining gates;
  four failed-fixture guards ACK. No math/tolerance change. Builds r1/r2 retain
  missing clang-scan-deps/sibling ccache helper/read-only default-cache failures;
  r3 actual header failure, r4 succeeds. Disable module scan/use complete setup
  helper mount/fail on setup errors. FourCPU/jobs4/RAM10GiB/mem+swap12GiB/private
  2GiB ccache/no reset/read-only build root/no GPU/network. Tools c86648601,
  source-stage/export167e467e5. ORT protobuf33.6/Abseil20250814 differ from SDK30/
  20250512; ONNX namespace isolated,195 protobuf/absl exports inventoried, provider
  version script hides except GetProvider. No collision reproduced/ABI-wide proof;
  actual candidate ELF/import-order tests required. Candidate13 dependencies pinned
  URLs/upstream SHA1, bytes not fetched/configure not run. Provisional24GiB ordinary
  admission/20GiB growth/4GiB floor,link1/two-hour phase bound, not measured footprint;
  source stage admits40,796,561,408B. Full ORT core/binding/provider not built/loaded/
  installed. Active CPU ORT/SDK/Torch/ROCm/image/defaults/fallback/user apps unchanged;
  .105 not contacted.1695 native hashes/29 links/four AOT DBs match,136 preserved
  backing paths match; full installed SDK checked separately. Kernel logs match
  through cleanup on r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63. No owned jobs/
  containers/armed guards. Build PID10099, qualification11488, source-stage12693,
  export13014 exited0; WS/tmp/onnxruntime-b069/{probe-r4,inputs,export-r1};
  WS/logs/onnxruntime-b069-qualification-r2/final.json and matching -controller.log.
  Local kernel/artifacts/onnxruntime-b069 evidence1,440,485B
  SHAeb10bae63e6ab08dd09c54a0866ae74fe362bc72ea93076d72b942572f30a4d9;
  196 rawfiles5,500,940B plus source/dependency/plan/ELF/repair/reconstruction receipts.
  Earlier scratch reconstruction fails pack/index/reflog/tree checks (cause unknown);
  retained. Independent complete bundle-chain clone with auto maintenance disabled
  passes full fsck at134ee25.29 local tests/Black/pycompile/C++format/diff pass;
  Ruff absent/full upstream CI not run. No provider execution/partition/allocator/
  dynamic-shape/concurrency/dtype/model/autograd/performance claim. Historical
  Python CPU fault stays unresolved/distinct from fixed VM-heap corruption.
  [Evidence/plan](bc250_onnxruntime_b069_results/README.md).
  **Next B06.10 (planned):** fetch/hash selected dependency bytes with SHA256,
  create offline mirror, recheck space/bounds; configure/build matching isolated
  ORT core/Python/MIGraphX provider using repaired private header. Preserve installed
  CPU ORT/SDK/daily+RCCL/defaults/.105. Actual provider assignment/no-silent-CPU-
  fallback and independent references must pass before installation/activation.

**B06.8 installed opt-in SDK completed 2026-10-28:** source remains4797a61125bbec318005969f8497dbd50849cd6e;
  implementation de93d4fe992383ef4787d494706b52d5c7f707b6, export5f5a73b7a.
  Qualified B06.7 archive installed root-owned at /opt/bc250-stack/sdks/
  bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067; registration sdks/migraphx.json,
  retained version record and tools/migraphx-b068. New /usr/local/bin/bc250-migraphx
  verifies pinned SDK index/manifest/all bytes/exact file set/contained links and
  selected runtime manifest/image under shared daily lock, then delegates to
  ordinary bc250-torch with read-only SDK. Install/unregister/register use exclusive
  lock; live unregister/refusal/re-register exercised, payloads retained. Scratch
  runtime/SDK drift refusals, lock exclusion, default SDK absence and actual UID1000
  doas activation pass. Ordinary daily+RCCL activation8/Python50/CAPI+CLI6 each,
  retained core7/ONNX4/daily5/RCCL6=150 guarded; host14+user1=165 checks,
  ten ACK guards/68 GPU graph evaluations. Same FP32 IR10/opset18 six fixtures,
  CPU-double/CPU ORT references/tolerances, actual sibling HIPRTC helper observed;
  external script/sidecar/spaces/read-only mount and Torch HIP backward pass.
  All1681 native protected hashes/29 links/four AOT DBs match;136 preserved backing
  hashes verified locally before/after, full installed SDK separately checked.
  New pre-task r2boot83f6cb1b-4540-4149-839e-a6cf7cb80b63; batch kernel logs match,
  no agent reboot/cross-boot continuity claim. No native rebuild/cache reset needed.
  Active ROCm/Torch/image/defaults/fallbacks/user apps unchanged; .105 not contacted.
  Final ordinary free41,137,750,016B. No owned jobs/containers/armed guards/transport.
  Controller PID8265 and export PID10260 exited0; WS/logs/migraphx-b068-qualification-r1/
  final.json; controller log WS/logs/migraphx-b068-qualification-r1-controller.log;
  export WS/tmp/migraphx-b068/export-r1. Local kernel/artifacts/migraphx-b068:
  evidence623,640B SHAc0e0ce195e4507d57694422dcc0ffd878eaeb686f6aa5627e45cd683cf991396,
  188 rawfiles2,698,354B plus verification/registration/preservation receipts.
  29 local tests/Black/pycompile/diff checks pass; Ruff absent/full upstream CI not run.
  No arithmetic failure/tolerance relaxation. Explicit inference only: ordinary Torch
  launchers do not load SDK; no ORT GPU EP, graph autograd/all-model/dtype/performance
  or universal MXR claim. Historical Python CPU fault remains unresolved/distinct
  from fixed VM-heap corruption. [Use/recovery/evidence](bc250_migraphx_b068_results/README.md).
  **Next B06.9 (planned):** pinned ONNX Runtime MIGraphX EP/API/dependency/storage
  admission audit; establish feasible isolated candidate before build/provider
  qualification. Preserve installed CPU ORT/SDK/daily+RCCL/defaults/.105. Broader
  model/dtype/performance admissions remain separate; continue one bounded batch
  at a time in order, resolving necessary discovered prerequisites autonomously.

**B06.7 relocatable SDK completed 2026-10-25:** MIGraphX source
  4797a61125bbec318005969f8497dbd50849cd6e over622ea3f2a fixes generated CMake
  variables advertising nonexistent migraphx::migraphx_c instead of exported
  migraphx::c. External consumer reproduces old failure; fixed config/build/link
  and GPU C API/C++ wrapper pass. No backend math/ABI/whitelist change. Tools
  389966df4824b6e07e3accb67a679c5b2f2a5e7c. Source4782 files verified; full
  bundle-chain reconstruction/fsck reaches4797. Private SDK adds C API/CLI to
  existing Python3.12/GPU/parser; dependencies/policies remain B06.6.
  Archive kernel/artifacts/migraphx-b067/bc250-migraphx-2.18.0.dev-gfx1013-cp312-b067.tar.gz
  is41,753,676B SHA11f7f7ab1923132131dc0118af62fbd425e55cfd802d86fdcb57e4952583c8ba;
  565 files/12 contained links, headers/CMake/source/license/fetch pins/owned fixtures.
  Verified sibling HIPRTC helper copy handles installed deep-library layout;
  wrapper-derived license link materialized from source. No active SDK install.
  Two native extracts (one path with spaces) pass Python50+CAPI/CLI6 each, original
  build trees absent, actual SDK HIPRTC children and native renamed SQLite loaded.
  Core7/ONNX4/daily5/RCCL6 bring134/eight accepted guards,68 GPU graph evaluations.
  All1675 native hashes+136 preserved workstation-backing hashes match (1811
  combined; all original665 preserved);29 links/four AOT DBs unchanged. Kernel
  logs match through export/cleanup on r2 boot e955931f-5edb-42e5-8f14-eafc2fbaf361.
  All11 guards ACK (eight accepted/three failed fixtures). No arithmetic failure
  reproduced/tolerance relaxation. Earlier attempts retain transport/empty-wrapper/
  license-link/SSHFS-readlink failures and stale ORT-path/native-scratch-ownership/
  SQLite-SONAME test corrections. No GPU/VM-heap defect inferred from those.
  Builds remain external durable/b067 (r5 SDK, consumer-r2); fourCPU/jobs4/link1/
  RAM10GiB/mem+swap12GiB/private2GiB ccache/no reset/no build GPU/network, prior
  external16GiB/14GiB/root768MiB/two-hour bounds. Bridge interruptions stopped
  pre-scans; with root free externally observed~55GiB, immutable native stage at
  WS/tmp/migraphx-b067-native uses4GiB extraction admission/2GiB growth/floor.
  Qualification-r5/export-r5 use detached jobs and need no bridge/build mount.
  Native stage before58,866,450,432B/after58,476,675,072B; export58,079,264,768B.
  Free-space increase origin not established; no reserve-policy/retained-artifact
  cleanup by these tools. Native extracts/backing/sources/builds/caches retained;
  new source-transfer original name aliases its verified native SDK source copy.
  Evidence1,324,388B SHA27edca3ad576cb11241a27a357cd1ec56c802cd58c79799b1e80b6f57ef884ed,
  253 rawfiles5,574,136B; separate build archive232,761B/42files1,168,331B; SDK/source
  bundles/verification/cleanup in kernel/artifacts/migraphx-b067. 29 local tests,
  Black/pycompile/shell/C++ formatting pass; Ruff absent/full upstream CI not run.
  No owned jobs/containers/armed guards/mount/relay. Active ROCm/Torch/images/
  defaults/boot/fallbacks/user apps unchanged; .105 not contacted. CPython3.12/
  glibc FP32 static graph scope only; no ORT GPU EP/MIGraphX autograd/all-model/
  dtype/opset/performance or universal MXR portability claim. Historical Python
  CPU fault remains unresolved/distinct from fixed VM-heap corruption.
  [SDK/use/evidence](bc250_migraphx_b067_results/README.md).
  **Next B06.8 (planned):** versioned .104 opt-in SDK registration/identity/rollback,
  then ordinary launcher activation/retained daily+RCCL qualification. Check fresh
  space; preserve current defaults/fallbacks/.105. Broader models/dtypes/ORT GPU
  provider/performance remain separate admissions.

**B06.6 Python MIGraphX bindings completed 2026-10-25:** source remains
  622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83; pinned private pybind11 2.12.0
  3e9dfa2866941655c56877882565e7577de6fc7b. Python 3.12.3 extension/support
  libraries build; no MIGraphX source repair or installed package replacement.
  Tools checkpoint 8692aa5fe, accepted fixture/retries 0bfe2858c. Three fresh
  import orders each pass 49 composites: six FP32 IR10/opset18 models, file/buffer
  parsing/execution, uncompiled JSON/msgpack/buffer IR equality, GPU lowering,
  changed-input CPU-double/CPU ORT references, compiled file/buffer reload,
  explicit HIP copies, four refusals, CPU ORT inference/Torch HIP backward after.
  90 GPU graph evaluations; retained core7/ONNX4/daily5/RCCL6 bring 169/seven gates.
  665 protected hashes/four AOT DBs/links match; seven accepted guards ACK and
  kernel logs unchanged through export/unmount on r2 boot e955931f-5edb-42e5-8f14-eafc2fbaf361.
  4,782 MIGraphX/262 pybind source files verified; complete pybind bundle fresh
  clone/fsck passes. First build refused stale unused protobuf source mount before
  compilation; r2 corrected. First GPU fixture omitted compiler-exposed output
  allocation (main:#output_0); corrected per upstream API, same math/references/
  tolerances. Failed guard ACK/no kernel drift; all attempts preserved.
  New external durable/b066 under existing B06.4 temporary storage; native read-only
  pybind input WS/tmp/migraphx-b066/pybind11. Build~463MiB/private headers908KiB/
  cache51MiB; old builds/caches retained via reflinks. FourCPU/jobs4/link1/RAM10GiB/
  mem+swap12GiB/no build GPU/network/read-only root/private2GiB cache/no reset;
  external16GiB admission/14GiB growth/root768MiB floor/two-hour bound unchanged.
  Export root ordinary866,148,352B; external114,335,145,984B. 29 local tests,
  Black/pycompile/shell syntax pass; Ruff absent/full upstream CI not run.
  Local kernel/artifacts/migraphx-b066 archive21,375,875B SHA
  79738ca55dec9686b95d9303a01a4de084ac6398be6e34edeaf64dd430129c41;
  239 extracted raw files9,029,640B, separate failed guard and complete source bundle.
  No owned jobs/containers/armed guards/mount/relay; backing files remain.
  Active ROCm/Torch/images/defaults/boot/fallback/BIOS/user apps unchanged; .105
  untouched/not contacted. Candidate CPython3.12/glibc FP32 inference only; no
  active SDK/ORT GPU EP/MIGraphX autograd/all-model/IR/opset/dtype/performance claim.
  Historical Python CPU fault remains unresolved/distinct from fixed VM-heap damage.
  [Scope/replay/evidence](bc250_migraphx_b066_results/README.md).
  **Next B06.7:** versioned relocatable opt-in SDK archive with source/license/
  loader metadata; qualify relocation against existing runtime image before
  considering installation. Respect ordinary root headroom and preserve .105;
  ORT GPU provider and broader dtype/model/fusion/performance are separate.

**B06.5 native ONNX parser completed2026-10-25:** MIGraphX622ea3f2a unchanged;
  tools checkpointb6a629170/final4711e9892. Pinned full clean sources Abseil
 20250512.0 bc257a88f7c1939f24e0379f14a3589e926c950c and Protobufv30.0
  d295af5c3002c08e1bfd9d7f9e175d0a4d015f1e (tag2c82161c); private C++17/PIC/O2
  static builds/install and libprotoc30.0 pass. Existing profiler Abseil202601/
  Torch protobuf/Python protobuf preserved. Full Git bundles fresh-clone/fsck pass.
  Native ONNX24: six IR10/opset18 FP32 models (broadcast/ReLU, reduce, dense/ReLU,
  transpose/MatMul, conv/bias/ReLU, serialized weight/bias initializers), file/
  buffer IR equality, twelve GPU evaluations vs independent CPU-double and CPU
  ORT references/shape/type/host guards; four unknown/malformed file/buffer refusals.
  Actual GPU code_object/gemm/convolution/literal ops/HIP copies, fast_math false/
  tune off; no tolerance relaxation. CPU oracle generation12 passes separately.
  Retained candidate hand-built22/core7/ONNXaudit4/daily5/RCCL6 bring68/six gates.
 629 protected hashes/four AOT DBs/links and per-gate/export/unmount kernel logs
  match; six guards ACK. New pre-task r2boot e955931f-5edb-42e5-8f14-eafc2fbaf361,
  no agent reboot/cross-boot log identity claim.4,782 MIGraphX/1,532 Abseil/3,052
  Protobuf sourcefiles verified. MIOpen fusion whitelist still excludesgfx1013;
  ordinary conv/separate pointwise qualified, no architecture/dtype admission change.
  Reused B06.4 temporary SSHFS backing; new data local tmp/migraphx-b064/
 20261025-storage/durable/b065 (remote STORAGE=WS/tmp/migraphx-b064/
 20261025-storage/mount/b065). Native source snapshots WS/tmp/migraphx-b065/
  {abseil,protobuf}; parser uses old readonly622 input. FourCPU/jobs4/link1/
  RAM10GiB/mem+swap12GiB/read-only build root/noGPU/network; private2GiB caches/
  no reset/old shared cache untouched. Parser~451MiB/private deps49MiB/dependency
  builds6.6/69MiB/caches19/50MiB. External16GiB admission/14GiB growth/root768MiB
  floor/two-hour phase bound. Export root ordinary877,867,008B/external114,507,022,336B.
  Initial source staging EOF before build corrected with completed-archive SHA
  check/re-extraction. Live-script edit truncated final protoc version command
  after successful compile/install; accepted incremental retry snapshots immutable
  phase script, no rebuild/source change. TTY CPU-generator stop resumed cleanly;
  null stdin thereafter. Original attempts retained, no GPU numerical failure.
 29 local tests/Black/pycompile/shell/C++ format+syntax pass; Ruff absent/full
  upstream CI not run. Disabled-MLIR/unused rocm-path host warnings remain.
  No active ROCm/Torch/image/default/boot/fallback/BIOS/user-app/.105 change/contact;
  no artifact deletion/reserve policy change. Parser C++ isolated only, no Python
  bindings/ORT GPU EP/autograd/model speed/all-IR/opset/dtype claim. Unknown Python
  CPU fault remains distinct/unresolved from already-fixed VM-heap corruption.
  Raw accepted STORAGE/logs-{abseil-r1,protobuf-r2,parser-r1}; WS/logs/
  migraphx-b065-qualification-r1. Local kernel/artifacts/migraphx-b065 archive
 67,909,794B SHAa44098301bfa85484798d9695a6656b360161389ce157c72f8b2260ac10ce6c8,
 118 rawfiles2,346,845B plus models/source bundles/verification/cleanup. No owned
  jobs/containers/armed guards; temporary mount/relay closed, backing data retained.
  [Scope/replay/results](bc250_migraphx_b065_results/README.md).
  **Next B06.6:** reconnect durable storage; audit/build pinned Python bindings in
  isolated candidate, then Python ONNX inference/serialized-program roundtrips/
  Torch+CPU ORT import-order gates before considering installed SDK packaging.
  Preserve active runtime/.105; root headroom remains low. ORT GPU EP, broader
  dtype/IR/opsets/fusion/model performance remain separate admissions.

**B06.4 isolated MIGraphX FP32 graphs completed2026-10-25:** source
 622ea3f2ad8f1baa8a44edb4f1e07cbe6e532c83 over35b40656d; toolsf8635babe.
  Full minimal GPU backend/HIPRTC driver/owned C++17 fixture built on .104.
  Reproduced Lt-disabled lowering.cpp:288 missing gfx_default_rocblas; exported
  queries now exist in both configurations, return true without Lt, enabled
  implementation preserved. New upstream-style test; both compile configurations
  syntax-pass and both no-Lt overloads execute/pass. No whitelist/BF16/FP8 change.
  Native gfx1013 wave32 FP32 graph gate22: strided+broadcast pointwise/ReLU,
  row sums17/33/129/513, rocBLAS GEMM K7/17, MIOpen3x3 conv N1/2, changed-input
  second evaluation of ten compiled graphs. CPU-double outputs/host-input guards,
  actual GPU lowered ops, explicit HIP copies; fast_math false/tune off. Not GPU
  allocation-canary/autograd/model/performance/all-dtype or provider qualification.
  Core7/ONNXaudit4/daily5/RCCL6=44 perpass,88/two passes/ten ACK guard records;
  final frozen-tool recheck after export-only audit additions, no math relaxation.
 537 protected hashes/four AOT DBs/links, per-gate/final/unmount kernel logs match;
  same r2bootc6a50cfa.4,782 sourcefiles/172 pinned CMakefiles match archives.
  Five device objects nativegfx1013/no MFMA/WMMA; loader resolves candidate and
  selected native ROCm/SQLite libs (Lt may load transitively; dispatch remains off).
 29 local tests/Black/pycompile/shell/changed C++ formatting pass; Ruff absent,
  full upstream CI not run. Source repair bundle reconstructed/fsck passes.
  Root~1GiB ordinary space required separate durable workstation backing at
  tmp/migraphx-b064/20261025-storage/durable (~109GiB available), SSHFS slave over
  existing SSH/no new listener/key/boot mount. File fsync/mmap/rename/symlink/local
  lock pass. SSHFS3.7.6/FUSE3.18.3 installed, existing r2 FUSE module loaded;
  inherited APK .105 mirror index request refused, no .105 state/deployment change.
  Source copied read-only on native disk; build/cache external. FourCPU/jobs4,
  link1/RAM10GiB/mem+swap12GiB/noGPU/network/read-only build root; independent
 2GiB ccache/no reset/old shared cache untouched. Build~426MiB/cache~43MiB;
  external16GiB admission/14GiB growth limit and root768MiB floor/two-hour bound.
  Source-root wrapper configuration failure, slow-network-source owned stop,
  no-Lt compiler failure and initial fixture include diagnostic all retained.
  Disabled-MLIR unused host warnings remain. No deletion/reserve policy change.
  Active ROCm/Torch/image/defaults/boot/fallback/BIOS/user apps unchanged; Python
  CPU fault distinct/unresolved. Raw accepted external logs-build-r4/logs-fixture-r1,
  WS/logs/migraphx-b064-qualification-r2; local kernel/artifacts/migraphx-b064 archive
 67,753,287B SHAb3c1f4576ec1e9bef91458def557080d340e99ea4cfbf7742d72820f48230c32,
 154 rawfiles14,184,780B plus patch/bundle/verification/cleanup. No owned jobs,
  containers/armed guards; temporary mount/relay closed, backing artifacts retained.
  [Scope/replay/storage/results](bc250_migraphx_b064_results/README.md).
  **Next B06.5:** reconnect documented durable storage; audit/build pinned native
  ONNX parser prerequisites, then small ONNX→gfx1013 FP32 graphs against independent
  CPU references. Preserve CPU ORT/Torch defaults; parser/bindings/ORT GPU provider,
  BF16/FP8/CK/MLIR and larger-model coverage remain separate admissions.

**B06.3 strict MIGraphX C++17 repair completed2026-10-24:** source
 35b40656df72d075a69b155273cf0cc121009683 over95672916e; toolsa03d0e38a.
  DPP-only subwave/wave declarations now share the implementation guard; no
  arithmetic/default-DPP change or new subwave fallback. DPP0 block/LDS retained;
  unsupported subwave request deliberately refuses. Real HIPRTC reproduces old
  DPP0 C++17/Werror failure; patched DPP1/DPP0 compile without diagnostics and
  loaded modules match CPU-double/guards at33/129/513. Native gate71 includes
  retained30,30 subwave width/length cases,new upstream-style block test and10
  JIT compile/execute checks. Retained core7/ONNXaudit4/daily5/RCCL6=93 perpass,
  two passes186/ten ACK guards. Final HIPRTC scratch explicitly under WS/tmp.
 507 protected hashes/four AOT DBs/links and kernel logs unchanged; same r2boot
  c6a50cfa. Extracted native AOT objects contain no MFMA/WMMA in tested kernels.
  Actual unchanged feature predicates compiled/executed: gfx1013 FP8/MX/Lt false,
  BF16 true remains unqualified admission risk, not reproduced numeric failure.
  Source4781 files verified; committed unit-test reflow whitespace-only/hash-pair
  retained, reducer header exact. Fresh reconstruction from preserved upstream
  bundle plus repair bundle/fsck passes.29 local tests/Black/pycompile/shell/new
  unit clang-format pass; Ruff unavailable, full upstream CI not run.
  Builds4CPU/RAM10GiB/mem+swap12GiB/shared20GiB ccache/no reset/noGPU/network.
  Test-wrapper return and host-client AMD-platform omissions fixed before GPU;
  stale baseline-controller path refused before GPU, all attempts retained.
  Provisional full-build budget12GiB+cache2GiB+ordinary rescue floor2GiB=16GiB;
  current ordinary1,110,757,376B vs totalfree26,491,379,712B incl reserved, before
  archive. Budget refused/unmeasured footprint; no reserve policy/deletion change.
  No backend/provider installation/runtime/prefix/image/default/kernel/BIOS/.105/
  user-app work. CPU segfault distinct/unresolved. Raw WS/logs/migraphx-b063-*,
  accepted build-r4/qualification-r2; local kernel/artifacts/migraphx-b063 archive
 17,326,170B SHA6f5225f873fdbfe4ab7b94ee61820b419cfcf2b37f3aa8b50eefe2bc8100496f,
 147 rawfiles9,752,884B plus patch/bundle/verification. No own jobs/containers/
  armed guards running; .104 memory sync only.
  [Scope/evidence/budget](bc250_migraphx_b063_results/README.md).
  **Next B06.4:** establish durable build storage satisfying ordinary-space budget
  without deleting retained artifacts or consuming rescue reserves, then build
  minimal isolated FP32 GPU backend and reference-test actual pointwise/reduction/
  GEMM/convolution graphs. No BF16/FP8/CK/MLIR/parser/bindings/provider admission.

**B06.2 MIGraphX admission audit completed2026-10-24:** toolsa8a6d2327.
  New unmodified source95672916ed289d4be9d230a9be732e1bbad97e8c (rocm-10.1,
  no upstream10.2 tag seen); isolated required rocm-cmake6a7c5b7, existingbase10155
  unchanged.4,780 sourcefiles match staged archive. Full Git reconstruction bundles
  fresh-clone/fsck pass (initial shallow bundle failed, parent history fetched;
  working tree/revision unchanged). No signature qualification of lightweight tag.
  Native gfx1013 wave32 unchanged headers: FP32/FP16 strided+broadcast pointwise
  and12 reduction lengths/CPU-double/guards, DPP1+scalar/LDS DPP0=30 checks.
  Device objects nativegfx1013/no MFMA/WMMA in these kernels. Retained core7/
  ONNXaudit4/daily5/RCCL6 bring52/five gates;29 local tests/Ruff pass.
  Minimal GPU configuration passes with new isolated pinned CMake dependency;
  old dependency refused missing rocm_add_version_resource. GPU/HIPRTC/rocBLAS/
  MIOpen ON; Lt/CK/MLIR/AMDMLSS/Eigen/ONNX/TF/Python OFF. Configure is not a build/
  link/graph proof; no MIGraphX backend/provider installed. BF16 predicate currently
  admitsgfx1013 without dedicated validation (risk, not reproduced numeric bug);
  Lt whitelist excludesgfx1013; FP8/MX remain excluded. DPP0 AOT numerics pass but
  strict C++17 compile refuses unguarded subwave_reduce reference; DPP1 strictpass.
  JIT uses C++17/Werror, so scalar fallback strict JIT remains unresolved.
  Build4CPU/RAM10GiB/mem+swap12GiB/shared20GiB ccache/no reset/noGPU/network.
  Initial fat-object disassembly error corrected via offload-bundle extraction;
  wrapper path TypeError before build retained. No math/tolerance relaxation.
  Ordinary available1,234,112,512B vs totalfree26,614,734,848B incl reserved; no
  reserve policy change/full build started. Need explicit build/reserve budget.
 489 protected hashes/four AOT DBs/links, per-gate/batch/export kernel logs match,
  fiveguards ACK; same r2bootc6a50cfa. Dailyruntimea9a5/profiledb55/image1bb4/
  nativeTorcha524/ROCm/HIP/W8/CPUORT/defaults unchanged. No prefix/image/package/
  kernel/BIOS/user-app/.105/source/cache/prior-artifact deletion. CPU fault distinct/
  unresolved. Raw WS/logs/migraphx-b062-{build-r1,build-r2,configure-r1,probe-r1,
  qualification-r1}; binaries WS/tmp/migraphx-b062/20261024-r2/tmp. Local
  kernel/artifacts/migraphx-b062 archive8,720,498B
  SHA28b8390813ccdb72afe0c1fef2e7ee08c3810d93a88ddedfe95afab1ee9ac540,
  69 rawlogs1,525,263B/eight tools/source archives/hashmap/objects/configure/fiveguards
  verified, full source bundles retained. No own jobs/containers/armed guards running;
  memory sync `.104` only. [Findings/replay](bc250_migraphx_b062_results/README.md).
  **Next B06.3:** fix/test strict C++17 DPP-disabled declaration boundary, preserve
  DPP default, establish gfx1013 feature-admission tests and explicit durable disk/
  reserve budget before minimal FP32 GPU graph build. No BF16/FP8/CK/MLIR/EP claim.

**B06.1 ONNX export/CPU runtime installed2026-10-24:** tools6ae431c9a.
  Pinned onnx1.23.2/onnxscript0.7.2/onnx-ir1.1.0/onnxruntime1.31.0 CPU,
  ml-dtypes0.6.0/protobuf7.36.2/flatbuffers25.12.19; exact PyPI wheel hashes/
  licenses retained. Torchsourcea524 requires torch_2_11 exporter API, verified;
  modern dynamo/opset18, no legacy fallback. Existing NumPy/SymPy/packaging/
  typing_extensions versions preserved. Offline hash-required/nodeps image build,
  RAM1GiB/quota2/networkoff/pullnever; no native ROCm/Torch build/ccache operation.
  Image1bb4ac0ed3766d9c70431c1d5119e03cd4d10e710f4c04cd54254b5b87cd45d5
  retains W8imagee282 layers; dailyruntimea9a5afe3/profiledb556925 select it under
  daily EX lock. Recovery WS/recovery/onnx-b061-before restores prior W8 manifests;
  old images/candidates retained, identity rollback not exercised this batch.
 181 candidate+181 installed/22 gates plus33 early export-probe checks. Bothcohorts
  audit4/export33/ORT-first2: six CPU/GPU-origin dense/CNN/gated+LayerNorm exports,
  batches1/3/4, CPU-double/ONNX NumPy/HIP eager references, host transfer then Torch
  GPU leaf first gradients; three native input refusals; actual ORT node profiles
  CPU only. Available Azure+CPU EP; only CPU selected, Azure unqualified; native
  GPU OrtValue allocation refuses. ORT sessions before Torch import/GEMM backward/
  FFT pass (script-entry Torch absent). Retained W8recipe71/core7/compiled14/
  consumers5+6 pass. No ONNX GPU/autograd/training/W8-export/model-speed claim.
  Old ROCm EP removed upstream1.23; MIGraphX not qualified/registered, separateport.
  Initial qualifier stale report name refuses before GPU, old evidence untouched;
  corrected without math/tolerance changes.29 local tests/Ruff pass.
 477 protected hashes/four AOT DBs/links pinned with manifest/recovery changes;
 23 guards ACK, per-gate/batch/export kernel logs identical; same r2bootc6a50cfa.
  Native Torch/HIP/ROCm/math/SDMA0/heapoff/N3.2/defaults retained; no kernel/BIOS/
  user-app/.105/source/cache/image deletion. CPU segfault distinct/unresolved.
  Raw WS/logs/onnx-b061-{candidate,installed}-r1/final.json; local kernel/artifacts/
  onnx-b061 archive35,255,852B SHAfb55f104f7c83fea3812c3a09acf05de56983df4636c0ae28e984accf3029e44,
 284 rawfiles5,296,329B/seven source+manifest files/seven wheels/models/profiles/
  recovery/23 guards verified. No own jobs/containers/armed guards at handoff;
  memory sync `.104` only. [Use/scope/evidence](bc250_onnx_b061_results/README.md).
  **Next B06.2:** audit pinned MIGraphX source/dependencies, gfx1013 instruction/
  generic/scalar applicability and build headroom before any isolated GPU-provider
  experiment. Keep CPU ORT/W8/native defaults/.105; no architecture spoof.

**B05.6 W8 runtime packaging completed2026-10-24:** final toolsabd8a4e04,
public optional bc250_quant0.1.0. Core copy6832ff9e exact; modulea18e988f changes
only import to relative. Pure wheelMIT/provenance/RECORD, local/host SHAfa71248b;
imagee2821cf8507467b22323272439d636f7c0918e27b8aa2790a5f524b10165a184
retains base506f layers. Dailyruntime5a3a4c2f/profile6101835b now select that image;
native Torcha524/HIP/ROCm/math/SDMA0/heapoff/N3.2 unchanged. Both old manifests/
images retained; daily EX promotion/accept/rollback, SH candidate/ordinary gates.
817 candidate+817 installed checks/32 accepted gates: both cohorts public imports/
RECORD/source identity, W8 module71+recipe71, consumers5/6, core7/FFT245/sparse253;
nine actual old B05.5 checkpoint output/first-gradient/reverse roundtrips. No math/
tolerance change. Initial stageunsupported --cpus twice; final RAM1GiB/quota2/
networkoff/pullnever/pure Python only. Consumeraccelerate thencoretransformers
refuse under wrong plain-Torch test launcher, no GPU error; both generations
restored twice. Final retained fixtures use qualified bc250-train; W8 suites
bc250-torch. Prior candidate808+808/installed146+303 and both failures/old tools/
rollback records retained.457 protected hashes/four AOT DBs/links, same r2bootc6a50cfa,
per-gate/batch/export kernel logs unchanged;81 total guards ACK (32 accepted/49prior).
29 local tests/Ruff/wheel reproducibility/RECORD/license checks pass. No native
build/ccache reset/source/cache/image deletion/kernel/BIOS/user-app/.105 work.
Native direct backward/INT4 still refuse, broader TorchAO/NF4/full QLoRA/compiled/
model quality/speed/peak-memory claims absent; unknown CPU fault distinct/unresolved.
Local kernel/artifacts/quant-b056 archive4,165,410B SHAd57be371cd444afd56739af693c35c5477a1843125dde6475af944275a09aef5;
790 rawfiles18,442,720B/nine tools/wheel/images/recovery/81 guards verified.
No own jobs/containers/armed guards at handoff; memory sync .104 only.
Raw WS/logs/quant-b056-{candidate,installed}-r3/final.json; recovery WS/recovery/
quant-b056-before. [Package/API/evidence/rollback](bc250_quant_b056_results/README.md).
**Next B06.1:** audit pinned ONNX export/runtime execution-provider applicability,
then bounded isolated export/CPU-reference/GPU-interoperability qualification;
CPU export/runtime availability does not qualify a GPU execution provider.

**B05.5 explicit frozen-W8 module/state completed2026-10-24:** code
  e1543d1cf75dc679edcba1b2963e6515eb28a1cc. New optional WS/helpers/
  bc250_quant_b055.py FrozenW8Linear; explicit frozen standard nn.Linear conversion,
  copied bias/policy, no floating-base reference, INT8+FP32 persistent buffers.
  Matching explicit FP16/BF16/FP32 compute, first input/bias gradients via unchanged
  helper6832ff9e; CPU storage/load, native gfx1013 forward/conversion. Dtype/device
  moves preserve exact scales as FP32. State v1 plain metadata+tensors validates
  before own-module copies; complete fields even strictFalse/assignTrue refused.
  Parent model loading not globally transactional; no higher/base/scale gradients.
  295/10 ordinary gates:32 module+36 state+3 owned adapter AdamW/resume eachcohort,
  retained recipe71 eachcohort+daily5/opt-in6. Independent scalar CPU packing,
  CPU-double outputs/input and bias gradients; nine roundtrips/24 malformed states,
  exact continuous-vs-CPU-checkpoint model/optimizer step2. No numerical relaxation
  or production defect found.29 local tests/selected Ruff checks pass.
 445 protected hashes/four AOT DBs/links retained; allten guards ACK, per-gate/batch/
  export dmesg identical. Same r2 bootc6a50cfa-a368-4ba5-ab28-66250b4f1f1f.
  Daily561f/profile446f/Torcha524 unchanged. No active-prefix update/build/ccache/
  package/image/default/native/kernel change, deletion/.105 contact/user app work.
  Module only explicitly imported from helper directory; not installed/default.
  Native direct backward/INT4 refuse; no TorchAO/NF4/full QLoRA/model-speed/quality/
  peak-memory/long-run claim. Historical CPU fault distinct/unresolved.
  Raw WS/logs/quant-b055-qualification-r1/final.json; local kernel/artifacts/
  quant-b055 archive510,746B SHA2c0099691e7744e37f4623a269df593d90a48009731bd56de969bc5cbc1b57e7,
  96 rawfiles2,070,943B/six tools/ten guards verified. No own jobs/containers/armed
  guard at handoff; memory snapshot synced to `.104` only.
  [Usage/contracts/evidence](bc250_quant_b055_results/README.md).
  **Next B05.6:** package explicit W8 helper/module in independent optional runtime
  candidate, public imports/state/gradient/retained-consumer gates before locked
  promotion. Keep dense defaults/old images; no automatic conversion/full QLoRA.

**B05.4 strict W8 validation optimization completed2026-10-24:** code917b2823a.
Only explicit WS/helpers/bc250_quant_b051.py updated, native prefixes/images/
Torchsourcea524/daily561f/profile446f unchanged. Normal valid scalar reads/syncs
2→1; all data checks/messages/scale-first priority/limits/AMP retained, pack and
native forward/backward AST unchanged. HelperSHA6832ff9e (old54c98373 saved).
First tensor-list candidate slower/rejected; trace-only assumption fails after89
checks, source/data retained. Scalar candidate90 comparisons,15 alternating pairs*
30 calls/four-CPU affinity:daily forward1.075–1.145x/FB1.029–1.080x, RCCL1.077–
1.145x/FB1.029–1.075x in three FP32 fixtures.15/15 forward pairs and12–15 FB pairs
favor candidate. No model/dense-baseline/peak-memory/universal performance claim.
Current333 candidate/comparison+142 actual-import checks=475/eight gates. Bitwise
output/input/bias/pack equality, independent CPU refs,60 value/12 error-priority
cases and retained71/cohort/25 refusals pass.442 protected hashes/four AOT DBs/
links retained with old helper backup;29 local tests/selected Ruff checks pass.
User-confirmed restart before longer run starts; old5c04 data separate. Current
r2 bootc6a50cfa-a368-4ba5-ab28-66250b4f1f1f: eight live states ACK, three old
ACK logs preserved after /run cleared. Per-gate/current-batch dmesg matches;
no cross-boot identity/no agent reboot. No native build/ccache operation/.105/user
app/BIOS/fallback/deletion. Native W8 backward/INT4 still unsupported, helper
first gradients retained; not installed/no auto model conversion. Unknown CPU
fault distinct/unresolved. RecoveryWS/recovery/quant-b054-before/old helper,
locked identity rollback not exercised. FinalWS/logs/quant-b054-qualification-r1/
final.json +quant-b054-ordinary.json; no own jobs/containers/armed guard.
Local kernel/artifacts/quant-b054:626,696B archiveSHA75a22a82c56d551f55d878d6ec67edd8241a0f2c66b9b9ac3ae8aa4969609418,
104 rawlogs2,968,279B/eight tools/versions/traces/recovery/current guards verified.
[Results/contracts/recovery](bc250_quant_b054_results/README.md).
**Next B05.5:** owned explicit frozen W8 nn.Module conversion/versioned state
roundtrip, independent output/input/bias/adapter first-gradient references;
keep FP32 scales/checks/bounds/opt-in/defaults/.105, no model-quality/speed claim.


**B05.3 matched RCCL layout repair selected2026-10-24:** tools411e72917,
Torch sourcea524 unchanged. Independent WS/pytorch-quant-b053 from protected14cf
RCCL build+two-file full tracked overlay; generated HIP exactly two contiguous
calls. Pip exit0/4CPU/4jobs/link1/10GiB RAM/12GiB memory+swap/shared20GiB ccache/
no reset/noGPU/noNet. Initial~4.8GiB disk/~3GiB copy,1.25GiB floor enforced;
~1.8GiB before archive. No old source/build/cache deleted. Shared validator/harness
explicit backend/profile selector only; original helper hashes/backups preserved.
Candidate1732/24 plus ordinary371/5 pass, exact layout220/W8recipe71/retained
RCCL1441, public native220/consumers6/RCCL64/guards76/daily5.434 protected hashes/
four AOT DBs retained; intentional helper changes/registry switch,29 guards ACK/
per-gate/batch/final dmesg identical;29 local tests/selected Ruff/shell checks pass.
Registry `/opt/bc250-stack/profiles/rccl-single` →`/opt/bc250-stack/dev/rccl-single-b053`;
profile446f345e1438f093967a495c816859f4505bd06869dbea382842c2f081f14fd7,
libtorch_hipa31cbb15; RCCLca790 unchanged. Default daily native2f276/runtime561f/
Torchsourcea524/NCCLoff/image506f/HIPa116/ROCm/defaults unchanged. Both selected
cohorts now contain native layout fix; old14cf/profile42eb and old builds retained.
RecoveryWS/recovery/quant-b053-before/old registry/profile/helpers; locked rollback
not exercised. Same r2 boot5c04afbb/no reboot/.105/user-app/BIOS/fallback/delete.
Native backward/INT4 still unsupported, W8 not installed/no auto model conversion.
No numeric FP8/multi-device/protocol/transport/scaling/symmetric device/actual
payload offload/speed/universal claim; historical CPU fault distinct/unresolved.
Local kernel/artifacts/quant-b053:200,536,100B archiveSHA171924233619ce789f958f195d275b6fd07e4522b45a54ed883a015212710eb6,
332 rawlogs8,370,556B/native/source/eight tools/recovery/29guards verified.
WS/logs/quant-b053-qualification-r1/final.json and quant-b053-ordinary.json;
no own jobs/containers/armed guard. [Report/use/recovery](bc250_quant_b053_results/README.md).
**Next B05.4:** measured strict W8 validation/native forward/backward overhead,
bounded optimization with CPU/gradient/error-contract references and paired timing;
keep defaults/artifacts/.105 and error checks, no model speed claim.


**B05.2 native INT8 layout correction installed2026-10-24:** Torcha5249802372882dedccb91d3fd4d73be9fcdf2bd,
tools506ae3b9e. Generic `_weight_int8pack_mm` now normalizes converted FP32
activations/scales; generated HIP source differs exactly by two contiguous calls.
Same math/cast, no native backward or packed INT4 enablement. Device-generic18
regressions/eight targeted spin linters pass; old unused import removed, no fullCI.
Independent WS/pytorch-quant-b052 from2b+six-file full tracked overlay; pip exit0,
4CPU/4jobs/link1/10GiB RAM/12GiB memory+swap/shared20GiB ccache/no reset/noGPU/
noNet. Source verifier rejects auto-HIP NCCl translations; both match protected
B03.7 bytes; seal CMake0/1 spelling fixed, failures/early pre-GPU refusal retained.
No build retry/tolerance relaxation. Candidate947/11 +ordinary231/3 checks pass,
including216 exact native dtype/layout composites+boundaries, W8 first-gradient
recipe and sparse/FFT/compiled/profiler/Gloo/vision/audio; no broad CI/model claim.
402 protected hashes/four AOT DBs/links retained with intentional daily target/
manifest change and old exact metadata preserved.14 guards ACK/dmesg identical.
Active daily WS/pytorch-quant-b052/Torcha524/NCCLoff/Glooon; libtorch_hip2f2761cd,
runtime561f014e31c4ca0bba73861d56125e4e18124867904b7c23cff8d7e0fa36d2a1.
Image506f/HIPa116/ROCm/N3.2/math/SDMA0/heapoff/r2 boot5c04afbb unchanged;
no reboot/.105/user-app/BIOS/fallback/delete. Old2b build/source retained;
WS/recovery/quant-b052-before/runtime.json=32fd4559, locked rollback not exercised.
Opt-in RCCL14cf/profile42eb remains old/unfixed, ordinary loading passes; W8 helper
normalization still protects its use. No W8 package/model auto-conversion,
performance/universal/higher-gradient claim; unknown CPU fault distinct/unresolved.
Local kernel/artifacts/quant-b052:193,263,147B archiveSHAfce8095e5c163f36d100f9730137faad0fcd06b98014448e1f17db729a002288,
153 rawlogs5,516,695B/native/eight tools/source/recovery/14guards verified.
WS/logs/quant-b052-qualification-r1/final.json and quant-b052-ordinary.json;
no own jobs/containers/armed guard;29 local tests/selected Ruff checks pass.
[Report/recovery](bc250_quant_b052_results/README.md).
**Next B05.3:** disk/resource preflight, independent RCCL-enabled Torch layout
repair and exact/retained RCCL-gradient references before matched locked profile
update/ordinary acceptance/rollback. Preserve daily/.105/old sources/caches.


**B05.1 native low-bit audit/W8 prototype completed2026-10-24:** codec4383972c.
Existing generic HIP `_weight_int8pack_mm` contiguous FP32/FP16/BF16 references
pass; strided activation15/15 and strided FP32 scales12/15 wrong outputs
reproduced. Native bugs **not fixed**; input backward explicitly refuses. Native
packed INT4 gfx1013 conversion refuses CDNA2 gate; no architecture spoofing.
TorchAO/bitsandbytes absent/Triton3.8 available; no external low-bit package claim.
INT8/INT4 source hashes match active Torch2b and patched source14cf.
Explicit helperWS/helpers/bc250_quant_b051.py: frozen per-channel INT8/FP32 scales,
round-even/clamp[-127,127]/zero-row1, safe normalized native HIP forward/custom
first input derivative via scaled upstream/transposed bytes; bias/owned adapter
first grads. No full float base materialization, integer/scale/base/higher grads.
Input dtype retained under AMP/bias after native output cast; finite validation
host sync. Dimensions4096/weight4M/activation-output1M/MAC128M caps; not total
allocator bounds. Not installed/default, not packed INT4/NF4/full QLoRA.
Audit7+prototype71+core7+compiled14+daily5+opt-in6=110/six accepted gates.36
CPU scalar-pack/double output/input/bias composites, byte/tie/zero/guard/private
stream/FD/adapter/AMP/25refusals/two native GPU kernel trace checks pass. Initial
collector's null-gradient assumption failed at explicit refusal; corrected
collector/backend unchanged, failed source/result preserved.29 local tests and
selected Ruff checks pass;377 protected hashes/four AOT DBs/links retained.
Seven guards ACK including failure, per-gate/batch/all-run dmesg identical.
Recipe medians0.26045/0.25816ms vs dense0.08222/0.08134ms (two small FP32 shapes,
3 alternating pairs*3 eager calls incl validations/sync); ~3.2x slower. Fresh
packed payload33,280/66,560B vs FP16 65,536/131,072B (~1.97x smaller), excludes
caller-retained base/temporaries/activations/allocator. No model quality/speed/
peak-memory claim. Daily runtime32fd/Torch2b/HIPa116/image506f and opt-in14cf/
RCCLca790/profile42eb retained. R2 boot5c04afbb-ff2f-4f4b-9eac-8238bb9288fc,
new since B03.8; cause not investigated/no agent reboot. No native build/ccache
operation/.105 connection/user application touch/deletion. Local artifacts
kernel/artifacts/quant-b051:475,903B archiveSHA6051eee04658afdb81fd017087ae01f2b268843303202965a57906447ebb2f41,
63 rawlogs1,497,225B/five helper hashes/native source/seven guards verified.
WS/logs/quant-b051-qualification-r1/final.json; no own jobs/containers/armed guard.
Unknown CPU fault distinct/unresolved. [Report/API/limits](bc250_quant_b051_results/README.md).
**Next B05.2:** native activation/scale layout fix in qualified Torch, isolated
resource/ccache-bounded pip build and independent layout regressions; preserve
native backward refusal/daily/opt-in/.105/fallbacks. No model auto-quantization.


**B03.8 ordinary-launch opt-in RCCL completed2026-10-23:** code36cabf9be.
`.104` registry `/opt/bc250-stack/profiles/rccl-single` points to
`/opt/bc250-stack/dev/rccl-single-b038`. Both installed launchers accept
`--runtime-profile rccl-single`; omit it for unchanged daily Torch2b449c4/NCCLoff/
Glooon/runtime32fd4559/image506f/HIPa116/r2 boot2db1c7c2. ProfileTorch14cf/native
libtorch_hipeb405cb1 and copied RCCLca790dc7/SONAME links; manifest42eb7631.
Reuses daily ROCm/cache/consumers, not relocatable independent release. Native
identity verification, RO `/workspace/rccl`, controlled LD path and fixed
window0/CUMEMboth0/P2Poff/IBoff/lo/async-error1; conflicting overrides refused.
Private user-owned context preserves selection. Daily SH lock before hash checks
through launch; updater rejects independent profile locks (use daily EX).
Ordinary1461/26 pass:old1441 plus packaged5/6 and daily4/5; retained22 gates use
installed bc250-train without test-only RCCL mount/env injection. Supplemental
private-context5, nine CPU refusals and live lock exclusion pass. Local29 tests
and selected Ruff checks/format pass.372 protected hashes/four AOT DBs/links
retained; two launcher copies intentionally replaced/backed up.27 guards ACK,
accepted per-gate/main batch dmesg identical. No native rebuild/ccache operation/
reboot/.105 connection/user application touch/deletion. Rollback helper retained,
not exercised; profiles/builds/cache/source preserved. Local kernel/artifacts/
rccl-b038:7,580,482B archiveSHA64beea0d3b9396f84207d5d055786b4028f7a9009612fd3f1d656a549f6a2efa,
284 rawlogs6,078,114B/profileDSO/recovery/27guards/source verified. Unknown CPU
fault distinct/unresolved. Rank-one/single-GPU only: no transport/scaling/full
backend/speed/symmetric device/actual offload payload claim. WS/logs/rccl-b038-
qualification-r1/final.json +WS/logs/rccl-b038-context.json; no own jobs/containers/
armed guards at handoff. [Use/results/recovery](bc250_rccl_b038_results/README.md).
**Next B05.1:** audit A08/pinned low-bit paths on .104, select an owned bounded
weight-only quantization recipe and independent output/gradient references;
keep daily defaults/.105, no architecture spoofing/new Torch wheel.


**B03.7 isolated FP8/API admission guards completed2026-10-23:** tools819bb624f,
Torch14cfef2 overf807099/ade6fcc; RCCL38fd6/B03.4 unchanged. Lowlevel FP8 numeric
reductions reject before comm while byte-copy remains; filtered gfx1013 PG FP8
admission uses actual HIP device, other-device policy retained. f807 build passes,
r1 fails missing PG Reduce refusal after33 checks.14cf fixes Reduce/_reduce_oop;
rebuild-r2 passes; exact six-line hipified follow-up delta verified. Failed source/
all native libs preservedWS/recovery/rccl-b037-reduce-before, r1 release reused.
Initial8/followup3 targeted spin/lintrunner-a pass; devicegeneric test added, not
full CI. R2=1441/22 gates:new76+retained1365. New36 refusals (15 lowFP8/20 PG/
multicastallocator),32 guarded raw-bit FP8copy refs/all256 encodings/four formats/
17+257; generic MemPool registration/PreMulSum/later allocation/deregister; window
optout success/NULL; stats/suspend-resume0-1-0/post-GPU refs; device/DSO/lifecycle.
Tracked offload bytes allzero: no payload relocation/reclamation/speed proof.
Symmetric window skip is not functional support; full symmetric device/graphs/
concurrency/shrink/multi-device/transport/protocol/scaling unqualified.356 protected
paths/four AOT DBs/links unchanged incl old340;22 accepted guards ACK/gate+batch
dmesg same;23 total incl failure ACK. CandidateWS/pytorch-rccl-b037/release-r2
isolated, libtorch_hipeb405cb1. Active Torch2b449c4/NCCLoff/image506f/runtime32fd/
HIPa116/native systems1f9609/ROCm/defaults unchanged.4CPU/4jobs/one link/10GiB/
12GiB/shared20GiB ccache/no reset/no GPU/network compile. Same r2 boot2db1c7c2,
no reboot/.105/BIOS/fallback/user application touch/deletion. Unknown CPU fault
distinct/unresolved. SHA-verified1,083,504,640B archive/291 rawlogs8,996,665B/native/
23guards/recovery local kernel/artifacts/rccl-b037. BaselineWS/logs/rccl-b037-
qualification-r2/final.json; helper backupWS/recovery/rccl-b037-before. No own jobs/
root SSH; historical running checkpoint superseded.
[Report/profile/reproduce](bc250_rccl_b037_results/README.md).
**Next B03.8:** package opt-in bounded single-device runtime, RCCL DSO RO mounts/
loading and explicit window/CUMEM/P2P/IB settings; ordinary-launch output/backward
qualification using owned fixtures. Keep daily prefix/Gloo/NCCL-off and rollback.

**B03.6 isolated public RCCL contracts/broader regressions completed2026-10-23:**
tools9dc76a795, candidate Torchade6fcc/RCCL38fd6/B03.4 unchanged. No production
patch/native library rebuild/ccache operation/default change. R1 passes1365/21:
178 contracts+RCCL64/GPU Gloo41+retained18-gate1082. New178:128 low-level identities
(eighttypes/twosizes/disjoint-output vs exact-alias/fourcollectives),8 broadcasts,
5zero wrappers/4availability/12host refusals;3 PG PreMulSum/init/destroy cycles,
subgroup/event handoff,8PG host refusals,4 BF16/FP32 analytic DDP composites
(bucket views/no_sync/two microbatches/two SGD updates),loadedDSO. Exact CPU refs/
offset guards/private streams. Small DDP constant1/8/identity case; no broad BF16/
FP8/partial-overlap/higher-gradient/graphs/registration/transport/multi-device/
protocol/scaling/speed claim. No tolerances weakened.340 protected paths/four AOT
DBs/links unchanged vs B03.5;21 guards ACK/gate+batch dmesg identical. Daily
Torch2b449c4/NCCLoff/image506f/runtime32fd4559/HIPa116/native systems1f9609/ROCm/
defaults unchanged. Isolated B03.5 release-r3 reused; no deployment. Native payloads
remain verified B03.5/B03.4 artifacts. Source audit notes low-level FP8 mapsuint8,
not numeric FP8 proof; fnuz PG host refusals tested, e4m3fn/e5m2 version admission/
symmetric/registration/offload remain separate. Same r2 boot2db1c7c2/no reboot/
.105/BIOS/fallback/user application inspection/modification/test/deletion. Unknown
CPU fault unresolved/distinct from VM-heap/CSR/B04.2 queue. SHA-verified6,543,360B
archive/257 raw logs5,386,517B/21guards/helpers/recovery local kernel/artifacts/
rccl-b036. BaselineWS/logs/rccl-b036-qualification-r1/final.json, helper backup
WS/recovery/rccl-b036-before; no own jobs/root SSH.
[Report/reproduce/limits](bc250_rccl_b036_results/README.md).
**Next B03.7:** audit/guard excluded/version-admitted FP8 arithmetic versus byte
copy and symmetric/registration/offload paths; define bounded single-device
promotion profile while daily prefix stays unchanged. No multi-device proof.

**B03.5 isolated Torch/RCCL integration completed2026-10-23:** toolsce089950f;
source/candidate Torchade6fcc1ae over2b449c4, RCCL38fd6/B03.4 candidate unchanged.
Native gfx1013 consumer build/link succeeds with USE_NCCL/SYSTEM_NCCL=1; ROCm
find_package(rccl),2.32.3>=2.23,42 declared called APIs export (lexical scope only).
Reproduced BF16 host PreMulSum zero after31 passing checks; float template passed
wrong scalar bytes. Torchade6fcc uses at::BFloat16 and adds BF16 regression loop.
Eight spin linters/lintrunner-a pass. First GPU negative is host only; corrected
host/device refs pass. R2 native64/GPU Gloo41 pass then old CPU NCCL-off contract
fails; explicit --nccl-available fixture option retains default/numerical checks.
R3 passes172/seven gates:64 RCCL+41 GPU Gloo+31 CPU+14dense+14CNN+1profiler+7core.
RCCL rank1/one physical GPU:24 offset/stream SUM identities,24 PreMulSum fourfloat
CPU refs,10 other identities,2 FP32/64 analytical gradients,2 two-step SGD/DDP
CPU refs,2 metadata/loaded DSO/lifecycle composites. No multi-GPU/transport/full-
protocol/library/scaling/speed/higher-gradient claim; small exact float inputs.
340 protected paths/four AOT DBs/links unchanged; seven accepted guards ACK and
all accepted gate/batch dmesg identical. All11 guards incl failed runs preserved.
CandidateWS/pytorch-rccl-b035/releaseWS/rccl-b035/candidate-release-r3 isolated;
active Torch2b449c4/image506f/runtime32fd4559/HIPa116/native systems1f9609/ROCm/
defaults unchanged, daily NCCL off. Original native/source recoveryWS/recovery/
rccl-b035-bf16-before; shared helpers/metadata recoveryrccl-b035-before. R1/r2
release links target the reused build, original payloads are in recovery. Exact
production overlay over retained2b archive; upstream test change local/bundle.
4CPU/4jobs/one link/10GiB/12GiB/shared20GiB ccache/no reset/no GPU/network compile.
Same r2 boot2db1c7c2, no reboot/.105/BIOS/fallback/user application touch/deletion.
Unknown CPU fault distinct/unresolved. SHA-verified1,080,432,640B archive/142 raw
logs6,797,890B/native/11 guards under local kernel/artifacts/rccl-b035; hipified
source delta verified only intended scalar type. No jobs/root SSH remain.
BaselineWS/logs/rccl-b035-qualification-r3/final.json.
[Report/reproduce/boundaries](bc250_rccl_b035_results/README.md).
**Next B03.6:** isolated public out/type/alias/error/stream/lifecycle contracts and
BF16 DDP where applicable; broader stack regression before promotion. Keep daily
runtime/.105 and explicitly audit filtered/excluded API paths.

**B03.4 broader isolated native RCCL completed2026-10-22:** tools7ea133762,
source38fd6ba67d unchanged;375 specialized objects: SIMPLE201/LL174. Ring/Tree typed filter,905 generated files,
375 admitted/489 guarded-out entries; LL128 guard retained, no spoofing. Build/
client exit0.629/4 gates=native550+GPU Gloo41/CPU31/core7. Native120 PreMulSum and
420 AllReduceWithBias CPU references across ten dtypes/17–1025/in-out/scalars,
non-aligned bias guards and integer comparisons/unsigned modulo; small exact
floating inputs only. Six identity/four metadata-lifecycle checks separate.
One rank/device: no multi-GPU/protocol runtime/autograd/performance claim.
328 protected paths/four AOT DBs/links unchanged, four guards ACK, dmesg identical.
CandidateWS/rccl-b034/install-r1 isolated; active runtime/Torch NCCL-off unchanged,
B03.3 preserved. Build4CPU/4jobs/one link/10GiB/12GiB; client2CPU/4GiB/5GiB;
no GPU/network during compile. Shared20GiB ccache standard CXX/client, custom
RCCLDEV/device-link bypasses launcher; no cache speed claim. Same r2 boot; no
reboot/.105/BIOS/fallback/user application touch/deletion. Unknown CPU fault
unresolved/distinct from fixed VM-heap/CSR/B04.2 queue. SHA-verified15,267,840B
archive,50 raw logs1,353,457B/native payloads/guards/recovery under local
kernel/artifacts/rccl-b034. BaselineWS/logs/rccl-b034-qualification-r1/final.json;
helper backupWS/recovery/rccl-b034-before. No own jobs/root SSH remain.
[Report and exact identities](bc250_rccl_b034_results/README.md).
**Next B03.5:** pinned Torch NCCL/RCCL consumer/API applicability and independent
NCCL-enabled Torch candidate with one-device output/first-gradient references,
before deployment; retain current Gloo/runtime/.105.

**B03.3 isolated native RCCL feasibility/readiness completed2026-10-22:** tools7e7896b10,
RCCL source38fd6ba67d over1f9609fd6 (c653 NVTX_NO_IMPL header-selection/no-op macro,
38fd direct iostream/CRLF preserved). r1 duplicate full/stub types and r2 missing
std::cerr declaration failed; r3/client build exit0, failures/originals preserved.
Explicit gfx1013/filtered FP32 Ring-SIMPLE build, no spoofing. One-rank/CE/DDA code
also compiles; symmetric/rocSHMEM/GIN/MSCCL/ROCTX/Python off. No full-library or
multi-rank Ring correctness claim. Isolated161/4 gates=native82+Gloo GPU41/CPU31/
core7. Native72 oneRankReduce PreMulSum cases/six dtypes/three sizes/in-out/host-
device scalars/CPU exact output/input guards/non-default stream, six identity
copies/group and four metadata/lifecycle checks. DSO path/version/gfx1013 verified.
323 hashes/four AOT DBs/links pass; four guards ACK/gate+batch dmesg identical.
DSO4af5420f outside active prefix; Torch NCCL off. Active image506f/runtimeJSON
32fd4559/Torch2b449c4/HIPa1162308/native systems1f9609/ROCm/defaults unchanged.
4jobs/4CPUs/one link/10GiB RAM/12GiB mem+swap/shared20GiB ccache/no reset/noGPU;
client2CPUs/4GiB/5GiB mem+swap/noGPU. Same r2 boot2db1c7c2/no reboot/.105/BIOS/
fallback/app touch/delete. Unknown CPU fault remains distinct/unresolved.
Local kernel/artifacts/rccl-b033 source/native/client/55 logs1,495,447B/four guards/
recovery/archive3ee65ba1 verified11,816,960B. BaselineWS/logs/rccl-b033-qualification-
r1/final.json; no own jobs/root SSH closed. [Report](bc250_rccl_b033_results/README.md).
**Next B03.4:** broaden isolated native RCCL kernel types/protocols beyond FP32
Ring/SIMPLE and qualify feasible single-device operations before Torch NCCL
integration; no multi-GPU/P2P/XGMI/GDA/SDMA transport/scaling proof from one GPU.

**B03.2 Gloo HIP host-staged collectives/GPU DDP qualified2026-10-22:** code418db3a90.
No production patch/native build/ccache/runtime/default change. Two ranks share
cuda0/Gloo TCP/loopback; collectives use pinned CPU staging, model math stays HIP.
Installed108/6 gates =newGPU41+CPU31+compiled28+profiler1+core7. Eight SUM dtypes,
three sizes/offset guards/private-stream async consumption; other FP64 reductions/
collectives, FP32/64 analytic first gradients, FP32/64 GPU DDP SGDmomentum+AdamW/
two updates versus independent CPU global-batch output/all params+gradients.
Fixed tolerances,318 hashes/four AOT DBs/links retained; six guards ACK/gate and
batch dmesg identical. Legacy functional/reduce-scatter FutureWarnings retained;
newer APIs/GPU send-recv/BF16 DDP/RPC/checkpoint/non-loopback/multi-device/scaling
remain unqualified. Raw-pointer GPU send-recv path intentionally unexecuted.
RCCL source permits explicit target validation/non-GFX9 wave32 branch; no native
build/runtime proof or hardware-impossibility claim. Native Torch2b449c4/image506f/
runtimeJSON32fd4559 unchanged. Miner scheduling respected; same r2 boot2db1c7c2,
no own reboot/.105/BIOS/fallback/user app touch. Unknown CPU fault remains distinct.
Local kernel/artifacts/distributed-b032 retains73 raw logs1,659,204B/six guards/
helpers/archive8cdbb0e5 verified; baselineWS/logs/distributed-b032-qualification-r1/
final.json. No own jobs/root SSH closed. [Report/use](bc250_distributed_b032_results/README.md).
**Next B03.3:** independent native gfx1013 RCCL configure/build feasibility and
single-device readiness, keeping current prefix/.105/defaults; no multi-GPU/P2P/
XGMI/GDA/SDMA transport/scaling claims.

**B03.1 native local CPU Gloo installed2026-10-22:** integration8e6416f39,
patched Torch source/native2b449c4. Independent native build exit0; distributed/
Gloo/TensorPipe+libuv ON, NCCL(RCCL)/MPI/UCC/ibverbs/TLS OFF. Pinned dependency
files verified (Gloo276/TensorPipe292/libuv467), existing Gloo HIP guard conversion
retained as reconstruction evidence, not a new runtime repair. CPU-only31 checks
without GPU; candidate-r2/installed-r3 each1082/18 gates (new31+retained1051).
Two loopback CPU ranks/torchrun/stores/eight SUM dtypes/collectives/analytic first
gradients and FP32/64 DDP SGDmomentum+AdamW/two updates versus independent global
batch references. GPU distributed/RPC/non-loopback/multi-board/checkpoint recovery
and performance remain unqualified. Candidate-r1 historical absent-codec assertion
fixed by selecting existing --codec-available; numerical tolerances unchanged.
Native Torch DSOs changed; image506f3b26/ROCm/kpacks/N3.2/math/SDMA0/heapoff retained.
RuntimeJSON32fd45595989e09597dbb0fcfc8992595e8861c0581a49827fcf965e35a865e0.
Prior306 hashes pass, installed318 includes new/recovery identities and intentional
runtime metadata update; four AOT DBs preserved.47 guards ACK/accepted gates and
passing whole batches dmesg identical. Exact old link/runtime recovery in
WS/recovery/distributed-b031-before; locked deploy helper --rollback unused.
Build4jobs/4CPUs/one link/10GiB RAM/12GiB mem+swap/shared20GiB ccache/no reset/noGPU.
New r2 boot2db1c7c2 observed after build, no agent reboot/cause unknown; miner
preflight refusal preserved, user freed GPU before qualification. User application/
.105/BIOS/fallback/old source/build/cache/image/release untouched. Unknown CPU fault
unresolved/distinct from VM-heap/CSR/B04.2 queue/B02 assertion cleanup. Local
kernel/artifacts/distributed-b031: source/native payload/579 raw logs15,615,864B/
47 guards/recovery; archive7e53e0ed SHA-verified. BaselineWS/logs/distributed-b031-
qualification-r3/final.json. No own jobs/root SSH closed.
[Report/use/recovery](bc250_distributed_b031_results/README.md).
**Next B03.2:** inspect pinned Gloo HIP tensor handling/native RCCL applicability;
select bounded single-board GPU tensor/software-transport probe, no .105/multi-
board/P2P/XGMI claims.

**B02.1 explicit bounded CSR solve installed2026-10-21:** coded2c3ea105,
bc250_sparse0.1.0 SHA272cc68c8e898e9f8eed5816a55b5ae9e2076861b575304713daae47f0b300a6.
Pinned general Torch API rejects ROCm; hipSOLVER QR densifies, CSRRF requires
initial factors; existing N2.13 triangular solves do not supply factorization.
Opt-in GPU dense-LU fallback, canonical square CSR1..1024/RHS1..64, four dtypes,
major-dense-storage estimate<=64MiB (not total allocator/workspace cap). Safe GPU
metadata predicates before index consumption, finite/zero-pivot/FP64 residual
checks; explicit AMP-disabled first adjoint. Parts API supports complex/real
stored-value and RHS gradients; CSR wrapper real value/all RHS gradients,
complex values copied to avoid native CSR view bug. Native public API unchanged;
no CPU solve/scalable sparse/factor reuse/performance claim.
Candidate-r4/installed-r5 each381 checks=new253/triangular-edge80/sparse41/core7.
New192 CPU-double output/all-leaf grads (four dtypes/two indices/three sizes/two
RHS/layouts/AMP), eight pivot/analytic/wrapper,13 diagonal65/257/1024/RHS64,
one exact budget, two finite differences,35 refusal/contract checks, one traced
fixture+135-kernel trace. Maxoutput4.011e-7/absolute residual7.807e-7 in fixtures;
fixed single rtol3e-4/atol3e-6, double2e-10/2e-12, FD<1e-8 (max1.189e-11).
Initial launch failures: missing /var/tmp restored1777. Audit3 complex constructor
grad rejection retained; revised audit8 passes. Candidate-r1 AMP gradient rounding,
r2 complex CSR values-view error and r3 native invariant-checker device assertion/
cleanup139+libc fault retained; helper precision/copy/safe predicates fix their
respective paths. Fresh-process core7 recovery passes, no reboot. This known
assertion cleanup trigger does not resolve historical unknown Python CPU faults;
VM-heap/CSR/B04.2 queue defects remain distinct. Native bad-CSR assertions unchanged.
Image506f3b262fa5fe8c29b9babec07b03078dd6f85492a9ac2cffc65955297e4349;
runtimeJSONf09cae594646107e8101e2bc383e4ee7257a3c3354c218805a9a6350091f35b1.
PurePython image layer/no native build/ccache operation; old5d0a6 image/history/
Torch/ROCm/defaults retained. Locked rollback via deploy_sparse_b021_on_host.py
--rollback restores exact WS/recovery/sparse-b021-before/runtime.json; unused.
Sixteen600s guards ACK; all eight accepted gate dmesg+two whole batches identical;
306 stack hashes/four AOT DBs/links preserved. User application excluded; same r2
bootc646d874, no agent reboot/.105/BIOS/delete. Local kernel/artifacts/sparse-b021
has variants/recovery/154 SHA-verified raw logs4,956,287B/16 guards/archive8b05a494.
BaselineWS/logs/sparse-b021-qualification-r5/final.json; disk21,405,282,304B observed.
No own jobs/root SSH closed. [Report/use/recovery](bc250_sparse_b021_results/README.md).
**Next B03.1:** inspect Torch distributed/Gloo/RCCL configuration; select bounded
local CPU process-group readiness first, .105 unchanged. Native large sparse
factorization remains open; compile/graphs/higher gradients/universal stability
and hardware collectives are unqualified.

**B01.1 explicit BF16 FFT fallback installed2026-10-21:** code427492188,
bc250_fft0.1.0. Audit3 confirms GPU BF16 FFT rejects outside/inside AMP while
explicit float promotion works/backpropagates; existing sources/no old helper
reviewed. Opt-in18 FFT adapters promote only BF16 to FP32 on same device; C64/F32
outputs without out, inherited native out/other-dtype contracts and cast-back
BF16 leaf gradients. No global torch.fft change/native BF16 FFT/CPU fallback.
Final candidate/installed327 each: new245 (216 ops/norms/layouts/AMP CPU-double
output/all-leaf grads; non-BF16/contract/out guards; explicit DFT/finite differences/
32-kernel native FFT trace), retained real72/contracts3/core7. Maxoutput6.404e-7,
gradrelativeL2.001981<.004; fixed bounds. Initial dynamic candidate327 pass,
explicit export assignment fixes lint visibility; full final replay. Owned temp
stage-r2 failure retained/r3 fixed.13 guards ACK/all gates+batch dmesg identical,
306 stack hashes/four AOT DBs/links retained. PurePython image layer/no native
build/cache reset/ROCm/Torch/default change; image5d0a6d9cef758b2f7677a2b52b9d29ac51e24970703259a10ca8e44bee93feb1.
RecoveryWS/recovery/fft-b011-before/runtime.json via locked deploy helper rollback,
old94c610 image/history retained; rollback unused. User application excluded,
same r2 bootc646d874/no agent reboot/.105/BIOS/delete. Local artifacts/fft-b011
under kernel preserve modules/recovery/141 SHA-verified raw logs4,140,885B/13 guards.
BaselineWS/logs/fft-b011-qualification-r3/final.json; no own jobs/root SSH closed,
disk21.440GB observed. Unknown CPU fault unresolved/distinct from VM-heap/CSR/
B04.2 queue. [Report/use/recovery](bc250_fft_b011_results/README.md). No memory/
speed/all-device/shape/compile/graph/higher-grad/stability claim. **Next B02.1:**
review sparse solve/factorization APIs versus N2.13 triangular evidence before
choosing native integration or bounded explicit fallback with reference contracts.

**B04.5 CPU common image codecs installed2026-10-21:** code93a601b36,
TorchCodec source3eecace0 unchanged, no new source patch. JPEG/PNG/WebP enabled,
CUDA/nvJPEG/image-default off; AVIF/GIF/HEIC disabled as build choice. Missing
runtime libwebpdemux2 copied from existing build image to private image lib dir,
not dpkg-registered; exact DSO/copyright/loader config exported locally. Candidate/
installed each203 checks: image86/video55/audio46/vision9/core7. Independent PNG8/
16 samples, modes/dtypes/orientation6, JPEG Pillow comparisons, exact WebP pixels/
animation, PNG/JPEG encoder forms/views, refusals and GPU spatial output/input
backward+CPU save pass fixed gates. CPU codecs, explicit GPU JPEG rejection;
other formats/palette/interlace/all EXIF/timing/HDR/security/performance/universal
stability unqualified. [Report/use/recovery](bc250_image_b045_results/README.md).
Image94c610368d0fcf596a9f9f6355a5d5d28fd1111d54d4033c46e6b5977cae1ece;
old87875 retained, existing ROCm/Torch/defaults/kpacks/B04.4 repair unchanged.
Locked rollbackWS/recovery/image-b045-before/runtime.json, old helper preserved;
rollback unused.4 Ninja/CPUs/8GiB RAM/10GiB mem+swap/shared20GiB ccache/no build
GPU devices; private N8 packaging boundary. Ten guards ACK600s/all gate+batch
dmesg identical/306 stack hashes/four AOT DBs/links retained. User application
excluded/uninspected/unmodified/untested. Same r2 bootc646d874/no agent reboot/
.105/BIOS/delete. Local kernel/artifacts/image-b045 source/wheel/sysdeps/notices/
recovery/build manifests/218 SHA-verified raw logs2,870,579B/ten guards. Baseline
WS/logs/image-b045-qualification-r3/final.json; no own jobs/root SSH closed;
disk21.468GB observed. Unknown CPU fault unresolved/distinct from VM-heap/CSR/
B04.2 queue. **Next B01.1:** review BF16 FFT/promotion/autocast before implementing
any missing explicit FP32 helper; complex64 output/input-gradient references.

**B04.4 CPU video/seek repair installed2026-10-21:** code22f796252/source
3eecace02076800cf6aa9be6364ed2f1f611548d, reviewable TorchCodec patch preserved.
Original runtime passes40 checks then scalar VFR time seek fails across packet
length gaps; exact/custom modes now use scanned next-PTS index like batch API,
approximate/packet durations unchanged. Candidate108/installed108 checks: video55,
retained audio46/core7. Nine-frame26x18 independent exact RGB FFV1/H264-RGB I/P,
layouts/index/time/range/batch/source forms, prescribed VFR PTS, encoder views/
bytes/filelike/incremental chunks versus FFmpeg CLI, refusals and actual HIP
spatial output/input gradients plus CPU encode. No GPU codec/B-frame/YUV/HDR/
all-format/custom-mapping/approximate-VFR/speed/stability claim. Image codecs off.
Image87875c40998bf083ecbc991e7dbe00b4e5549755b84d0f5b36d470167ab5de6d;
only codec wheel/new image/runtime metadata changed, existing ROCm/Torch/defaults/
B04.2 CLR repair retained. Locked rollbackWS/recovery/video-b044-before/runtime.json,
old dcd025 image/source/build/cache kept. Seven guards ACK600s incl failed gate;
passing gates/whole batches dmesg identical,306 stack hashes/four AOT DBs/links
retained. User application uninspected/unmodified/untested; owned fixtures only.
Build4 CPUs/Ninja/8GiB RAM/10GiB mem+swap/shared20GiB ccache/no GPU devices;
existing dependency image reused, private packaging N8 remains. Entry bootc646d874
same r2 during tests; no agent reboot/.105/BIOS/delete. Local artifacts/video-b044
under kernel retain source/wheel/recovery/build manifests/130 SHA-verified raw
logs2,536,706B/seven guards. BaselineWS/logs/video-b044-qualification-r3/final.json;
no own jobs/root SSH closed, disk21.589GB observed. Unknown CPU fault remains
unresolved/distinct from VM-heap/CSR/B04.2 queue and this timestamp semantic defect.
[Report/use/recovery](bc250_video_b044_results/README.md). **Next B04.5:** feasible
CPU TorchCodec JPEG/PNG/WebP image paths and independent owned pixel references.

**B04.3 CPU audio file codec installed2026-10-21:** code9c5199ccb; preserved
sources/bc250-torchcodec v0.17.0 d9dc54238e173000bc3969eb3959777026ff9ae8,
no source patches, all471 tracked files match archive. CPU FFmpeg6.1.1, CUDA/image
codecs disabled; qualified Torch2.13/HIP7.17/Python3.12/ABI1 retained. Public
load/save now available through bc250-torch/train, CPU decode/explicit HIP tensor
transfer/CPU save. Existing normalized-FP32 and legacy-option warning contracts
unchanged. Candidate136/installed136 checks: file46 (independent PCM16/24 RIFF,
FLAC, MP3 CLI comparison, views/crops/empty/metadata/file-like input, independent
saved PCM, warnings/refusals and GPU resample output/input-gradient+CPU save),
DSP34/native40/vision9/core7. Exact PCM and owned MP3 CLI match, fixed precision
limits retained. Ten600s guards ACK, each GPU gate/both passing batches dmesg
identical; no universal codec/model/video/image/GPU codec/long-stability/speed
claim. Image dcd025fbee1ffd180b9ff4f3ab8b753f2415d8a58914d77506ea12d4777a88d7;
only dev image/codec metadata changed. Existing native ROCm/Torch/kpacks/defaults
and B04.2 CLR repair retained. Old97fac image/release/builds/caches kept; recovery
WS/recovery/codec-b043-before/runtime.json, locked deploy helper --rollback.
Initial app-doc/path/arch/preflight failures retained; no failed GPU gate hidden.
Build4 Ninja/4 CPUs/8GiB RAM/10GiB mem+swap/20GiB ccache/no GPU devices; private
FFmpeg development packaging, N8 release/license disposition still separate.
**User direction:** stay out of finetune-n7; eight historical app identities
excluded without further reads.306 stack paths/four AOT DBs/links retained,
shared qualifier include_training_app=False/current stack baseline required;
no app edits/tests/checkpoint migration or renewed qualification. Boot8ea98081
observed at entry, same r2 during gates; no agent reboot/.105/BIOS/deletion.
Local kernel/artifacts/codec-b043 preserves source/wheel/proof/build manifests/
recovery/145 SHA-verified raw logs3,703,497B. BaselineWS/logs/codec-b043-qualification-r3/
final.json; no own jobs/root SSH closed; disk34.927GB observed. Historical unknown
CPU fault unresolved/distinct from VM-heap/CSR and B04.2 queue defect.
[Report/use/recovery](bc250_codec_b043_results/README.md). **Next B04.4:** pinned
CPU video APIs with owned raw-frame decode/seek/encode references/HIP transfer;
CUDA/image codec builds remain off until separately qualified.

**B04.2 torchaudio installed2026-10-21:** native2.11.0+gfx1013,
source1ae7f5d8023e82be51b500399a37fceb45481a2b (upstream34c52a67), integrationd75350905/
controllers7bf89b571. Existing patched Torch retained. ROCm build/current-stream
IIR/hipify/RNNT guard+counter patches preserved. Reproduced RNNT cold-hostcall
CPU crash traced with machine-code-matched HIP symbols to reclaimed internal
transfer queue in occupied-bitfield fill. CLR1f9609fd69c4923291f58934495520fe62e66ef8
locks/reacquires/checks transfer queue through fill; counter fix alone failed.
Candidate142/installed142/post-user-reboot48 composite checks pass: DSP34,
native40 (IIR waveform/coefficient gradients, RNNT CPU DP/enumerated cost and
logit gradients, exact CTC alignment, actual GPU/current-stream trace), retained
vision60/core7 and direct60-stream hostcall-initialization probe. No actual printf
service/all-shape/second-derivative/speed claim; TorchCodec load/save still absent.
Image97fac19f0d5747265c4ff8328f99d2eaacaf8ff62e5092d22cda300f7130a632,
HIP SHAa1162308076a99ff5cf5e4e15ca8bff53928d8bc4e624c42033d7e15dd2d1e3b.
Runtime base62ab2437c plus this CLR delta; Torch/rocBLAS/otherlibs/N3.2/math/SDMA0/
heapoff unchanged. Locked rollback restores image metadata/HIP/source together:
WS/recovery/audio-b042-before, deploy_audio_b042_on_host.py --rollback. Old image,
release/builds/caches retained; content-bound checkpoints may detect changed HIP,
no silent migration.26 guards ACK600s; passing r4/r5/r6 dmesg identical, whole
session differs due retained RNNT faults and concurrent external+own staging OOM.
No exclusive-causality claim. User confirmed rebootefd16227 -> c81853b2, same r2;
installed hashes and48 postboot checks pass. No agent reboot/.105/BIOS changes.
314 final protected paths/four FFT AOT DBs/links retained. One entry app-doc drift
preserved without trainer requalification. Local kernel/artifacts/audio-b042 has
source/bundles/patches/both wheels/native recovery/symbols/301 SHA-verified raw
logs; interrupted empty export-r1 kept, fsynced export-r2 verified. No own jobs,
root SSH closed. Disk17.55GB observed. Historical unknown CPU fault remains
unresolved/distinct from fixed VM-heap/CSR and this queue defect.
[Report/use/recovery](bc250_audio_b042_results/README.md), baseline
WS/logs/audio-b042-qualification-r6/final.json. **Next B04.3:** pinned TorchCodec
ABI/FFmpeg/CPU file load-save reference qualification before installation.

**B04.1 torchvision installed2026-10-20:** prep626e19ef2/integration220bd80e5;
local preserved sources/bc250-torchvision release/0.28 revision8fb87713a24951e639c494b0f2a8a81b5f8e33a6.
Native0.28.0+gfx1013/Pillow13 against qualified Torch2.13/HIP7.17/Python3.12/ABI1;
no custom vision operator patch or replacement Torch. Candidate/installed each67
checks: FP16/32/64 NMS/four ROI/grouped deform conv outputs/promised gradients,
independent NMS/bilinear CPU oracles, strided/empty, actual native GPU trace,
CPU PNG/JPEG/WebP+HIP transfer, four FP32 transform gradients, synthetic FP32
ResNet18 output/input/all62 parameter gradients, core7/consumer imports. Fixed
limits unchanged, maxL2 ops0.001519/transforms5.986e-7. CPU codecs explicit;
BF16 custom ops/GPU decode/video/second-order/universal-model/speed unqualified.
Independent active image a0b85fa88c9ecfbec88a8774f19db0fbc18fcfeaa086a4e402b69aa99ba55e27;
old5df0a58392de and release untouched. Only dev runtime.json image/vision metadata
changes; recoveryWS/recovery/vision-b041-before/runtime.json, locked deploy helper
--rollback. New packages in image /opt/torch-venv; existing consumers remain
read-only. Shared qualifier optional release override backed up. Two app-doc
entry drifts preserved, not renewed app qualification. Failed missing /var/tmp,
package output directory and candidate-r1 indirect libsharpyuv loader diagnostics
retained; fixed per-codec ORIGIN RUNPATH+licenses in package-r3. Candidate-r1
ops51 pass; corrected candidate-r2/installed-r3 all67 pass. Eight600s guards ACK,
each GPU batch dmesg identical.313 final hashes/four AOT DBs/links retained, only
runtimeJSON differs intentionally from entry. Native runtime62ab2437c/rocBLAS
8389197c/nativeTorchdc8fea0/source2b449c4/otherlibs319a235d6/defaults/r2 boot48bdf727
unchanged; no .105/BIOS/reboot/delete. Build4 Ninja/10GiB memory/12GiB mem+swap,
20GiB ccache explicit HIP path; automatic CPU wrapping bypassed. Source/build/
images/caches retained; SHA-verified local kernel/artifacts/vision-b041 with raw/
repaired wheels/Pillow/source archive/109 log files. Disk18.191GB free observed.
No own job/container/guard; root SSH closed. CPU fault unresolved/distinct from
fixed VM-heap/CSR/N38 OOM. [Report](bc250_vision_b041_results/README.md), raw
WS/logs/vision-b041-*, finalbaseline vision-b041-qualification-r3/final.json.
Next B04.2 compatible pinned torchaudio/backend audit, isolated build and meaningful
CPU/GPU audio signal references; broad port scope, no extra trainer restoration.

**N3.10 broad compiler coverage2026-10-20:** user explicitly prioritizes broad
ROCm/Torch/library support; Qwen/LoRA is one optional fixture, not scope boundary.
Code7c8156c57 replaces proposed Qwen-only MLP audit with synthetic residual dense,
gated/RMS, group2/dilation2 CNN+GroupNorm/GELU/pool/head and training BatchNorm.
FP32/FP16-BF16 autocast with FP32 parameters, two shapes/layouts each, fullgraph
Inductor dynamic=False.48 eager/compiled cases: CPU FP64 output/input/all-parameter
gradients, running buffers and independent CPU AdamW;12 CPU finite-difference
probes/four profiled CPU-reference replays.56 composite checks pass;52 GPU
forward/backward executions/48 optimizer updates. MaxrelativeL2 dense0.004123/
gated0.006393/CNN0.008807/BN0.004749; unchangedFP32/FP16/BF16 limits. CPU AdamW
parametererror<=1.49e-8; BN3 buffers match.24 FX graphs/314 generated-kernel metric,
101 GPU trace kernel records incl Triton; no eager fallback. No production defect
reproduced/source fix/install or speed claim. Existing is_big_gpu guard excludes
HIPmajor10 max-autotune; policy retained, not claimed physical impossibility.
Four600s guards ACK, whole-batch dmesg identical/313 protected identities/four
FFT AOT DBs/links retained. New r2 boot48bdf727 observed at entry (not agent
reboot); native runtime62ab2437c/rocBLAS8389197c1/Torchdc8fea0/source2b449c4/other
libs319a235d6 unchanged. Four compiler/CPU threads, JIT cache extended not reset;
no .105/BIOS/fallback/delete/nativebuild/prefix/default change. Disk28.976GB free
observed.56 files1,304,438B verified local kernel/artifacts/training-n310/remote;
[report](bc250_compiler_n310_results/README.md); WS/logs/training-n310-* and
baseline training-n310-qualification-r1/final.json. No own jobs/container/guard.
Historical CPU fault unresolved/separate from VM-heap/CSR/N38 host OOM.
Next B04.1 matching pinned torchvision native gfx1013 extension/CPU-GPU image
operator audit, isolated build and reference gates before installation; no other
Torch wheel. Torchaudio separate. B01/B02/B03/B05/B06/B07 and broader N2/N3/N4–8
remain real scope even when unnecessary for Qwen; no universal-support percentage.

**N3.9 trainer reconciliation2026-10-20 local/19UTC:** code9899e0659,
continuity5a24fe4e7/guidea7fa50d00/appguidanceabb5256ab. Exact reviewed patch
restores qualified N37 trainerb9bf2587/provenance/A07 attention CLI; keeps current
opsd058b14a status edit and all arithmetic. Math/N3.2 default unchanged. New
runtime-bound checkpoints enforce130 selected files/mapped native identities;
old provenance-bound ownedN37 checkpoint correctly refuses ONLY cosmetic ops
source drift before weights/output and stays SHA-identical. No silent migration.
Actual masked32x1024/rank16/accum16 fresh2/new resume/ownedN36 legacy/installed
resume exact372 adapters/full AdamW/scheduler/CPU+GPU+Python RNG/order/progress/
run.json/loss/evaluation. Installed numerical state exact preN37 after excluding
metadata marker. Short tiled rank4 fresh2/resume1 exact full state and provenance;
old math fingerprint unchanged/tiled distinct/backend change refused.31 composite
checks/83 train micros/eight eval forwards;175 app tests candidate+installed and
four provenance CPU contracts local+.104 pass. Initial174 app tests had3 stale
fixture errors (missing attention_backend), fixture fixed+one forwarding test;
no launcher runtime change. Seven600s guards ACK, dmesg identical.313 protected
paths final:309 original unchanged, trainer/guide intentionally updated; tests/
AGENTS added. App AGENTS attention paragraph synchronized with qualified opt-in
scope. BackupWS/recovery/training-n39-before incl shared validator old source;
baselineWS/logs/training-n39-guidance-r4/final.json; GPUbaselinequalification-r3.
Disk-backed owned scratchWS/experiments/training-n39-r1; evidence local
kernel/artifacts/training-n39, [report](bc250_training_n39_results/README.md).
Runtime62ab2437c/rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/otherlibs
319a235d6 retained, same r2 bootd3857b96. No native build/install/cache reset/
delete/reboot/.105/BIOS/fallback change; CPU fault unresolved/separate from fixed
VM-heap/CSR and historicalN38 OOM. No own GPU job/container/guard deadline.
Next N3.10 actual LoRA MLP compiled/eager audit after reading prior compiled-block
evidence; keep math/N3.2 defaults and full-trainer/performance claims separate.

**Historical N3.8 vocabulary audit2026-10-19:** implementationd7b7a2bc4/settlinged0946ef6/
resourcepreflightf85831f3b. Actual1024-token BF16 forward[1024,2048]@[2048,248320]
and input-backward[1024,248320]@[248320,2048]; all14 admitted solutions+default
per role pass full chunked CPU FP64/guards/immutable inputs. FP64 tensor bounds
1.916GiB/64MiB (not RSS); two2.547GB fixtures verified durable locally. Initial
private atomics0 differed from eager Torch atomics1/32MiB workspace: raw2.443x
backward ratio is NOT Torch gain. Corrected actualTorch/candidate isolated
ratios1.078x forward(-339)/1.723x backward(-533). Reduction order differs.
Experimental combined CSV only; N3.2 installed preset retained. r7 full-model
ABBA loss/all372 gradients/CPU AdamW/RNG passes, worstgradientrelativeL2=1.446%
<4%; strong timing transition prevents stable speed inference. r8 extra-settling
fails global host OOM/SIGKILL with concurrent UID1000 train; retain failed log,
not CPU segfault or VM-heap corruption. User reserves board for r9; >8GiB RAM/
no competing GPU clients preflight and two extra full untimed steps per worker.
r9 passes: ABBA reference5.809/5.435s versuscandidate5.352/5.352s; ratios
1.085/1.015x still variable, no new default. Firstloss exact/all372 gradients
worst1.446%/CPU AdamW/RNG pass; finalreference gradients exact.14 composite
checks/41 completed train micros; eight600s guards ACK, successful gates dmesg
identical, whole batch differs dueOOM. [Report](bc250_training_n38_results/README.md).
Entry againstN37 has exactly two drifts: current replica/train.py removes N37
provenance/A07 tiled CLI, bc250_ops.py only changes printed status. Current app
preserved, all309 other protected identities match; isolated N38 model uses
byte-exact old qualified hookf0bcb35e. Thus N37 historical deployment below is
NOT current trainer feature qualification at that entry; N3.9 above restores it. EntrybaselineWS/logs/training-n38-
entry-r2/final.json is a preservation snapshot, not application qualification.
Next reconcile trainer features, then owned CLI fresh/resume before default
preset decision; do not silently migrate new-provenance checkpoints. Native
runtime/prefix/N3.2 defaults/kernel/.105 unchanged. RawWS/logs/training-n38-*;
local kernel/artifacts/training-n38. CPU fault remains unresolved/distinct.

**Historical N3.7 checkpoint provenance2026-10-19:** codee55d8b23a/app0f2dfa078 deployed.
Supplemental JSON digest binds run.json/training.pt without changing fingerprint.
130 selected file identities incl110 GEMM assets/15 mapped DSOs/preset/hooks/
selected env/kernel/version metadata. New invalid/changed identity refuses before
weights/output; legacy still resumes with explicit continuity-unverified warning
and gains current provenance only on NEW saves. Preflight structurally validates
sidecar/binding, defers live comparison. Actual1024/rank16/accum16 masked data:
fresh2/new resume/owned N36 legacy/installed resume exact372 adapters/full state/
loss/eval/progress; installed numeric state exact N36 excluding only metadata.
80 train micros/five eval forwards,26 qualified composite checks/four CPU tests
local+.104. Original8 refusal assertions pass but r1 guard fails two new network
XFRM/bridge kernel messages; failure retained, no GPU fault in delta. r2 preserves
three passed training gates/source identities, repeats only refusals requiring
byte-identical dmesg; r3 guarded install/replay passes. Six guards ACK600s;
whole-batch log is NOT identical.308 original paths unchanged, trainer intentionally
updated, module/guide newly pinned;311 final hashes/four FFT AOT DBs match. Backup
WS/recovery/training-n37-before-r3; baseline WS/logs/training-n37-qualification-r3/
final.json; [report](bc250_training_n37_results/README.md). Same native revisions/
r2 bootd3857b96; no native build/install/reset/delete/reboot/.105/BIOS/fallback
edit. No own jobs. Disk~5.1GiB→~1GiB observed; RAM scratch owned regular files and
links verified durable local kernel/artifacts/training-n37, priorN35/36 preserved.
CPU fault unresolved/distinct from VM-heap/CSR/N3.2 loader/log guard failure.
Selected eager metadata, no complete image/security/determinism/convergence/speed
claim. Next N3.8 vocab forward/backward actual-layout audit and chunked CPU-double
reference design under bounded memory; keep N3.2/model acceptance unchanged.

**N3.6 actual trainer CLI2026-10-19:** codebee5c6ce9. Deployed inner CLI on
current N3.2 preset, owned32 exactly1024-token records/32 distinct masked-target
counts72–289/rank16/alpha32/BF16/FP32 adapters/math/checkpoint/accum16. Two fresh
updates plus checkpoint1→2 separate process: exact372 adapters/full AdamW/
scheduler/CPU+GPU+Python RNG/shuffled order/progress/run.json/training loss/
token counts/LR; evaluation2 exact.48 train micros/three eval forwards,7 checks;
two guards ACK600s/dmesg identical/309 hashes/four FFT AOT DBs unchanged. No
production app/native/runtime change, build/install/cache reset/delete/reboot/
.105/BIOS/fallback edit. Same r2 boot d3857b96, disk~5.1GiB. Volatile
/tmp/bc250-training-n36-r1 exported/SHA-verified durable local
kernel/artifacts/training-n36/scratch-r1; priorN35 preserved. No active own jobs.
[Report](bc250_training_n36_results/README.md); baseline
WS/logs/training-n36-qualification-r1/final.json. CPU fault remains unresolved/
distinct from fixed VM-heap/CSR and N3.2 loader. No real-data/convergence/speed/
mid-accum/CPU-model/cross-runtime promise. Fingerprint pins data/model/args/
package versions, omits numerical runtime/preset/hook content; qualifier pins
these separately. Next N3.7 compatible supplemental provenance for new
checkpoints, owned same-runtime/changed-runtime tests, explicit legacy behavior
preserved. Vocabulary-projection tuning remains open afterward.

**N3.5 accumulation16/replay2026-10-19:** code986093a3c. Current-preset actual
Qwen3.5 synthetic1024/rank16/BF16/math/checkpoint, FP32 adapters:372 GPU accumulated
gradients versus independent CPU FP64 sum16 pass fixed2e-6 bound, worst1.17090e-8.
Exact losses/IDs/three RNG reference replay; independent clip, two CPU AdamW/
scheduler updates maxparamabs1.863e-9/3.725e-9. Fresh optimizer-boundary checkpoint
step2 matches uninterrupted372 parameters/gradients/full optimizer/scheduler/
losses/inputs/progress/CPU+GPU+Python RNG exactly. No actual CLI/real-label/
mid-accumulation/CPU-model/convergence/throughput claim. Mount overwritten by
consumer setup r1 fixed; r2 CPU oracle aliases checkpoint step counter on load,
clone/nonalias assertion fixed. r3 reuses passed first48 microbatches and immutable
artifacts, correct fresh16/exact comparison;80 total incl failed-oracle16. Eight
composite checks pass, three guards ACK600s, whole-batch dmesg identical,309 hashes/
four FFT AOT DBs/links/runtime/user files unchanged. Failures/source retained.
Entry NEW r2 boot d3857b96-1b51-4e39-8f3e-f076f0c0422a; agent did not reboot.
Only881MiB disk: no deletes/reserve change. /tmp tmpfs preflight>2GiB scratch/
>6GiB RAM; explicit N35-only512MiB text-log floor, other gates keep2GiB. ~836MiB
artifacts volatile remote /tmp/bc250-training-n35-r2, durable copied/SHA-verified
local kernel/artifacts/training-n35/scratch-r2. Logs WS/logs/training-n35-*, local
kernel/artifacts/training-n35/remote; [report](bc250_training_n35_results/README.md).
Next baseline WS/logs/training-n35-qualification-r3/final.json; helper backup
WS/recovery/training-n35-before. Runtime62ab2437c/rocBLAS8389197c1/Torchdc8fea0/
source2b449c4/other libs319a235d6 unchanged; no native build/install/persistent cache
reset/.105/BIOS/fallback change, no own job/container/guard, disk~880MiB. Historical
CPU fault unresolved/distinct from fixed VM-heap/CSR/repaired N3.2 loader/oracle.
Next N3.6 actual deployed CLI owned1024/r16/acc16 resume/fingerprint qualification;
older42-token/rank4 CLI evidence retained, user files untouched.

**N3.4 square projection audit2026-10-19:** code6531025ef, no native/runtime
change. Actual BF16 [1024,2048]@[2048,2048] TN84/NN42 mm calls, two layouts,
all14 admitted solutions each+default CPU-double/guards/input preservation pass
fixedrelativeL2<0.004, worst0.001702264. Initial fixture1.34234x TN-346/1.07391x
NN-533; TN timing regime shifts retained. Full-model TN-346 test keeps installed
N3.2 NT-490 baseline active in both workers: worst372-adapter gradientrelativeL2
0.0765128583>unchanged0.04, candidate REJECTED. Lossdelta0.00033474 passes existing
forward-model0.015 contract; each own-gradient CPU AdamW replay maxabs1.863e-9,
RNG hashes equal. One warm median pair5.625697/5.581987s; ABBA stops after failed
first candidate, no qualified model gain. No tolerance relaxation/production
arithmetic defect claim. Follow-up14 TN exact-output audit: -343/-338/-345/-350
match captured default; sustained2s warmup/six-round ratios0.69190/0.99441/0.99386/
0.72009, no gain. Other solutions change624–639 of2097152 outputs; -346638 at
relativeL2=5.45847e-5. No square override deployed; experimental CSV stays inactive.
Four guards ACK600s, three GPU passes/one candidate rejection; dmesg identical,
309 identities/four FFT AOT DBs/links/runtime/user trainer/config unchanged.
Runtime62ab2437c/rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4/other native libs
319a235d6 remain. No build/install/cache reset/reboot/.105/BIOS/default/fallback
change; same r2 boot, no active job/container/guard, free~5.34GiB. Historical CPU
fault unresolved/separate from fixed VM-heap/CSR/repaired N3.2 loader/rejection.
[Report](bc250_training_n34_results/README.md); raw fixtures/gradients/errors
WS/logs/training-n34-* and local ignored kernel/artifacts/training-n34/remote
copied SHA verified. Helper backup WS/recovery/training-n34-before; runtime
recovery WS/recovery/gemm-n32-before. Next baseline
WS/logs/training-n34-qualification-r3/final.json. Next N3.5 current-preset1024/rank16
accumulation16/update/owned fresh-process checkpoint resume; older42-token/rank4
CLI evidence retained. Large vocab tuning still open, requires memory-bounded design.

**N3.3 optimized-model profile / reverse GEMM audit2026-10-19:** code7838d94bc;
installed runtime still62ab2437c/rocBLAS8389197c1/nativeTorchdc8fea0/source2b449c4,
other native libraries319a235d6. Current automatic preset active without preload.
Actual-base synthetic1024/batch1/accum1/BF16/rank16/math/checkpoint replay exact
loss/372 nonzero clipped gradients/update/AdamW/RNG;36 conv/36 gated-delta calls,
three finite warm updates median5.442552s/peak5351892480B.38296 kernels sum5.403136s,
mm75.387%; reverse [1024,2048]@[2048,6144]26.427%, prior shape15.423%, square7.105%,
vocab6.884%/4.333%, frozen conv7.529%. No new speed improvement claim. Reverse
actual TN132/NN24 mm calls, two layouts, all14 admitted solutions each+default
CPU-double/guards/immutable inputs pass unchanged tolerances, worstrelativeL2
0.001666206. Best-339/-533 ratios1.09229/1.09099 across six alternating rounds;
full-model benefit not demonstrated, no deployment/new preset. Generic shape/
audit-label helpers added, four analyzer contracts/Black/Ruff-F/syntax pass.
r1/r2 three GPU processes pass, ACK600s/dmesg unchanged; r2 controller alone
expected3 layouts but actual2. Fixed derived count; r3/r4 seal saved gates with
unchanged original helpers/protected identities, no GPU repeat; r4 keeps inherited
user-file hashes. Failure/source retained, no arithmetic defect/tolerance change.
Final309 protected files/four FFT AOT DBs/links/runtime/trainer/config unchanged.
No build/install/cache reset/reboot/.105/BIOS/default/math/SDMA/heap/fallback change;
same r2 boot, no running job/container/guard, free~5.92GiB. Historical CPU fault
unresolved/separate from fixed VM-heap/CSR and repaired N3.2 cold loader.
[Report](bc250_training_n33_results/README.md), full trace/fixtures WS/logs/training-n33-*
and local ignored kernel/artifacts/training-n33/remote, copied SHA verified.
Next baseline WS/logs/training-n33-qualification-r4/final.json; helper backups
WS/recovery/training-n33-before, runtime recovery still WS/recovery/gemm-n32-before.
Next N3.4 square2048 projection actual layout/solution audit; full-model gates
before any new demonstrated candidate deployment.

**N3.2 exact-layout BF16 tuning installed2026-10-19:** TheRock code62ab2437c,
rocBLAS/Tensile8389197c1/patch0033. Existing gfx1013 scalar solutions audited against
actual strides: TN48/NN48/NT18 mm calls, all14/14/8 admitted candidates pass CPU
float64 full reference/guards/input preservation. Only rocBLAS NT2048x1024x6144,
lda2048/ldb1024/ldc2048, batch1/BF16output/FP32compute/beta0 selects-490; six-round
fixture173.797->4.841ms. Other layouts keep default. Old loader cold override
internal error reproduced twice; preload metadata lacked codeObjectFilename.
Lazy hydration when missing/empty fixes cold public Torch use. Cached targeted
rocblas r2 build, ccache20GiB/OMP4/max16/nativegfx1013/no GPU/network, active dist
read-only; first compile/controller errors retained. No other native DSO changed.
Actual-base synthetic1024/rank16/BF16/math/checkpoint ABBA model loss/RNG exact,
372 clipped gradients worstrelativeL2=2.030649%\<unchanged4%; every worker CPU
AdamW update/state pass, maxparamabs1.863e-9. Warm whole-step ratios1.522/1.557x;
installed5.507s/1.562x. No bitwise gradient/continuation or sustained/convergence
claim. Reference repetition exact. New launcher automatic only current mutable
prefix, exact-problem CSV+DSO+110 gfx1013 assets pinned; mismatch fallback,
manual override precedence, outer BC250_GEMM_TUNING=0 restores old path. Frozen
previous unchanged. Installed cold3/core7/conv30/model4/optout2 pass,12 guards
ACK600s including2 reproduced failures, dmesg identical whole batch. Exclusive
lock encloses install/gates; old DSO/runtime/launchers/source retained in
WS/recovery/gemm-n32-before, unchanged kernel pack hash-pinned. Final309 protected
paths vs193 prior, only active rocBLAS aliases/runtime.json intentional changes;
user trainer/config/checkpoints/four FFT AOT DBs/other libraries intact. Native
Torchdc8fea0/source2b449c4, other native libraries319a235d6; only rocBLAS rebuilt
at8389197c1. Same r2 boot/no reboot, .105/BIOS/boot fallbacks/math/rocBLAS backend/
SDMA0/heap-off retained. Historical CPU fault unresolved/separate from repaired
loader and fixed VM-heap/CSR. [Report](bc250_gemm_n32_results/README.md),
[usage](bc250_torch_usage.md); raw WS/logs/gemm-n32-\*, full local ignored
kernel/artifacts/gemm-n32/remote (copied SHA verified); candidate
WS/candidates/gemm-n32-r2. SOURCE_GEMM_N32.json in WS/current; next baseline
WS/logs/gemm-n32-deployment-r1/final.json. No active job/container/guard, free~6.13GiB.
Next N3.3 reprofile optimized model/exact capture replay and next measured layout.

**Practical N2 review / N3.1 current-model profiling2026-10-19**, TheRock code
`4e76c2f01`: [review](bc250_n2_practical_review.md) classifies residual capabilities
rather than adding an exhaustive N2.19 test sweep. No recorded mandatory N2
implementation blocking current LoRA found; support-matrix stale N2.14 cursor and
fixed-A01 group1 boundary corrected. Optional B01/B02/B07, CK backward/RPP and
broader scientific scopes remain open. Historical CPU fault remains unresolved.
Actual Qwen3.5-2B base, owned nonzero rank16 adapters, BF16/autocast/math/checkpoint,
synthetic batch1/accumulation1:65/1024 exact profiled/unprofiled loss,372 nonzero
clipped gradients, adapter update/AdamW state/CPU+GPU RNG.36 conv/36 gated-delta
calls, three warm unprofiled steps per shape. Current installed app hook read-only;
no user training data/checkpoint access, source/library/default change or build.
Offline kernel-category/launch-correlation accounting fixes new collector's
annotation double counting/worker-thread attribution;4 contract tests pass.
Initial r1 numeric/profile pass retained; r2 controller NameError before GPU fixed;
r3 final1024/r4 short65 pass. No A09 runtime defect/repair claim in this batch.
At1024 median8.676s/peak5351892480B,38314 kernels, summed8.680s; mm84.71%, backward
submitted76.82%; BF16 [1024,6144]@[6144,2048]45.94%, reverse16.53%.65 median0.726s/
peak4232892416B,17286 kernels. Synthetic traces are not sustained user throughput
or an improvement. [Report/source/raw trace hashes](bc250_training_n31_results/README.md).
Three GPU guards ACK600s, dmesg identical across all attempts,193 protected
identities/four AOT DBs/user trainer/config/links/runtime unchanged. No native
build/cache reset/reboot/boot-policy/.105/BIOS/SDMA0/heap-off/math/rocBLAS change.
Same r2 boot d0532075-88fb-4a2f-974e-f61578d20d9f; free~6.85GiB, no active agent
job/container/guard. Native Torchdc8fea0/source2b449c4/libraries319a235d6/source-doc
c29021b40e7. Raw WS/logs/training-n31-*, full local ignored kernel/artifacts/training-n31.
Next preservation baseline WS/logs/training-n31-qualification-r4/final.json.
Next N3.2 audit dominant BF16 GEMM selection/strides, CPU/qualified-GPU reference
and paired timing of one scalar candidate, then model gradients before any
reversible deployment. Do not infer universal backward/performance qualification.

**Current Qwen3.5 training follow-up2026-10-16**, implementation690042b55:
application-scoped native scalar Triton frozen depthwise causal conv K<=4,
FP32/FP16/BF16 forward/dx/SiLU; unsupported/trainable weights keep original.
Chunk64 FP32 causal loss recomputes backward probabilities; full BF16 logits remain.
Enabled by default only in .104 finetune-n7 replica/train.py, Transformers5.19.0,
math SDPA/head256. New51 operator/reference/guard groups,4 small LoRA pairs and
real Qwen3.5-2B BF16 rank16 geometry pass:372 adapter gradients max relativeL2~1.12%
at65tokens,1024-token checkpointed backward/update. Isolated1024/vocab248320 loss
extra peak2.84->0.65GiB, not whole-model savings/speed/convergence. PEFT shortened
target names exposed a separate resume bug; compare exact resolved modules now.
Synthetic deployed CLI fresh/resume gives372 exact adapter tensors plus exact
optimizer/scheduler/RNG/progress;4 mismatches refused. Final logs have no causal
fallback/MIOpen/OOM warnings; FLA reference warning retained. Original user log
continued through251/saved250: allocator retries were not a proven fatal OOM.
All final guards ACKed/exit0/dmesg identical; failed initial resume retained.
Sources/patch/local caller copies, hashes and rollback:
[report](bc250_qwen35_training_results/README.md), SOURCE_QWEN35_TRAINING_OPS.json,
WS/recovery/qwen35-training-before. No native rebuild/DSO/prefix/boot change or
active agent job. Global MIOpen timing/workspace and Python CPU fault remain open,
distinct from fixed VM-heap/CSR corruption. Exact next: profile reference gated-
delta forward/backward and choose one bounded native gfx1013 substage; N2.7 queued.

Current user-priorities1–4 follow-up2026-10-15, Torch4ee65717d69e9f51021272e3f37d0f342f68fc08:
long head128/GQA references through4096tokens and real Qwen3-1.7B BF16 rank4 LoRA
512/1024-token synthetic reference passes completed. Causal tile bounds/tile32
selection improve paired BF16 head128 kernel fixtures1.67–2.05x versus the original;
no broad/whole-model speed claim, math default retained. Qwen1024 whole-model
allocated peak unchanged5,818,112,000 bytes: other allocations dominate. Autocast
mixed-QKV conversion and gradients qualified; tiled compile/graphs/dropout remain open.

Small full-parameter Qwen FP32/FP16/BF16 AMP, larger batch2/length128–256, CPU
float64 gradients,2updates and fresh-process model/optimizer/scheduler/scaler and
all RNG continuation passed. Six dynamic compiled/eager math-attention gradient
cases at batch2/3 passed. Native TF32 capability API fixes a false Tensor Core
warning. Actual user LoRA CLI resume defect reproduced (GPU RNG ByteTensor) and
fixed, with is_trainable=True plus scheduler/Python/shuffle state persistence;
new-format and legacy fixture resumes update adapters. Legacy missing state
cannot provide exact replay; matching schedule arguments still matter.

Python CPU IP0x3b3e6 remains **unresolved**. Recovered single historical event,
no original core/stack;8 fresh allocator-instrumented solver/FFT/sparse/GEMM
processes pass1920 references without reproduction. Launcher defaults Python
faulthandler on (user override allowed); diagnostic-only bounded core capture.
Separate retained failure: full mixed-operator attempt stops at MIOpen conv
EvaluateInvokers elapsed<=0 and workspace480/provided0, exit1, not SIGSEGV.
No false full55-operator pass or crash-repair claim. All final scoped gates
exit0/dmesg unchanged/guards ACKed. Next isolate this convolution failure and
capture crash evidence; N2/K6 deferred by user. Sources/builds/cache/user files/
checkpoints/BIOS/kernel/governor/.105 preserved. [Full report and replay](bc250_training_expansion_results/README.md),
SOURCE_TRAINING_EXPANSION and WS/recovery/*-before retain state/rollback identities.

Historical first opt-in attention batch (superseded within the scopes above):

Opt-in scalar tiled SDPA2026-10-15, Torch`bda0831ba49ef9d0ba2968a800642e5abaec5dd5`:
three native Triton scalar kernels implement tiled online softmax and recomputed
backward with FP32 accumulation and grouped KV gradients, without full score/
probability storage. Built-in flash/efficient/CK flags stay off; ordinary functional
SDPA remains math. Explicit Transformer attention+mask registry helper and user
finetune-n7 --attention-backend tiled provide opt-in access. D<=128, sequence<=4096,
finite FP32/FP16/BF16, broadcast boolean/additive masks, no dropout/mask gradients/
higher-order/compiled/graph qualification. Dense masks still cost their input storage.

18 independent CPU-float64/GPU-math output+gradient cases pass; small padded FP32
Llama/checkpoint gradients pass. Installed7-group gate adds BF16 Qwen1.7B head128/
16-to8GQA at65tokens, two updates of a **small random** Qwen3 BF16 LoRA model with
CPU-float64 references and cached decode, streams and diagnostics. Unchanged eager
Torch19-reference regression passes; all final gates exit0/dmesg unchanged/guards
ACKed. Extra allocated FP32 B1/H2/D32 forward+backward peak at2048tokens2.52MiB vs
math129.50MiB; tiled slower in measured single-event fixtures. No full Qwen1.7B,
sustained stability, universal memory or speedup claim. Next: longer head128 and
representative full-model training/memory/time qualification before default use.

Exact warning bug repaired: !debug was passed as text to TORCH_WARN_ONCE, printing
1Torch during successful fallback. Normal selection now quiet; explicit diagnostic
checks still explain unavailable built-in kernels. User train/chat use new
nn.attention.sdpa_kernel(MATH), backward included for checkpoint recomputation;
broad warning filters removed. Math stays default. Targeted12-job/20GiB-ccache
rebuild linked SDP-utils plus regenerated HIPHooks; loaded libtorch_hip SHA578a3583...
Exclusive lock used for changes/tests. Existing source/build/cache/images/checkpoints,
BIOS/kernel/governor/.105 preserved. Recovery WS/recovery/sdpa-{probe,callers,metadata}-before,
SOURCE_SDPA metadata, and [full report/replay/evidence](bc250_attention_results/README.md).
User training/chat finished before tests; no agent jobs remain. Earlier harness/
reference/staging failures retained; unknown Python CPU segfault still separate.

[Training evidence](bc250_training_memory.md), root `d744f0952`, proves real
SmolLM2-135M full FP32 fine-tuning and FP16/BF16 AMP LoRA for a bounded example.
Model revision `93efa2f097d58c2a74874c7e644dbc9b0cee75a2`, weights SHA256
`80521b40281d6ce74e35c9282c22539e75aa0ac8578892b2a59955ef78d55da1`;
134,515,008 base parameters. LoRA rank 8 / alpha 16 / q_proj+v_proj gives 460,800
trainable parameters. Batch 1 / 32 tokens / one text, mathematical SDPA and AdamW:
four updates plus one fresh-container resume update. This is not a convergence
study or proof of larger-model/full-model-AMP/QLoRA support.

Full FP32 loss fell 4.647392→2.845694→2.507597, peak~2.92GB. LoRA loss fell
~4.647392→4.0175→3.803, peak~704MB; frozen base remained unchanged, activation
checkpointing used, FP16 scaler skipped no updates. Resume preserved
model/adapter, optimizer, scheduler, scaler and CPU/GPU RNG; logits and exact RNG
probes were checked. CPU/GPU logit relative L2 was5.56e−6; representative gradient
relative L2 was8.36e−5–4.68e−4. Near-zero elementwise reference failures were
retained and replaced with justified aggregate/max-error checks, not ignored.

Memory/concurrency gates passed 10 GiB GPU pressure + 768 MiB CPU storage, fragmented
reuse, four streams, eight basic graph replays and two GPU processes in 123.99 s.
This respects shared memory; near-OOM/hang recovery was not established.
No CPU/RAM cgroup cap is imposed by the normal launcher; its default 1 GiB shared
memory mount is separately configurable. GPU use still competes with host RAM.

Retain `WS/{training-model,training-env-stable}`, validation configs/reports and
`WS/logs/training-memory/`. Obsolete generated validation checkpoint weights/state
and duplicate tokenizer files were retired20261016; regenerate synthetic fixtures
under new helper labels when needed. Genuine user checkpoints were untouched.
Qualified consumer pins are Transformers 5.2.0,
PEFT 0.18.1, Accelerate 1.12.0. `bc250-train` mounts the frozen consumer bundle
through `current/python-consumers`; it does not replace our custom Torch.
Do not redownload or recreate these because conversation context was compacted.

P1 Torch evidence:64 checks, including 52 eager/compiled dynamic-shape indexing,
normalization, training andmath SDPA cases, plus 12 explicit Lt cases with 36
observed actual successful library calls. LD_PRELOAD tracing is test-only and
must resolve RTLD_LOCAL dependencies; never enable it in the ordinary launcher.
Additional prior eager 56/compiler 18/operator 55 checks remain historical scopes.

Launcher `build_tools/bc250/bc250_torch.py` handles Python/script/-c/-m/stdin/shell,
retains host HOME/cwd/absolute paths, mounts script parents, supports explicit
mount modes and private environment handoff across doas. It runs rootful Podman
as uid/gid 1000 plus device groups. Arbitrary data outside automatic mounts needs
an explicit mount; do not claim an unrestricted host filesystem. Thirteen IO
scenarios include persistent packages, HTTP downloads, DataLoader spawn,
interruption handlers and checkpoint resume. Ctrl-C/termination checkpoints
require the user's application to implement a handler.

Current P2 launcher source added a shared runtime lock; cooperating host updates
must use `helpers/update_dev_prefix.py -- COMMAND...` with the exclusive lock.
This prevents overlap of launcher workloads and prefix replacement, not arbitrary
manual writes. `/usr/local/bin/bc250-torch.before-priority2` and recovery launcher
copies are retained. Metadata `/opt/bc250-stack/launcher.json` predates the P2
lock change; installed-script hash plus committed source and P2 launcher report
are newer evidence. Do not overwrite historical metadata to fake provenance.

N1.1 .104 gate: `WS/logs/n11-hiptensor-r3/result.json`, root report
`bc250_scalar_library_results/n11-hiptensor.json`. Old library rejected the new
case; targeted12-job/20GiB-ccache build17.1s and GPU gate2.7s passed under the
exclusive runtime lock. Kernel logs identical (1009 lines). Thirty-repeat HIP
stream-event timings across seven types: A-only12.6–13.5µs, mixed27.7–29.3µs;
includes submission gaps, no previous implementation/speedup comparison.
Independent prior library: `WS/recovery/n11-hiptensor-before.so.0.1`.
Two initial container failures (`WS/logs/n11-hiptensor{,-r2}`) happened before
compilation: cgroups were unmounted after a host reboot, and runc rejected
NoCgroups. Existing cgroups service started in unified mode and enabled in boot;
no kernel change or reboot, SSH/fallback intact. Do not use disabled cgroups
with the installed runc. N1.1 did not rebuild Torch or alter .105/frozen candidates.

N1.2 .104 gate: `WS/logs/n12-hiptensor-r2/result.json`, root reports
`bc250_scalar_library_results/n12-{hiptensor,reconstruction}.json`. C32F/C64F
scalar path uses HIP complex addition/multiplication, shared geometry/guards;
public host scalar ABI and FP64 capability check retained.450 complex cases
(426 std::complex<double> references,24 descriptor rejections), then273 real
regressions passed. All27 identity/NEG/CONJ A/B/C combinations executed on seven
layouts/two types; mixed/unilateral sums, guards, signed-zero/NaN and imaginary
scalars, null zero-beta C, in-place conjugated C/D,16-mode union with unit axes,
destroyed source descriptors/preferences, nine algorithm/workspace pairs/type.
First17.1s build passed; gate stopped on test-only false alias expectation for
adjacent scalar C/D tensors (`WS/logs/n12-hiptensor`, preserved attempt1 report).
Prior active library restored before releasing lock. Fixed assertion checks
multi-element spans; force source timestamps on retry to regenerate staging
rather than retain the restored old dist hardlink. Cached restage4.9s,
complex2.3s/real2.8s gates passed; kernel logs identical1009 lines.12jobs/20GiB
ccache, targeted hipTensor+dist/exclusive lock. Thirty-repeat tiny stream-event
timings: GEMM8.7–9.9µs, mixed21.5–23.2µs including submission gaps; no speedup
or large-performance inference. Independent prior library:
`WS/recovery/n12-hiptensor-before.so.0.1`. No new host/kernel/Torch changes.

N1.3 plan contracts: common plan APIs validate pointers, enums, exact attribute
sizes and workspace preferences before copying/allocating; failed setters/query
leave state/output unchanged, failed plan creation returns null with owning
cleanup. Native scalar defaults use cache NONE, JIT NONE, autotune NONE, count0,
rank0; PEDANTIC/DEFAULT-JIT/INCREMENTAL/positive count or rank return NOT_SUPPORTED.
Three algorithms remain deterministic aliases, not new performance models.
`WS/logs/n13-hiptensor-r2/result.json`:304 API checks +12 CPU-referenced GPU
executions (R32F/R64F/C32F/C64F ×3algorithms; limits0/1/UINT64_MAX); mutated/destroyed
source preferences/descriptors do not invalidate existing plans.273 real/450
complex regressions passed, kernel logs identical1009 lines. Initial17.6s build;
first probe passed316 but runner's count assertion failed (API checks were not
printed individually), preserved in attempt1 report and `WS/logs/n13-hiptensor`;
prior prefix restored before unlock. Reporting fixed, production code unchanged;
cached restage4.9s, plan1.5s/real2.7s/complex2.3s gates under exclusive lock,
12jobs/20GiB ccache. Prior library `WS/recovery/n13-hiptensor-before.so.0.1`.
Common validation changed, but non-gfx1013 numerical execution and every internal
initialization-failure injection are not qualified. No new performance claim.

N1.4 .104 `WS/logs/n14-sparselt/result.json`, repository
`bc250_scalar_library_results/n14-{sparselt,reconstruction}.json`:604 new numerical
broadcast cases +5 descriptor rejections,234 prior sparse regressions passed.
Native API now admits zero batch stride for A/B/C with matching logical counts;
upstream `rocsparselt_auxiliary.cpp` requires matching counts and compress/prune
sources collapse stride0 to one physical matrix. Native compression size/indexing
and strip/TILE launch counts now follow that physical layout; no duplicate sparse
writers. Multi-batch D stride0/incompatible counts remain rejected. Span checks
reject partial prune/compress and matmul input/D or C/D overlap; exact in-place
pruning remains valid. All7 masks ×sparse A/B×orders×transposes×five type/prune
combinations, selected batch1/stride0 and partial tiles, six activation epilogues;
CPU double references, independently decoded pack, plan/descriptor compression
agreement, pack/D guards and poisoned original sparse buffers. Targeted16-job
hipSPARSELt+dist build6.9s, new gate3.1s/prior1.6s under exclusive lock;20GiB ccache.
Kernel logs identical1009 lines; no performance claim. `SOURCE_N14.json` verifies8
changed files;21 patches reconstruct exact tree. Prior library retained at
`WS/recovery/n14-sparselt-before.so.0.1` (backup name; original runtime file0.2).
No host/kernel/Torch/hipTensor rebuild or .105 changes in this batch.

N1.5 .104 `WS/logs/n15-sparselt-r2/result.json`, repository
`bc250_scalar_library_results/n15-{sparselt,reconstruction}.json`:526 independent
CPU-double-referenced native GPU bias cases +41 API checks;234 scalar and609
broadcast regressions passed. Per-output-row bias is added before activation;
stride0 shares the vector, positive stride>=D rows allows padded per-batch bias.
Default type matches the matrix; f16/bf16 may also use FP32. Public pointer/stride
comments are swapped: upstream memcpy pointer-size/int64 stride and row mapping
are authoritative. Null pointer disables bias. Exact sizes, admissible types,
alignment, stride cap and D/bias span overlap guarded; failed setters/getters
preserve state/output. Plan captures pointer/type/stride while device data stays
caller-owned. Source bias settings mutated after plan creation; typed/FP32 bias,
orders/transposes, sparse A/B, strip/TILE/partial tiles, six activations,
broadcast inputs, poisoned zero-alpha/beta and valid in-place C/D passed with
unchanged tolerances and immutable bias/pack/D guards. First6.6s build and probe
passed567 checks but stdout buffering mixed PASS/stderr records (565 parseable);
runner restored prefix before unlocking. `WS/logs/n15-sparselt`/attempt1 report
retained; line-buffered reporting fixed accounting, production code unchanged.
Cached restage2.8s, bias2.7s/scalar1.6s/broadcast3.0s gates,16jobs/20GiB ccache,
targeted hipSPARSELt+dist/exclusive lock; kernel logs identical1009 lines.
`SOURCE_N15.json` verifies8 changed files,22 patches reconstruct exact tree;
prior library `WS/recovery/n15-sparselt-before.so.0.2`. No performance, new
matrix-type/vector-scaling or long-stability claim; .105/kernel/Torch unchanged.

N1.6 `WS/logs/n16-sparselt-r5/result.json`, root
`bc250_scalar_library_results/n16-{sparselt,reconstruction}.json`:404 native GPU
alpha-vector numerical cases +34 API checks;234 scalar/609 broadcast/567 bias
regressions passed. Public flag int32 0/1, captured plan mode; scalar alpha stays
host FP32, enabled alpha argument becomes borrowed device FP32 vector[M] indexed
by logical output row and shared across batches. No host vector dereference.
Upstream `clients/include/spmm/testing_spmm.hpp` allocates M device elements and
passes the vector as alpha; `tensile_host.cpp` supplies it as scaleAlphaVec with
scalar multiplier1. Beta enum has no pinned upstream handler/client evidence:
scalar beta preserved, getter/disable0 supported, enable1 NOT_SUPPORTED. A native
beta-vector extension is still unimplemented, not a hardware impossibility.
Flags/sizes/null buffers/alignment and D/vector exact/partial spans guarded;
source mode mutation after plan creation, mixed signed-zero/negative/positive
rows, poisoned all-zero alpha vectors and zero-beta C, layouts/transposes/types,
pruning/compression, bias/activations and compatible in-place C/D passed.
Initial/r2/r3 CPU reference mismatches were solely GPU+0/reference-0 from unused
zero-alpha multiplication; r3 repeated after a failed local edit helper. r4
reached zero-beta cases and found GPU-0/reference+0 from adding unused beta*C.
Hex diagnostics/failed reports retained in `n16-sparselt-attempts.json` and host
logs. Reference skips both unused terms to honor zero-scalar semantics; strict
bitwise/padding comparisons and nonlinear tolerances retained. Production GPU
source/library hash unchanged across attempts; every failed gate restored N1.5
under the exclusive lock. Do not repeat the diagnosed reference defect.
Initial6.6s build, successful cached restage2.8s, vector2.2s/scalar1.7s/broadcast3.0s/
bias2.7s gates,16jobs/20GiB ccache. Kernel logs identical1009 lines.
`SOURCE_N16.json` verifies8 changed files,23 patches reconstruct exact tree;
prior library `WS/recovery/n16-sparselt-before.so.0.2`. No general performance,
new-type or long-stability qualification; .105/kernel/Torch unchanged.

N1.7 `WS/logs/n17-sparselt-r4/result.json`, repository
`bc250_scalar_library_results/n17-{sparselt,reconstruction}.json`:134 native GPU
INT8 numerical cases/16 API checks and234/609/567/438 scalar/broadcast/bias/vector
regressions. Pinned upstream type table admits INT8 inputs with compute_i32;
`spmm/rocsparselt_spmm.cpp` still instantiates FP32 scaling arguments. Integer
bias defaults FP32; CPU GEMM/bias/activation clients round with nearbyint then
saturate INT8. Native backend now implements signed INT8 input/output only, INT32
retained-product accumulation, FP32 scaling/bias and nearest-even saturation.
2:4 retains K/2 products, each <=16384 in magnitude: K<=262140 guarantees all
partial sums fit INT32. Descriptor rejects K262144; K262140 GPU execution passed.
K4096/-127 products and tailored FP32 scale/bias expose loss from incremental
FP32 accumulation (>2^24), checking exact integer sum then one FP32 conversion.
Signed half ties, ±128/127 extrema, saturation, orders/transposes, sparse A/B,
padded/broadcast inputs, strip/TILE/partial tiles, row alpha vector, FP32 bias,
RELU/ABS/leakyRELU/GELU, zero scalars, snapshot/alias/in-place/immutable guards and
independent pack/prune references passed. Integer sigmoid/tanh enabling rejected
per upstream; widened INT8→FP16/BF16/INT32 outputs remain unimplemented, not
hardware-limited. Nonfinite epilogues/exhaustive conversion edges unqualified.
Initial TILE fixture converted fractional entries to zero and violated its own
nonzero-cardinality assertion. r2 extra legacy CPU assertion used a truncating
cast; fixed nearest-even saturation. r3 passed all150 checks, then count parsing
failed because numerical PASS lines were absent; reporting fixed. Failures and
restorations retained in `n17-sparselt-attempts.json`; production source/library
hash unchanged across all four attempts. Every failure restored N1.6 under lock.
Initial build6.9s, successful cached restage2.8s, integer1.6s/scalar1.6s/broadcast3.0s/
bias2.7s/vector2.2s gates,16jobs/20GiB ccache/exclusive lock. Kernel logs identical
1009 lines. `SOURCE_N17.json` verifies8 changed files,24 patches reconstruct exact
tree; prior independent library `WS/recovery/n17-sparselt-before.so.0.2` retained.
No performance/automatic Torch routing/new matrix-instruction/long-stability
claim; .105/kernel/Torch unchanged. Next N1.8: inspect existing Lt scalar/register
shape dispatch and timing evidence; select one measured small/skinny improvement.

N1.8 `WS/logs/n18-lt-r4/result.json`, repository
`bc250_scalar_library_results/n18-{lt,lt-attempts,lt-performance,lt-pack,reconstruction}.json`:
664 CPU-double-referenced GPU cases plus403 scalar/561 extended Lt regressions.
Native 32-lane scalar subgroup/output (eight/block) for K<=64 and either min(M,N)<=4
or M,N<=32 with area<=512; no LDS/barriers, physical D-order output mapping. FP32
tree reduction changes rounding from ascending-K FMA; shared existing epilogues
retain scalar/bias/activation/AUX/DGELU. Same/mixed f32/f16/bf16, orders/transposes,
padded/broadcast batches, host/device/zero scalars, compatible in-place C/D, tails,
cross-batch blocks, K0/1/15/16/17/63/64/65, area512/544 and guard checks passed.
Initial direct one-thread/output candidate passed504 cases but selected ratios
0.46–0.76x; rejected. Its DSO-only restore left a mismatched external BLAS pack:
A2 aborted134 on missing GPU symbol. Exact prior DSO SHA matched frozen P2, whose
retained pack restored the baseline pair under lock. r2 cooperative path passed
504+964 cases and paired A2 restoration, but32x32 ratios0.91–0.96x regressed; final
area<=512 bound keeps32x32 tiled. r3 passed664+964; final host grid-bound
review added fallback to the tiled path when subgroup blocks exceed INT_MAX. r4
repeated664+964 and all36 paired timing fixtures; oversized launches not executed.
Every attempt/log preserved; no GPU fault/reset
or new kernel messages. N1.8 runner restores/hashes both DSO and pack on failures.
**Correction to historical rollback wording:** N1.1–N1.7 reports only checked DSO
restoration; they did not verify/restore external packs. Do not treat those fields
as complete runtime-pair recovery. Their successful final GPU gates remain valid
recorded evidence; old runners need paired recovery before reuse. This missing
code-object-symbol abort is separate from unresolved Python CPU segfault0x3b3e6
and the already fixed allocator corruption/default-pool issue.
36 deterministic same-input C++ fixtures, column-major NN/batch1/host scalars/default
epilogue, A1/B1/B2/A2 order; five GPU-event samples of32-launch HIP graph/pass.
Selected skinny ratios1.14–1.86x, selected small1.21–1.61x; controls0.998–1.006x;
max repeated-pass median drift0.24%. Kernel-focused graph replay, not uncaptured
application latency or Torch speedup; clocks untrusted, no clock attribution.
Final targeted16-job/20GiB ccache build11.8s, numerical3.7s/scalar2.3s/extended2.1s;
timing passes1.0–1.2s, all under exclusive lock. Kernel logs identical1009 lines.
`SOURCE_N18.json` verifies12 source files plus BLAS DSO/pack hashes;26 patches
reconstruct exact tree. Pack payload comparison:1 Lt entry changed,68 unchanged
including all67 rocBLAS entries. Recovery DSO `WS/recovery/n18-lt-before.so.1.5`
and pack `WS/recovery/n18-blas_lib_gfx1013-before.kpack` independent copies retained.
No .105/kernel/Torch/frozen candidate changes, automatic consumer routing, universal
bitwise/nonfinite or sustained qualification. N1 bounded roadmap items complete;
optional widened INT8/beta-vector extensions remain open. Next N2.1 solver audit.

## Defects and traps already investigated: do not rediscover them

| Issue                           | Finding / resolution                                                                                                                         | Evidence to preserve                                                                                                                           |
| ------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------------- |
| Intermittent CG/CSR zero output | Proven live CSR analysis metadata overwrite associated with experimental VM heap; compiled default now uses the established pool backend     | Root `47925cbd5`, systems patch0003, `rocalution-step-trace.log`, allocation/raw-metadata/first-kernel probes, 40 plus fresh-process200 checks |
| Experimental VM heap            | Default bypass is a mitigation, not a repair. VMM/async pools retained; explicit opt-in remains experimental.                                | `rocdevice.cpp`, root `bd6e4c8a4`, release gates                                                                                               |
| SVM CPU-unmap returned error | Patch0004 retains queue eviction after failed invalidation following confirmed stop; partial/full targeted -EIO tests and fresh GPU references pass. Failed stop/retry migration remain open; validation locking qualified separately below. | `bc250stack/kernel/r3/evidence/svm-cpu-unmap/` |
| SVM validation invalidation | Patch0005 flushes validation map/unmap after BO unlock; PROT_NONE bounds fixed (trace0→2); two concurrent54-reference GPU processes and full regressions pass. No global lockdep/failure-path claim. | `bc250stack/kernel/r3/evidence/svm-map-deferred/` |
| HPD IRQ initialization | Patch0006 queries active IRQ capability before bulk clear; dummy ack/set/clear errors8 each→0 and IRQ ASSERT WARNs2→0. Four real clears/DC/two disconnected DP connectors retained. Physical display/resume unqualified. | `bc250stack/kernel/r3/evidence/hpd-capability/` |
| Explicit fixed-VA VMM remap | K5 GEM_VA runlist fix8f3c0d3: unchanged two-process VMM gate and HIP/ROCr26-check references pass; historical r2/r3 failures preserved. No broad/failure-path claim. Separate from VM-heap corruption. | `bc250stack/kernel/r3/evidence/k5-summary.json`, `k5-gem-va-full.json`, original/diagnostic failures |
| Later Python CPU segfault       | Execution fault at0x3b3e6, no identified stack/operator; cause remains unknown. Stable-prefix sparse gates and independent candidate passed. | Older operator-candidate/kernel logs, P2 candidate/launcher20 cycles each; do not merge with the proven CSR defect                             |
| Triton incorrect buffer reads   | Missing RDNA1 descriptor flag branch; globals passed while buffer path failed. Fixed.                                                        | Triton patch0004/root `9fdae0b5d`; both memory paths validated                                                                                 |
| Triton low-precision lowering   | Unsupported gfx13 packed dot/BF16 multiply replaced by widened scalar arithmetic; signed limits and rounding-tie/NaN tests retained          | Four Triton patches and MLIR/LLVM/assembly/GPU gates                                                                                           |
| NHWC avgpool backward           | Padding correction in Torch source                                                                                                           | Torch patch0001, operator coverage                                                                                                             |
| MIOpen bias/fusion              | Legacy bias beta1 doubles output; beta0 is correct. BF16 and FP16 NHWC pointwise required private scalar HIP fallback; output routing fixed. | Library patch0016, provider CPU references; mixed-intermediate rejection                                                                       |
| Lt SWISH                        | Public beta is fixed1; activation arg0 is not Swish beta                                                                                     | 465 then561 extended consumer cases; hipDNN Lt48                                                                                               |
| RPP audio                       | Spectrogram destination stride and decibel reduction/reference/zero handling fixed; FFTS dependencies/linking fixed                          | Patches0010/0013/0015 and audio26                                                                                                              |
| Source generation               | Torch CMake cache can leave HIPHooks stale even after pip; explicitly hipify before editable rebuild                                         | `build_torch.sh`, `rebuild_torch_pooling.sh`                                                                                                   |
| Installed headers               | CK public headers and provider TOML/.hip require explicit superbuild tracking; noop build alone does not prove staging                       | `math-libs/CMakeLists.txt`, `ml-libs/CMakeLists.txt`                                                                                           |
| Kernel packs                    | hipSPARSELt owns a separate sparselt artifact; extracted pack collisions fixed                                                               | `math-libs/BLAS/CMakeLists.txt`, descriptors; do not merge it back into generic sparse                                                         |
| Reference/test mistakes         | CPU.double() can alias float64 inputs; gradients need detached clones. Some upstream mixed-inference CPU fixtures were limited.              | Corrected P2 math references and meaningful training fixtures, not all-skipped success                                                         |
| Build resources                 | Global dist restarted unrelated MIOpen and was stopped137; targeted component+dist passed0                                                   | P2 stage logs; preserve objects and resume global builds only when needed                                                                      |
| Old/new timing                  | Seed-only fixtures differed; replaced with identical serialized bytes. Clock reports conflict.                                               | Matched reports; excluded raw runs are not comparison proof                                                                                    |

## Performance already measured, with the correct interpretation

Root `24ea38d84`, [performance report](bc250_performance.md), and
`bc250_performance_results/` record nine matched workloads on `.104` 10.2/Torch 2.13
versus `.105` 7.2.4/Torch 2.9.1. GEMM/copy mostly differed ~1–6%; copy~0.2%.
Small CNN results were highly variable on `.105` (59.9% IQR); an apparent 3.4x ratio
is not a stable improvement. Different boards, thermal states and untrusted
clock telemetry prevent attribution of small differences to ROCm alone.

SmolLM2 batch 1/32-token timings were ~131.56ms full FP32 update,~111.72ms FP32 LoRA
without checkpointing,~260ms AMP LoRA with checkpointing and~139–141ms without.
These are training-step processed tokens, not generation tokens/second. The
smallest model may benefit from FP32 because overhead outweighs low-precision compute.

P2 is a separate same-board scalar-kernel comparison: 71 CPU-checked timing cases,
964 Lt API checks, medium/large Lt GEMM~2.1–2.4x faster than prior Lt. Small shapes were
roughly flat/slightly worse. Default Torch BLAS did not change; this is not a 2.4x
whole-Torch or old/new board speedup. Plain compiled reductions often lost to eager.
Explicit HIP graph replay for compiled SiLU/square/sum at 1,048,576 elements measured
~4x over eager; tested large sum/dot~1.35–1.4x. Captured storage/shapes must remain fixed.
For known SPD matrices, Cholesky factorization/reuse was cheaper than general LU solve;
never silently assume all matrices are SPD. The benchmark source encodes exact cases.

## Evidence registry and reproducible workflow

| Scope                             | Durable location                                                                                                                       |
| --------------------------------- | -------------------------------------------------------------------------------------------------------------------------------------- |
| Initial/expanded native gates     | `WS/logs/`, `bc250_port.md`, historical support matrix/roadmap                                                                         |
| Pool corruption/regressions       | `rocalution_diagnostic.cpp`, `sparse_analysis_trace.cpp`, `WS/logs/*rocalution*`, allocator/raw-metadata logs                          |
| Training/memory                   | `bc250_training_memory.md`, `WS/logs/training-memory/`, retained model/adapter artifacts                                               |
| Matched old/new/model performance | `bc250_performance_results/*.json`; `/home/nik/bc250work/rocm-performance-20261013` on both hosts                                      |
| P1/P2 numerical evidence          | `bc250_scalar_library_results/priority1-*.json` and `priority2-*.json`; host logs                                                      |
| Independent frozen release        | 600-second soak40,339 cycles; prior candidate39,657; review46,378 file hashes. These do not qualify current mutable P2 long stability. |
| P2 candidate (retired payload)    | `candidates/priority2-20261014/runtime.retired.json`,191 historical DSO/pack hashes; validation/candidate.json retained after20261016 retirement |
| P2 launcher                       | `/home/nik/bc250work/priority2-launcher.{json,pt}`,20 cycles/40 sparse matmuls, compiled gradients and checkpoint                      |
| Kernel comparison                 | P2 report records82 new bridge/veth messages, no new fault/reset/OOM; logs are not byte-identical                                      |

Always back up and restore changed DSOs together with the external `.kpack` files
they reference, under the exclusive prefix lock. Verify both hashes; a DSO hash
alone cannot establish runtime rollback. N1.8 paired A2 execution checks recovery.

Build without mapped GPU devices; run bounded GPU gates serially after staging.
Use existing objects/20 GiB ccache:16 normal jobs, hipTensor 12, hipDNN providers 6,
rocALUTION 8, Torch 12/link 1, Flang 4/LLVM link 1. rocFFT HIPRTC AOT cache has its own
persistent 16-worker pool, not ccache. Evaluate setup_ccache.py in each builder shell.
Native bootstrap compiler remains under `/usr/lib/llvm-18`; masquerade aliases preserve
old LLVM command lines, not architecture identity.

Exact opt-in: `THEROCK_GFX1013_SCALAR_BACKENDS=ON` with exclusive
`THEROCK_AMDGPU_FAMILIES=gfx1013`. A leftover cache variable named
`THEROCK_ENABLE_GFX1013_SCALAR_BACKENDS=OFF` is not this option. Top-level group
flags can be OFF while dependencies/selected child projects are activated; inspect
actual targets/artifacts, not just group flags. HIPBLASLt provider is enabled;
RCCL/rocSHMEM/MPI/debug/media are off in this profile.

Prefer component `hipBLASLt+dist hipTensor+dist hipSPARSELt+dist` targets to global
dist for local changes. `.104` source exports include root-owned directories: transfer
explicit changed files via tar/checksums, not rsync directory timestamps over them.
Update docs/provenance separately; do not relabel older archive baselines.

Torch rebuild must use explicit HIP generation:

```bash
python tools/amd_build/build_amd.py --out-of-place-only
python -m pip install -e . -v --no-build-isolation
```

Triton lowering changes require `make` before MLIR/LLVM tests. Keep qualified wheel
unless source changed. Exact patch reconstruction helper is
`build_tools/bc250/verify_priority2_reconstruction.py`; it needs local Git objects,
not gitless host exports. Do not rerun all GPU gates just to edit memory/docs.

## Earlier agents, legacy work and archive recovery

BC-250 milestones:

| Anchor                          | Work already done                                                 |
| ------------------------------- | ----------------------------------------------------------------- |
| 7a5c2ca0f                       | Native gfx1013 target/bootstrap                                   |
| a640c9296                       | Initial independent ROCm/Torch/Triton deployment                  |
| 47925cbd5 /4b000d542            | Safe allocator default; expanded candidate/release gates          |
| b2707998a /761ec5803 /f1895f712 | Persistent host paths/files/read-only mounts/shell                |
| d744f0952                       | Real SmolLM2/AMP LoRA/resume/memory concurrency                   |
| 24ea38d84                       | Matched old/new and training performance                          |
| c8dc436d3                       | Frozen snapshot review/pause policy, subsequently revoked by user |
| fdde5561c                       | Scalar backends/operators expansion                               |
| 71f9f0ee4                       | P1 consumers/CK Tile/DNN/Torch/dev activation                     |
| ce0025df5                       | P2 algebra/primitives/math/performance/candidate/runtime-lock     |

Separate branch `users/nik/alpine-musl-port`, tip `36c62955f`, has 22 commits after
the common base. It is **not an ancestor of the active BC-250 branch**. It contains
native-musl toolchain/runtime/COMGR-stack/portable FFT/SMI/OpenCL work, qualified
on Ryzen 8645HS/Radeon 760M **gfx1103**, not BC-250. Examples: `a05baf1ae`,
`7aacbbd89`, `ef108c4ff`. Recover its historical doc with
`git show 36c62955f:docs/development/alpine_port.md`. This work is reusable research,
not current BC-250 native-musl qualification; don't switch branches or integrate it
silently. The current port runs glibc userspace in Podman on Alpine.

Local project thread index: `/home/nik/.codex/state_5.sqlite`, read-only lookup by
cwd `/home/nik/Documents/Vibe/therock`. Raw rollouts live under
`/home/nik/.codex/sessions/YYYY/MM/DD/`. Retrieve only relevant project transcripts if
source/evidence cannot answer a question; do not copy titles/prompts/credentials into
committed memory. Root BC-250 thread ID `01a12a83-7fad-70d1-ab81-57e2e3cf6253`;
rollout filenames may change across compaction/resume, so find by ID rather than hardcode
one current filename. Prior roles include build_guidance, triton_review,
remaining_runtime_audit, remaining_libraries_audit, hipblaslt_scalar,
hiptensor_scalar, hipsparselt_scalar, p1_hipblaslt/p1_ck_tile/p1_torch_compiled,
p2_communication/p2_tensor_sparse/p2_primitives. The P2 communication agent was
redirected to math coverage; its name is not proof RCCL was ported.

Historical `.105` reference audit (superseded by M105.1 migration2026-10-23) found Alpine 3.24.2/r2 kernel,
legacy sources under `/home/nik/bc250work/rocm-libraries-gfx1013`, branch
`gfx1013-cyanskillfish`, revision `b578bcd410d0641337ce0f34be055b88560f5c24`.
Other retained legacy work: `/home/nik/bc250-stack`, `pytorch-build-gfx1013`,
`research-handoff`, `bc250-rocm-lab`, `bc250-gfx1013-fix`. Do not copy old patches
blindly. M105.1 explicitly authorizes the recorded replacement and retirement;
the old unique sources/Git/modified MIOpen files are archived locally. Its private cgroup2 mount namespace
benchmark workaround was ephemeral, not persistent migration.

## Historical documents and how to avoid stale tasks

`bc250_port.md`, `bc250_release_20261013.md`, `bc250_development_handoff.md`,
`bc250_support_matrix.md`, `bc250_scalar_library_expansion.md`, the old roadmap,
training memory and performance docs are valuable milestone evidence. They may
say the frozen release is active, Lt GEMM unsupported, tensor contractions absent, sparse Lt
excluded, CK Tile untested, Torch unpatched or performance not started. **Those statements
are superseded by the current inventory for its specific scopes.** Do not delete snapshots
or mark every old box done; map completed work to evidence and select an actual remaining gap.

Before repeating any task: identify the component/operation, search this inventory,
patch subjects and validation registry, compare current code/profile, then state why
new work/test is necessary (new change, uncovered case, genuine failure or invalid old
reference). An old unchecked box, context compaction, stale PID, mutable tag name or
component version number mismatch is not a reason to restart the port.

## Unified workspace organization (after the memory audit)

Current local workspace: `/home/nik/Documents/Vibe/bc250stack`. The existing
TheRock checkout moved intact to `therock/`; `/home/nik/Documents/Vibe/therock`
is a compatibility symlink. The separate fresh PyTorch checkout moved intact to
`references/pytorch-upstream`, with its original `Vibe/pytorch` path preserved.
Canonical `torch/` and `triton/` aliases point to the qualified patched sources
inside TheRock, so there is no duplicated or silently upgraded port checkout.

Kernel source was found on .105 at
`/home/nik/bc250work/alpine-rocm-7.1/sources/linux-7.1-bc250-test` and copied
read-only into `kernel/source`. Five source/configuration anchors were verified.
The kernel binary SHA256 is
`792f904835f27574b680d2fd5cfd7e42b348fb3a3d9b39d8e8c87348873c3b0d`, identical
on both hosts. Configuration, legacy scripts and the existing bc250-linux
packaging repository are preserved separately. This is an already-patched source
snapshot, not a clean upstream checkout; generated build products/private signing
keys/initramfs archives are excluded. No remote installation or reboot occurred.

Workspace README/AGENTS and workspace-manifest.json explain component identities.
The workspace repository tracks coordination files; source changes stay in their
component repositories. Board support copies retain driver and CU-unlock history.
New session history may use the new workspace cwd; the earlier Codex thread-index
lookup by the old therock cwd remains useful for archived sessions.

## N2.1 complex Hermitian eigenvectors — 2026-10-15

P2 solver32 audited and rerun unchanged. Native Torch4ee6571 complex eigh dispatch
uses hipSOLVER generic64 DnXsyevd to rocSOLVER complex heevd; no backend
code/selection change.
New22 CPU-complex128/eigenpair/reconstruction/orthogonality/projector/gradient
checks pass for complex64/128, batch2 at17/65/129/257, both triangles, padded
inputs/guards/ignored-triangle NaNs/nondefault stream and repeated subspaces33.
257 crosses source default blocked threshold256; two-stage11000 unqualified.
Final22+32 processes exit0 with identical dmesg and acknowledged600s guards under
runtime lock. Initial harness broadcast failure retained;18-case intermediate
pass retained. Source, mapped-library hashes, dispatch audit and fixed tolerances:
[solver results](bc250_solver_results/README.md). No native rebuild, prefix/boot/
power-policy change or active job. Original CPU crash and separate MIOpen defect
remain unresolved; no universal/performance claim. Next N2.2 nonsymmetric eig
coverage/dispatch audit and one bounded complex eigenpair gap.

## N2.2 nonsymmetric eig compatibility — 2026-10-15

Pinned Torch4ee6571 -> hipSOLVER DnXgeev -> host LAPACK cgeev/zgeev -> device
outputs. This is a CPU compatibility eigensolve, not native GPU; N2.1 Hermitian
eigh remains native separately. New16 groups pass: complex64/128 batch2 at9/33/65,
normal/nonnormal known spectra with basis condition1/20, padded/transposed guards,
nondefault streams, eig/eigvals CPU/analytic eigenvalues and residual/projectors,
two eigenvalue-gradient and two grouped empty-input cases. Max residual3.17e-7/
6.72e-16. First run exit0/dmesg identical/600s guard acknowledged under lock.
No runtime/backend change or native build; no active job. Mapped hipSOLVER and
OpenBLAS hashes, dispatch audit/source/test/results:
[N2.2 report](bc250_solver_results/N22.md). Native non-Hermitian eig, defective/
repeated/extreme-scale/convergence-failure/performance scope remain unqualified;
Python CPU crash and MIOpen defect remain open. Next N2.3 native Hessenberg/QR/
Schur feasibility and smallest reviewed implementation/convergence step.

## N2.3 native small Schur prototype — 2026-10-15

Audited native GEHRD/GEQRF/UNGQR pieces; no pinned native nonsymmetric Schur
iteration found. Added isolated gfx1013_schur_candidate.py, cap16 complex64/128:
owned native GEHRD copy/current-stream private handle, GPU reflector-Q formation,
bounded explicit complex shifted GPU QR, similarity accumulation and checked
roundoff deflation. Python loop/scalar checks synchronize host; no CPU matrix
solve. Positive info on budget exhaustion retains partial T,Q. Schur vectors
are not right eigenvectors. Native production eig remains unimplemented.
Initial17/final21 groups pass with unchanged dmesg and acknowledged600s guards:
orders4/9/16 basis1/10, CPU/analytic spectrum/Hessenberg/Schur invariants, guards/
nondefault stream, zero/one-step exhaustion, repeated triangular, ±i rotation,
scalar and9 argument rejections. Max Schur reconstruction1.35e-6/2.90e-15.
Source/evidence/limits: [N2.3](bc250_solver_results/N23.md). No default dispatch/
Torch/DSO/prefix/boot policy change or native rebuild; no active jobs. Extreme
scaling/balancing/general convergence/vectors/batching/gradients/performance not
qualified. CPU fault/MIOpen defect remain open. Next N2.4 guarded small-Schur
right eigenvector recovery and explicit spectrum/convergence refusal contracts.

## N2.4 native small-Schur right eigenvectors — 2026-10-15

Extended isolated cap16 candidate with GPU triangular back-substitution,
dimensionless scaling/homogeneous rescaling, checked gap/pivots/residuals and GPU
Q transform/unit norm. Positive Schur info and malformed/nonunitary T/Q refused;
scaled pairwise gaps<=8sqrt(eps) reject repeated/close spectra without perturbation
or host solve. All21 groups pass first run: complex64/128 orders4/9/16 basis1/10,
CPU/analytic spectra and vector projectors, original residuals, stream/guard/
Schur immutability, analytic nonnormal2x2, repeated/Jordan/close/zero refusals and
above-floor control, budget0/1 refusal, scalar zero/nonzero and7 malformed cases.
Max original residual3.84e-7/7.79e-16. Exit0/dmesg identical/600s guard ACKed;
no active jobs. Source, hashes, report and evidence: [N2.4](bc250_solver_results/N24.md).
Prior helper retained in WS/recovery/solver-n24-before. Native GPU arithmetic
with Python scalar control, no production Torch eig/DSO/prefix/kernel change or
native rebuild. General scaling/balancing/conditioning/repeated eigenvectors,
batching/gradients/performance unqualified; CPU fault/MIOpen defect remain open.
Next N2.5 bounded scale/separation ladder with nonnormal/refusal controls before
any opt-in integration. Host-LAPACK default retained.

## N2.5 native small candidate scaling — 2026-10-15

Reproduced4/6 seeded baseline magnitude failures (tiny256-step exhaustion, large
nonfinite/fill) with unit controls passing. Fixed isolated Schur driver by GPU
component normalization before native GEHRD/QR, finite checked original-unit T
restoration on success/partial results, and safe right-recovery normalization
with unchanged8sqrt(eps) separation. Bitwise subnormal/zero distinction and
nonrepresentable-output refusal tested; no tolerance/iteration changes.
Seeded6 replay now6/6; scale40 (c64 1e-25..1e25, c128 1e-200..1e200,4/9 basis1/10),
boundary30 (18accept/12refuse; passed before too), failure/control16 and N2.4
regression21 pass. Final logs unchanged/guards acknowledged, no active jobs.
[Source/evidence/limits](bc250_solver_results/N25.md); baseline helper retained
at WS/recovery/solver-n25-before. Production Torch eig/DSOs/prefix/kernel/boot
unchanged; no native rebuild or performance claim. General dynamic ranges,
balancing/subnormals/conditioning/batching/gradients remain unqualified; original
CPU fault and separate MIOpen defect unresolved. Next N2.6 bounded cost/profile
versus host eig and one measured overhead-reduction step before any integration.

## N2.18 scalar CK DL grouped forward — 2026-10-19

TheRock `746528fa6`;128 FP32 grouped convolution CPU-reference comparisons +8
vector-shape refusals pass, GEMM FP32/FP16/INT8 examples3 and Torch core7 retained.
Plain/Add+ReLU, four1x1/3x3 shapes with stride/dilation/asymmetric padding and tile
tails, groups1/3, contiguous/padded strided layouts, default/private streams.
Independent CPU double oracle: dyadic exact, fixed non-dyadic tolerance, max abs
error4.255575e-6.256-byte guards/output holes and immutable inputs/weights/D pass.
No production arithmetic defect or runtime/new operator/routing/performance change.
Four bounded standalone build attempts preserved (ccache20GiB/OMP4/one job/native
gfx1013/no GPU/network). r1 test namespace error; r2 r3-default guard inactive
on already-running r2; r3 invalid padded packed-layout fixture. Corrected r4
passes with explicit strided-layout contract, no tolerance/reference relaxation.
Launcher explicit --guard-release for retained r2/r3, unchanged r3 default and
existing boot guard, early mismatch refusal/guard diagnostics. Old helper at
WS/recovery/ck-dl-n218-before. Four GPU guards including failed fixture ACK/600s;
all4 attempts' dmesg identical/191 protected hashes + new binary sealed.
Kernel observed r2 at entry, boot d0532075-88fb-4a2f-974e-f61578d20d9f; no reboot
or boot-policy change. Native Torchdc8fea0/source2b449c4/libraries319a235d6/source-
doc c29021b40e7/links/runtime/defaults/four AOT DBs/.105/BIOS preserved. Free~7.16GiB,
no running agent job/container/guard. [Report/evidence](bc250_ck_dl_n218_results/README.md).
Raw WS/logs/ck-dl-n218-*, outputs WS/build/bc250-coverage/ck-dl-n218-r{1,2,3,4},
local ignored kernel/artifacts/ck-dl-n218. Next preservation baseline
WS/logs/ck-dl-n218-qualification-r4/final.json plus binary_sha256. CPU Python fault
unresolved/separate from fixed VM-heap/CSR and test-only errors. Forward FP32 only;
backward/other types/layouts/configurations/graphs/concurrency/tuning unqualified.
Next review N2 practical gaps versus current model workloads/B01/B02/B07/A09,
then select next N3 batch without repeating completed AMP/compile/resume/CK Tile.
No automatic N2.19 or exhaustive API-combination gate; broad N2 remains open.

## N2.17 floating sorting/selection and Torch consumers — 2026-10-18

TheRock `5826261d8`; native rocPRIM 656 / hipCUB 656 / Thrust 648 checks pass for
FP32/64 ascending/descending stable pairs, explicit NaNs-last comparisons,
five stable selection policies and16 float partial-bit refusals. Sizes 0..65537,
default/private streams, guarded outputs/tails/workspace and immutable inputs/
indices/flags. Exact keys/payload indices preserve zeros/subnormals/infinities/
quiet-NaN sign+payload bits; pinned radix NaN bit-order is distinguished from
custom comparison and Torch policies. Torch62:54 finite output/index/base-
gradient references plus eight special-value output/index fixtures. No nonfinite
gradient, whole-Torch consumer or speed claim. P2 primitives138/core7 pass.
Initial r1 build failed new adapter's mixed-type auto declaration, fixed in r2;
original source/compiler logs and partial binaries preserved. Three standalone
units only, ccache20GiB/OMP4/three jobs/native gfx1013, no mapped GPU/network.
Six GPU gates/guards ACK/600s, dmesg unchanged entire batch incl failed build;
181 protected hashes match, new binaries separately sealed. No production
defect/Torch/library/header install/default/prefix change. Active native Torch
`dc8fea0` / source `2b449c4` / libraries `319a235d6` / source-doc `c29021b40e7` and image/kernel/boot/
links/DSOs/four FFT AOT DBs/.105/BIOS/r2 fallback preserved. No active jobs;
free ~7.16GiB. [Report/evidence/contracts](bc250_sorting_n217_results/README.md).
Raw WS/logs/sorting-n217-*, outputs WS/build/bc250-coverage/sorting-n217-r{1,2},
local ignored kernel/artifacts/sorting-n217. Preservation baseline
WS/logs/sorting-n217-qualification-r2/final.json plus binary_sha256 for new test
products. CPU Python fault remains unresolved/separate from fixed VM-heap/CSR
and explicit compile error. Next N2.18 audit scalar CK DL admission/GEMM then
one admitted convolution/grouped CPU-reference case; XDL/WMMA/modern CK plugin
off, completed CK Tile work retained.

## N2.16 floating/segmented/custom primitives — 2026-10-18

TheRock `2db3b18c1`; rocPRIM 576 / hipCUB 480 / rocThrust 480 new CPU-reference checks,
retained P2 INT32 138 and Torch core7 pass on first attempt. FP32/64 reduce,
inclusive/exclusive/segmented/by-key/in-place scan, uneven/empty segments and
288 exact custom associative argmax/first-index ties. Sizes 0..65537, default/
private streams; guarded inputs/outputs/temporary storage, immutable read-only
inputs/keys/offsets. Dyadic sums exact; non-dyadic CPU long-double refs use fixed
tolerances, max error 5.72e-6/8.77e-15. Thrust reduce-by-key has no empty/init
contract; checked nonempty keys/count/results separately from offset APIs.
Only three standalone test units built, ccache20GiB/OMP4/three jobs/native gfx1013,
builder fa3e8c254, no GPU/network. New WS/build/bc250-coverage/primitives-n216-r1
preserves all prior builds/P2 binaries/cache. No production bug or new operator/
Torch/header/DSO/default/prefix install. Five guards ACK/600s, dmesg unchanged,
168 protected identities and new binary hashes sealed; four FFT AOT DBs retained.
Active Torch `dc8fea0` / source `2b449c4` / native libraries `319a235d6` / source-doc `c29021b40e7`,
links/kernel/boot/BIOS/.105/r2 fallback unchanged. No active jobs; free ~7.19GiB.
[Report/evidence/contracts](bc250_primitives_n216_results/README.md); raw
WS/logs/primitives-n216-*, local ignored kernel/artifacts/primitives-n216.
Next preservation baseline WS/logs/primitives-n216-qualification-r1/final.json
plus binary_sha256 for new test products. No broad precision/determinism/speed/
graphs/concurrency/Torch consumer-autograd claim. CPU Python fault unresolved/
separate from fixed VM-heap/CSR. Next N2.17 bounded floating stable sorts and
selection after existing P2 audit, explicit ordering/special-value contracts.

## N2.15 RNG offset/state/stream qualification — 2026-10-18

TheRock `0e4557ee5`; libraries documentation `c29021b40e7` on native319a235.
State/layout80, analytic distribution22, offset/isolation/stream/dropout20 and
fresh-process continuation1 pass. Low-precision/complex normal/integer, guarded
strides, empty/odd/tail, seed>2^63/offset>2^40; exact GPU/CPU/Python continuation
ten fields, native dropout output/backward CPU mask refs and five GPU kernels.
Direct native68 includes five pseudo/four Sobol variants, offset reset/stream
rebinding, exact prefix relations and scalar CPU Philox through both wrappers.
Initial native59/60: LFSR113 offset accepted despite stale rocrand.h restriction;
implementation and upstream offset tests confirm intentional support. Comments
corrected, C++ wrapper distinction retained, nine extra LFSR probes pass incl
high-offset non-aliasing. MTGP32 TYPE_ERROR still expected. No arithmetic/ABI/
binary/default/prefix/header-install change; comments staged in source only with
hash-verified recovery WS/recovery/rng-n215-source-before. Patch0032 retained.
RNG28/core7 pass; seven successful GPU containers plus initial assertion failure,
eight guards ACK/600s, dmesg unchanged whole batch;153 protected hashes/four FFT
AOT DBs match. Active Torchdc8fea0/source2b449c4/libraries319a235/kernel/boot/
fallbacks/.105 preserved. No active agent jobs; free~7.21GiB. CPU Python fault
unresolved/separate from fixed VM-heap/CSR and explicit contract exception.
[Report/evidence/limits](bc250_rng_n215_results/README.md); raw WS/logs/rng-n215-*,
ignored kernel/artifacts/rng-n215. Current preservation baseline
WS/logs/rng-n215-qualification-r2/final.json. No broad statistics/tail/ordering/
concurrent/graph/compiled/cross-version replay or speed claim. Next N2.16 audit
P2 primitives138 then bounded floating/segmented reduction/scan references.

## N2.14 FFT plan/layout/odd-real qualification — 2026-10-18

TheRock `44568c2e2`; no native/build/default/prefix change required. Installed
Torch native `dc8fea0` / test source `2b449c4`, libraries `319a235d6` unchanged.
Complex96/real72 each verify CPU-double output, weighted first backward into
the full leaf, private-stream GPU layouts, input immutability and guarded out.
Includes odd17/45/7x13, three norms, crop/pad, conjugate/broadcast and four-dim
decomposition. Three explicit DFT/finite-difference/cache-reuse checks pass;
trace34 kernels includes native rocFFT forward/backward. Direct hipFFT16 plans/
32 executions check single/double real/complex, caller workspace, batches3,
advanced strides and synchronized stream rebinding. Native rocFFT permits
input/gap writes; direct tests guard exterior spans, public tests all gaps.
FFT30/core7 retained. Initial48 double fixture errors were non-leaf GPU .grad,
fixed by detaching test leaf; no production or tolerance change, evidence kept.
Seven guards ACK/600s, six passed processes, dmesg unchanged including failure;
145 hashes/four AOT DBs/runtime/links/kernel/boot/r2 fallback/.105 preserved.
No running agent job/guard/container, free~7.22GiB. No speed/universal-correctness
claim; B01 BF16 promotion/native in-place/planar/giant/callbacks/graphs/higher
gradients remain unqualified. [Report/evidence](bc250_fft_n214_results/README.md).
Current preservation baseline WS/logs/fft-n214-qualification-r2/final.json;
raw WS/logs/fft-n214-*, ignored kernel/artifacts/fft-n214. Historical Python CPU
segfault unresolved/separate from fixed VM-heap/CSR and this fixture exception.
Next N2.15 audit P2 RNG28/checkpoint coverage; bounded offset/state replay and
private-stream sequencing against exact replay/analytic distribution contracts.

## N2.13 sparse triangular coverage and exact pivots — 2026-10-18

Installed native Torch `dc8fea0` (implementation `a434d03`), final tests `2b449c4`;
TheRock build `bbf556448`, controllers `39ea58574`. Existing generic CSR SpSV/SpSM
and BSR paths pass bounded four-dtype/index/layout/stream/flag CPU references.
Pre-fix healthy256 pass; singular/empty failures16/32 then39/80 expose finite
placeholders after division avoidance. New ROCm-only2D Torch GPU postcheck marks
exact zero/missing scalar pivot rows as NaN using compressed diagonal binary
search and bit-exact zero tests; no host reads/dense conversion/workspace/pivot
perturbation. One extra GPU launch on nonunit/nonempty solves; no throughput or
near-zero/conditioning claim. Other singular rows unspecified; legacy empty-A
NaNs including unit mode retained. Unit solves bypass. Direct raw library APIs
unchanged and general sparse spsolve/factorization B02 remains separate.

Each candidate/installed mount passes433 new public groups (256 matrix,80 pivot/
empty,32 larger CSR65/257+BSR66/258 block3,64 implicit unit,one real kernel trace),
96 native generic fixtures/192 solves (N/T/adjoint, host/device alpha, private
stream, two submissions/analysis, sentinels), sparse41/core7/CPU-first profiler.
Final pivot group tests NaN writes into strided sentinel output/private stream;
only that group rerun after test-only strengthening, native binaries unchanged.
BSR int64 public metadata is downcast within admitted bounds; no giant native64
BSR, batching, sparse triangular autograd/compile/graphs/universal performance
qualification. Only libtorch_hip.so changes; CPU profiler fix remains identical.
121 prior identities preserved,22 guards ACK/600s and unchanged dmesg whole batch.
Independent WS/pytorch-sparse-n213, pip-only after required HIP generation,
ccache20GiB/jobs12/link1/native gfx1013; old sources/builds/caches/third_party/
releases/fallbacks retained. Only dev Torch link/runtime metadata changes; ROCm
DSOs/packs/image/kernel/boot/defaults/.105 unchanged. Recovery
WS/recovery/sparse-n213-before, SOURCE_SPARSE_N213.json in WS/current; final
preservation baseline WS/logs/sparse-n213-qualification/final.json with candidate/
current-runtime hashes. [Report/results/recovery](bc250_sparse_n213_results/README.md).
Raw WS/logs/sparse-n213-* and ignored kernel/artifacts/sparse-n213. No active jobs;
free~7.23GiB. Python CPU segfault unresolved/separate from fixed VM-heap/CSR.
Next N2.14 FFT plan/layout/odd-real output+gradient coverage after P2 FFT30 audit;
AOT cache preserved, BF16 promotion B01 separate. Patch0010 incremental mbox on
Torch da45369 retains all three native/test commits; historical reconstruction
gap remains.

## N2.12 native triangular recovery — 2026-10-18

TheRock codeb1c94bddc adds explicit recovery_backend=device to isolated
right_eigenvectors; one32-thread HIP solve, cap16 complex64/128. Per-row
homogeneous column scaling retained; existing finite/unitary/triangular/gap/
unitnorm/residual checks common to both paths. N2.6 reference/default and
production host-LAPACK eig unchanged.128 retained groups, six CPU fixtures/36
backend evaluations,32 all-order direct cases,two near-gap rescaling stresses,
six sentinel/native ABI cases,six ABI refusals,four public/output refusals pass.
Default exact immutable N2.11; real GPU kernel traced. Paired three-sample
A1/B1/B2/A2 recovery1.32–2.32x/whole warm candidate1.17–1.34x in six fixtures;
recovery~1.90–2.10ms, host eig~0.213–0.408ms remains faster than native full
~3.75–11.79ms. No general/model/LoRA gain/integration; scalar reads remain18,
not all synchronizations. Nine guards ACK/600s, eight successful process gates;
initial required trace check honestly failed: CPU-only-first profiler leaves
later GPU captures empty. Source points to one-time cpuOnly registration; final
fresh process starts CPU+CUDA and traces native kernel. **A09 initialization
ordering defect remains unresolved**, independent of solver arithmetic. Both
traces/logs retained.104 protected hashes match; dmesg unchanged entire batch.
One-unit hipcc/ccache20GiB/OMP4 build; no Torch/installed ROCm DSO/prefix/default/
image/kernel/boot/.105 mutation. Prior helper and oracle retained under
WS/recovery/solver-n212-before and helpers/schur_n211_reference.py. [Report](bc250_solver_results/N212.md)
and n212/ contain full results/commands/hashes; raw WS/logs/solver-*n212*.
No active job/container/guard, disk~13GiB free. Python CPU segfault unresolved/
separate from fixed VM-heap/CSR. Next A09/N4 first-CPU→GPU initialization
reproducer/fix in separate candidate with both-order numerical/trace gates,
then broader N2 sparse/FFT/RNG coverage triage. N2.1–12 complete within recorded
bounds; N2–N8 still seven broad workstreams. Further small-eig optimization is
optional; no larger/batched/close-spectrum/balancing/gradient/compile/graph/
concurrency/general convergence contract added.

## N2.11 explicit GEHRD sessions — 2026-10-18

TheRock code807642198 adds explicit caller-owned GehrdSession, one PID/thread/
device/stream; native handle/bindings reused, fresh owned GEHRD copies retained,
close waits pinned stream then destroys once. No global cache/default mutation.
Create3.13–3.18ms dominates binding0.11–0.13/alloc-copy0.06–0.08/destroy~0.13ms.
128 retained groups, six CPU/backend fixtures, three lifetime groups/16 refusals
pass: mixed-dtype reuse, close synchronization/idempotence, exception after GEHRD,
two private sessions/streams interleaved, wrong thread/stream/device/closed/busy/
PID. PID guard injected, no real post-fork GPU test; device mismatch metadata-only.
Default exact N2.10 and fresh/reused H/Q exact on fixtures. Seven guards ACK/600s/
unchanged dmesg;82 protected hashes unchanged. Six paired three-sample warm prep
7.03–12.52x/whole1.28–1.95x with N2.8 QR/device Q; creation3.32–3.41ms/close outside
warm pairs. Cold one-call prep4.24–4.51ms gives no warm gain; host eig0.219–0.408ms
still faster. No general/model/LoRA/Torch-wide gain; N2.9 regression unchanged.
Python-only/no build/ccache/DSO/Torch/image/prefix metadata/kernel/boot/default/
.105 mutation. gfx1013_gehrd_session.py accompanies candidate. Prior helper/oracle
retained WS/recovery/solver-n211-before and schur_n210_reference.py. [Report and
reproduction](bc250_solver_results/N211.md), raw WS/logs/solver-*n211*. No active
job/container/guard; disk~1.08GiB free. No general concurrency/batching/gradient/
compile/graph/convergence/production GPU eig contract. Python CPU fault remains
unresolved/separate from fixed VM-heap/CSR. Next N2.12 profile warm-session recovery,
bounded native triangular back-substitution, Python/N2.6 reference/fixed separation/
rescaling/unitnorm/residual/CPU gates/cap16/host LAPACK default retained.

## N2.10 device GEHRD reflector-Q — 2026-10-18

TheRock codecc8a05d3c adds explicit q_backend=device to isolated hessenberg/schur:
one32-thread HIP complex reflector product, cap16/complex64+128. Native GEHRD,
owned packed input/current-stream handle lifetime and reference/production host
LAPACK defaults unchanged. Initial order16 split GEHRD~0.37–0.40ms/Q1.87–1.91ms/
preparation~6.2ms; remaining setup/teardown not attributed.128 retained groups,
24 independent CPU complex/zero/mixed reflector cases with strided guards/streams,
six complete eigenpair fixtures across both Q/three QR backends, exact old-default
reference/H controls, zero/scalar/refusal and GPU kernel trace pass. Fixed bounds
unchanged. Six paired three-sample preparation1.11–1.39x, whole N2.8-control
candidate1.05–1.12x; diagnostic Q1.87->0.17/1.83->0.19ms at16 includes submission/
sync. Host full eig0.222–0.429ms still far faster; no general/model/LoRA gain or
production integration. N2.9 regression stays distinct. Seven guards ACK/600s/
unchanged dmesg;79 protected hashes unchanged. One-unit hipcc/ccache20GiB/OMP4,
no Torch/ROCm DSO/image/prefix metadata/kernel/default/boot/.105 mutation. Prior
helper/oracle retained WS/recovery/solver-n210-before and schur_n29_reference.py;
all previous binaries retained. [Report/reproduction](bc250_solver_results/N210.md),
raw WS/logs/solver-*n210*. No active job/container/guard; disk~1.08GiB free.
No general convergence/batching/gradient/compile/graph/concurrency contract.
Python CPU fault unresolved/separate from fixed VM-heap/CSR. Next N2.11 profile
binding/handle creation-destruction/allocations and one measured reuse/cache fix,
retaining explicit lifetime/current-stream tests/owned GEHRD inputs/Python oracle/
cap16/fixed numerical/failure gates/host production eig default.

## N2.9 bounded iteration chunks — 2026-10-18

TheRock code13f91e094 adds explicit device_chunked backend, chunk1..32/default16,
reusing N2.8 control and stopping immediately on convergence/budget/failure.
Separate T/Q input buffers; unchanged cap16/scaling/total QR budget/deflation/
shift/fill/separation/residual gates.128 retained groups,74 control oracles,
100 chunk boundaries, four long-chunk failures, seven public option refusals
and six full CPU eigenpair fixtures pass. Complete results/counters bitwise
exact across chunks1/2/8/16/32 on fixtures, boundary chunks exact vs repeated1.
Copies8/20/35->1/2/3 c64,11/26/46->1/2/3 c128; whole ratios1.034/1.046/0.984
and1.040/1.047/0.951 at4/9/16 are mixed, with16 regressing. Host eig0.224–0.426ms
remains much faster. Padded-Q variant passes but does not improve timing; retained
in WS/recovery/solver-n29-padded, original restored/verified against passed hashes.
12 guards ACK/600s/unchanged dmesg;76 protected runtime/source/recovery hashes
unchanged. Two one-unit hipcc/ccache20GiB builds; no Torch/ROCm DSO/image/prefix
metadata/kernel/boot/default/.105 mutation. Prior source/binary/oracle retained
WS/recovery/solver-n29-before and WS/helpers/schur_n28_reference.py. [Report and
reproduction](bc250_solver_results/N29.md), raw WS/logs/solver-*n29*. No active
job/container/guard; disk~1.09GiB free. No general convergence/gradient/batching/
compile/graph/production GPU eig/model speed contract. Python CPU fault remains
unresolved/separate from fixed VM-heap/CSR. Next N2.10 split GEHRD/reflector-Q
preparation and prototype bounded device Q accumulation, preserving native GEHRD,
Python/N2.8/N2.9 controls/cap16/fixed gates/production host LAPACK default.

## N2.8 bounded device Schur control — 2026-10-18

TheRock code67e5d1ef8 adds explicit device_control backend: trailing deflations,
nearest/exceptional complex shift, reused cooperative QR/similarity and finite/
128eps fill status in one HIP launch. At most one QR/launch, budget+1 launches,
one five-int status copy/launch; unchanged cap16/dtypes/scaling/budget/deflation/
separation/recovery residual/refusal rules. Default reference and production host
LAPACK unchanged.128 retained groups,74 direct state/boundary/root/failure/stream
oracle cases and six CPU eigenpair/paired fixtures pass. Reference matches N2.7
exactly on final six fixtures. Initial direct T/Q comparison ignored nonunique QR
phases; phase-aligned similarity and independent reconstruction fix harness with
unchanged bounds, no solver change. Failed log retained. Seven guards ACK/600s/
unchanged dmesg. Schur1.80–2.73x, whole candidate1.56–2.21x versus N2.7 in fixtures;
whole8.88–20.35ms still much slower than host full eig0.23–0.42ms. Order16 scalar
reads159/207->8 excludes explicit35/47 status copies; not a total-sync count.
70 protected runtime/source/recovery/metadata hashes unchanged; one-unit hipcc/
ccache20GiB, no prefix metadata/DSO/Torch/image/kernel/boot/default/.105 mutation.
Prior source/binary WS/recovery/solver-n28-before; immutable N2.7 oracle retained.
[Report/reproduction/evidence](bc250_solver_results/N28.md), raw WS/logs/solver-*n28*.
No active job/container/guard; disk~1.09GiB free. No general convergence/gradient/
batching/compile/graph/production GPU eig or universal/model/LoRA performance
claim. Python CPU fault unresolved/separate from fixed VM-heap/CSR. Next N2.9
bounded multi-iteration chunk to reduce status copies, preserving exact QR budget,
terminal/refusal semantics, Python/N2.8 oracle/fixed gates/cap16/host default.

## N2.7 cooperative device QR/similarity — 2026-10-18

TheRock code100f43e2a adds explicit device-step Schur backend, complex64/128
order2..16:32-thread HIP shared-memory Householder QR and full similarity/Q
update. Reference backend remains default; final six fixtures match prior N2.6
outputs/counts exactly. Normalization, iteration/deflation/fill, separation and
recovery residual/refusal gates unchanged.28 direct CPU QR/invariant/stream cases,
128 unchanged N2.3–5 groups and six complete eigenpair fixtures pass; budget0/1
partial/refusal contracts and profiler kernel evidence retained. Correct serial
prototype was slow; source/binary retained before cooperative rewrite.
Paired three-sample A1/B1/B2/A2 Schur1.24–1.46x in six fixtures; device10–36ms
still far slower than host full eig0.22–0.40ms. Separate instrumented order16
preparation~6.1ms, QR3.38/6.41ms, remaining checks/scaling/control22.46/28.24ms;
forced sync phase sums are diagnostic, not kernel-only/model/general gains.
Eight guards ACK/600s/unchanged dmesg;64 prior protected source/runtime hashes
unchanged. One-unit targeted hipcc build/ccache20GiB; no prefix metadata/ROCm
DSO/Torch/image/kernel/boot/default/.105 change. No active job/container/guard;
disk~1.09GiB free. Prior helper WS/helpers/schur_n26_reference.py; serial recovery
WS/recovery/solver-n27-serial. [Report/reproduction/evidence](bc250_solver_results/N27.md),
raw WS/logs/solver-*n27*. No production native eig, batching/gradient/general
convergence contract. Python CPU fault stays unresolved/separate from fixed
VM-heap/CSR. Next N2.8 bounded device shift/deflation/status to reduce measured
control overhead, preserving Python oracle/fixed gates/cap16/host eig default.

## N2.6 small native eig profile/recovery vectorization — 2026-10-16

Initial order16 recovery~37ms/376 Torch scalar reads; chose all-column rowwise
GPU back-substitution, n-1 instead quadratic row updates. Checked global gap
covers active pivots, masked inactive pivots set1; homogeneous scaling/final
finite/residual checks retained. Scalar reads4/9/16:34/124/376 ->18;107 unchanged
N2.4/N2.5 groups pass. Six synchronized wall-clock A1/B1/B2/A2 fixtures with CPU
references show recovery1.53–8.44x and whole1.07–1.72x improvements, raw3-sample
passes/controls retained. Native whole16–55ms remains far slower than host0.20–
0.40ms; no default integration/model/general speed claim. All logs identical,
guards ACKed, no active jobs. [Report/source/evidence](bc250_solver_results/N26.md).
No native compile, Torch/DSO/prefix/kernel/boot change; prior helper retained at
WS/recovery/solver-n26-before. Existing CPU fault/MIOpen issue remain open. Next
N2.7 split Schur preparation/QR-loop cost and bounded device-side iteration to
reduce remaining measured launch/scalar synchronization overhead.
