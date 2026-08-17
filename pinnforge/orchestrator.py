"""The block loop — a deterministic process, no model in the driver's seat.

The framework this replaces asked a main agent to dispatch block subagents,
count wall-seconds and decide when a block was finished. That worked, but it
put judgement where only bookkeeping was needed: whether a block is done is
a question the filesystem already answers.

So the loop here is plain Python. Per block:

1. pick the next id (or re-adopt a crashed one),
2. start one harness CLI in the run root and wait for it to exit,
3. read the evidence — summary written? budget spent? a real eval logged? —
   and either accept the block or dispatch a continuation,
4. append usage, regenerate the summary, move on.

Every transition is checkpointed to `state.json`, so an interrupted run is
resumed by re-reading it rather than by reconstructing intent.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path

from pinnforge import charter, integrity, paths
from pinnforge.config import RunConfig
from pinnforge.run import layout, ledger
from pinnforge.runtime import registry as runtime_registry
from pinnforge.runtime.base import AgentHandle, AgentUsage
from pinnforge.task import contract
from pinnforge.types import BlockState, EvalRecord, RunState, best_score, charged_seconds, utcnow

logger = logging.getLogger(__name__)

# A block gets this many dispatch segments before the orchestrator gives up
# on it. Segments are cheap (a resumed session keeps its context) but an
# agent that dies instantly would otherwise spin forever.
MAX_SEGMENTS = 6


def _usage_of(runtime, log_path: Path) -> AgentUsage:
    """Accounting from the harness's own log, if it keeps any.

    A harness that does not implement `extract_usage`, or whose log is
    unreadable, costs the run a dash in the summary — never a crash at the
    point where a block has already finished its work.
    """
    fn = getattr(runtime, "extract_usage", None)
    if fn is None:
        return AgentUsage()
    try:
        return fn(log_path)
    except (OSError, ValueError):
        logger.warning("could not read usage from %s", log_path)
        return AgentUsage()


class EnvironmentFailure(RuntimeError):
    """GPU or driver is gone — the one excuse for stopping instead of retrying."""


@dataclass
class BlockVerdict:
    done: bool
    spent: float
    budget: float
    has_summary: bool
    scored_evals: int

    @property
    def only_summary_missing(self) -> bool:
        return not self.has_summary and self.spent_enough and self.scored_evals >= 1

    @property
    def spent_enough(self) -> bool:
        return self.spent >= self.budget


def verify_block(run: Path, block_id: str, cfg: RunConfig, per_run_wall: float) -> BlockVerdict:
    """Is the block finished? Exactly the old three-part check, mechanised.

    Budget counts as spent once less than one full run's wall remains —
    otherwise a block would be resumed to burn a remainder too small to hold
    an evaluation.
    """
    recs = EvalRecord.read_all(paths.evals_path(run, block_id))
    spent = charged_seconds(recs, paths.budget_path(run, block_id))
    has_summary = paths.summary_path(run, block_id).is_file()
    scored = sum(1 for r in recs if r.scored)

    # b00 is the control node: `baseline.py` measured as-is by anchor.ensure(),
    # not a block that spends a budget or writes a summary. Its evidence is the
    # measurement, so that is all it is judged on. Holding it to the budget
    # would send the loop off to "finish" an anchor that is already complete —
    # and so would requiring a summary, because a cached anchor need not carry
    # one: only a task whose b00 was once written up by an agent has a `b00.md`
    # to restore, and the rest would look unfinished forever.
    if block_id == "b00":
        return BlockVerdict(
            done=scored >= 1,
            spent=spent,
            budget=0.0,
            has_summary=has_summary,
            scored_evals=scored,
        )

    threshold = max(0.0, cfg.sandbox.wall_budget_s - per_run_wall)
    return BlockVerdict(
        done=has_summary and spent >= threshold and scored >= 1,
        spent=spent,
        budget=threshold,
        has_summary=has_summary,
        scored_evals=scored,
    )


def gpu_healthy(gpus: list[int]) -> bool:
    """Cheap liveness probe; a dead driver must not look like a lazy agent."""
    if not shutil.which("nvidia-smi"):
        return True  # nothing to check against; assume the operator knows
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=index", "--format=csv,noheader"],
            capture_output=True,
            text=True,
            timeout=30,
            check=False,
        )
    except (subprocess.SubprocessError, OSError):
        return False
    if out.returncode != 0:
        return False
    visible = {line.strip() for line in out.stdout.splitlines() if line.strip()}
    return all(str(g) in visible for g in gpus)


class Orchestrator:
    def __init__(self, run: Path, cfg: RunConfig, state: RunState, root: Path | None = None):
        self.run = run
        self.cfg = cfg
        self.state = state
        self.root = root or paths.project_root()
        self.runtime = runtime_registry.get_runtime(cfg.agent.runtime)
        self.model = runtime_registry.resolve_model(self.runtime, cfg.agent.model)
        self.per_run_wall = contract.train_time(run / "task") or 0.0

    # ──────────────────────────── plumbing ────────────────────────────

    def _save(self) -> None:
        self.state.save(paths.state_path(self.run))

    def _block_env(self) -> dict[str, str]:
        env = {
            "FORGE_WALL_BUDGET": f"{self.cfg.sandbox.wall_budget_s:.0f}",
            "PINNFORGE_ROOT": str(self.root),
            "PINNFORGE_RUN": str(self.run),
        }
        env.update(self.cfg.sandbox.env)
        return env

    def _log_path(self, block_id: str) -> Path:
        d = paths.logs_dir(self.run)
        d.mkdir(parents=True, exist_ok=True)
        idx = len(list(d.glob(f"{block_id}.*.log")))
        return d / f"{block_id}.{idx}.log"

    def _dispatch(self, block_id: str, prompt: str, resume_session: str | None) -> AgentHandle:
        bs = self.state.blocks.setdefault(block_id, BlockState(block_id=block_id))
        bs.status = "running"
        bs.runtime = self.runtime.name
        bs.model = self.model
        bs.attempts += 1
        bs.started_at = bs.started_at or utcnow()
        self._save()

        paths.block_dir(self.run, block_id).mkdir(parents=True, exist_ok=True)
        return self.runtime.start(
            block_id=block_id,
            cwd=self.run,
            prompt=prompt,
            model=self.model,
            log_path=self._log_path(block_id),
            env=self._block_env(),
            command=self.cfg.agent.command or None,
            runtime_options=self.cfg.agent.runtime_options,
            max_turns=self.cfg.agent.max_turns,
            resume_session_id=resume_session,
            command_prefix=self.cfg.sandbox.command_prefix or None,
        )

    def _newest_mtime(self, block_id: str) -> float:
        newest = 0.0
        for p in [
            paths.block_dir(self.run, block_id),
            *paths.block_dir(self.run, block_id).glob("*"),
            paths.summary_path(self.run, block_id),
        ]:
            try:
                newest = max(newest, p.stat().st_mtime)
            except OSError:
                continue
        return newest

    def _await(self, handle: AgentHandle, block_id: str) -> None:
        """Block until the agent exits, logging stalls as they happen.

        A quiet block is not necessarily a stuck one — a full evaluation is
        several minutes of silence by design — so a stall is reported, never
        acted on. The exit itself is the signal the loop waits for.

        An interrupt is the exception. The agent runs in its own session, so
        a terminal's Ctrl-C reaches this process and not the block: left
        alone it keeps editing `blocks/bNN/`, keeps charging the budget, and
        keeps its training worker on the card holding the GPU lock — and
        `run resume` would then put a second agent on the same workspace.
        The group is signalled before the interrupt is allowed to propagate,
        SIGINT first so a harness that checkpoints gets the chance to leave a
        resumable session behind.
        """
        stall_s = self.cfg.stall_minutes * 60
        warned = False
        try:
            while handle.alive:
                if handle.wait(timeout=self.cfg.poll_seconds) is not None:
                    break
                quiet = time.time() - self._newest_mtime(block_id)
                if quiet > stall_s and not warned:
                    logger.warning(
                        "block %s quiet for %.0f min (still running)", block_id, quiet / 60
                    )
                    warned = True
                elif quiet <= stall_s:
                    warned = False
        except BaseException:
            logger.warning("interrupted — stopping block %s and its process tree", block_id)
            handle.interrupt()
            raise
        handle.close()

    def _record_segment(self, handle: AgentHandle, block_id: str, started: float) -> int:
        """Book what this dispatch cost. Called on every exit path, interrupt included.

        The GPU seconds survive on their own — `eval.py` writes them to
        `.budget` as it charges them — but the model spend lives only in the
        harness's log until it is read out here. Booking it after the wait
        returned normally meant a block interrupted mid-segment left its whole
        spend out of the ledger: an hour of tokens and cost, with the summary
        printing a dash where the money went, next to GPU seconds that proved
        the work had happened.
        """
        elapsed_ms = int((time.time() - started) * 1000)
        usage = _usage_of(self.runtime, handle.log_path)
        bs = self.state.blocks[block_id]
        bs.session_id = self.runtime.extract_session_id(handle.log_path) or bs.session_id
        bs.duration_ms += elapsed_ms
        # An alias is not a model. Two runs a generation apart both say
        # "opus"; only the resolved id says which one actually ran.
        bs.model_resolved = usage.model or bs.model_resolved
        self._save()
        ledger.append_usage(
            self.run, block_id, self.runtime.name, self.model, elapsed_ms, usage
        )

    def _check_integrity(self, block_id: str, segment: int, manifest: integrity.Manifest) -> list:
        """Verify the manifest and record the verdict. Every exit path, interrupt included.

        A dispatch that ended in an interrupt could have altered the record
        before it was stopped, and a segment with a manifest but no verdict is
        a hole in the artefact `run audit` reads: it counts verdicts, so an
        unverified dispatch makes the run look smaller and fully checked rather
        than partly unchecked.
        """
        violations = integrity.verify(self.run, manifest)
        integrity.record_result(self.run, block_id, segment, violations)
        if violations:
            for v in violations:
                logger.error("block %s violated the record — %s", block_id, v)
            self.state.blocks[block_id].exit_reason = (
                f"integrity: {len(violations)} violation(s) in segment {segment}"
            )
        return violations

    # ──────────────────────────── the loop ────────────────────────────

    def run_blocks(self, n: int) -> list[str]:
        """Run `n` blocks serially. Returns the ids that finished."""
        finished: list[str] = []
        self.state.status = "running"
        self._save()
        try:
            for _ in range(n):
                block_id = self._select_block()
                if self._run_one(block_id):
                    finished.append(block_id)
                ledger.write_run_summary(self.run)
            self.state.status = "finished"
        finally:
            if self.state.status == "running":
                self.state.status = "stopped"
            self._save()
            ledger.write_run_summary(self.run)
        return finished

    def _select_block(self) -> str:
        """Next id, or the highest existing one if it never finished.

        Re-adopting beats allocating a fresh id: the crashed workspace holds
        evaluations that were already paid for in GPU time.

        b00 is never a candidate. It is installed by `anchor.ensure()` and is
        the control node every other block is measured against — dispatching an
        agent onto it would spend one of the run's blocks rewriting the
        yardstick.
        """
        ids = [b for b in layout.existing_block_ids(self.run) if b != "b00"]
        if ids and not verify_block(self.run, ids[-1], self.cfg, self.per_run_wall).done:
            return ids[-1]
        return layout.next_block_id(self.run)

    def _run_one(self, block_id: str) -> bool:
        crashed = paths.block_dir(self.run, block_id).is_dir() and any(
            paths.block_dir(self.run, block_id).iterdir()
        )
        prompt = charter.block_prompt(block_id, self.run, crashed=crashed)
        resume_session: str | None = None
        last_progress: tuple[int, int] | None = None

        for segment in range(MAX_SEGMENTS):
            if not gpu_healthy(self.cfg.sandbox.gpus):
                raise EnvironmentFailure(
                    f"GPUs {self.cfg.sandbox.gpus} are not visible to nvidia-smi; "
                    "stopping instead of retrying"
                )

            # Bracket the dispatch: hash everything the block must leave
            # alone, then check it after. Detection, not prevention — see
            # pinnforge/integrity.py for why a jail is the wrong first move.
            manifest = integrity.snapshot(self.run, block_id, self.root)
            integrity.save(self.run, manifest, segment)

            started = time.time()
            handle = self._dispatch(block_id, prompt, resume_session)
            try:
                self._await(handle, block_id)
            except BaseException:
                # An interrupt is a normal end state — the README promises
                # Ctrl-C, a reboot and a dead harness all end the same way —
                # so the segment's accounting has to survive one. The GPU
                # seconds do already, because eval.py writes them to `.budget`
                # as it charges them; the model spend lives only in the
                # harness's log until it is read out here, and a block
                # interrupted mid-segment used to leave an hour of tokens and
                # cost out of the ledger entirely.
                self._record_segment(handle, block_id, started)
                self._check_integrity(block_id, segment, manifest)
                bs = self.state.blocks[block_id]
                bs.status = "interrupted"
                # an integrity verdict, if there was one, outranks this
                bs.exit_reason = bs.exit_reason or f"interrupted during segment {segment}"
                self._save()
                ledger.write_run_summary(self.run)
                raise
            self._record_segment(handle, block_id, started)
            violations = self._check_integrity(block_id, segment, manifest)
            bs = self.state.blocks[block_id]

            verdict = verify_block(self.run, block_id, self.cfg, self.per_run_wall)
            if verdict.done:
                bs.status = "done"
                bs.completed_at = utcnow()
                # "complete" used to overwrite the integrity verdict, so a block
                # that altered another's record still read as clean in state.json
                if not violations:
                    bs.exit_reason = "complete"
                self._save()
                self._relay(block_id, verdict)
                return True

            # A segment that bought nothing means the agent considers itself
            # finished, and the same prompt to the same session will get the
            # same answer. Resuming again is not persistence, it is a spin:
            # ldc_4's b02 stopped 104 s short of the threshold with its summary
            # written and 23 scored evaluations, and the loop re-dispatched it
            # four times in 42 seconds before giving up and calling it failed.
            progress = (round(verdict.spent), verdict.scored_evals)
            if progress == last_progress:
                return self._settle_stalled(block_id, verdict, segment)
            last_progress = progress

            # Continue the same session when the harness gave us a token to
            # resume with; otherwise a fresh agent picks up the workspace,
            # which is why the charter makes the workspace — not the
            # conversation — the block's memory.
            resume_session = bs.session_id
            if verdict.only_summary_missing:
                prompt = charter.summary_repair_prompt(block_id)
            else:
                prompt = charter.resume_prompt(
                    block_id, verdict.spent, verdict.budget, verdict.has_summary
                )
            logger.info(
                "block %s incomplete after segment %d (%.0f/%.0f s, summary %s) — resuming",
                block_id,
                segment + 1,
                verdict.spent,
                verdict.budget,
                "present" if verdict.has_summary else "missing",
            )

        bs = self.state.blocks[block_id]
        bs.status = "failed"
        bs.exit_reason = f"did not finish in {MAX_SEGMENTS} segments"
        bs.completed_at = utcnow()
        self._save()
        logger.error("block %s abandoned after %d segments", block_id, MAX_SEGMENTS)
        return False

    def _settle_stalled(self, block_id: str, verdict: BlockVerdict, segment: int) -> bool:
        """A block whose agent will not go further. Say which kind it is.

        `failed` has to mean "produced nothing usable", or the summary lies
        about the run: ldc_4's b02 was recorded as failed while holding the
        best score in the run, a written summary and 23 scored evaluations,
        purely because it stopped 104 s short of the budget threshold.

        The budget still decides `done`, because comparability rests on blocks
        having had the same GPU time. What changes is that a block which spent
        almost all of it and wrote up what it found is `short` — usable
        evidence that fell shy of the bar — and not a failure.
        """
        bs = self.state.blocks[block_id]
        usable = verdict.has_summary and verdict.scored_evals >= 1
        bs.status = "short" if usable else "failed"
        bs.completed_at = utcnow()
        shortfall = max(0.0, verdict.budget - verdict.spent)
        bs.exit_reason = bs.exit_reason or (
            f"agent stopped {shortfall:.0f} s short of the budget and made no "
            f"progress when resumed (segment {segment + 1})"
        )
        self._save()
        logger.warning(
            "block %s made no progress on resume — stopping at %.0f/%.0f s, %d eval(s), "
            "summary %s",
            block_id,
            verdict.spent,
            verdict.budget,
            verdict.scored_evals,
            "present" if verdict.has_summary else "missing",
        )
        if usable:
            self._relay(block_id, verdict)
        return False

    def _relay(self, block_id: str, verdict: BlockVerdict) -> None:
        recs = EvalRecord.read_all(paths.evals_path(self.run, block_id))
        best = best_score(recs)
        shown = format(best, ".5g") if best is not None else "no finite score"
        print(
            f"{block_id} done — best {shown}, "
            f"wall {verdict.spent:.0f}/{self.cfg.sandbox.wall_budget_s:.0f} s"
        )
